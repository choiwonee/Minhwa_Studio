import os
import sys
# UI 테마 라이브러리
import qdarktheme

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QMessageBox, QSlider, QComboBox, QGroupBox, QScrollArea, QTabWidget
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QFont

# 프로젝트 루트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 기존 유틸 및 모델 모듈 임포트
import utils.common as COM
import utils.depth_postprocess as DP
from models.depth_estimator import depth_estimator
from utils.gui_utils import ImageCanvas, GenericWorker, FloatingToolBar, ProcessingOverlay
# 설정 로더 추가
from utils.config_loader import config

# ==============================================================================
# Main Effect 2.5D Application Window
# ==============================================================================
class Effect25DProcessor:
    """ 2.5D 변환 로직을 담당하는 클래스 (UI와 로직 분리) """
    def __init__(self):
        self.original_image = None    # RGB 0~1 float
        self.original_uint8 = None    # RGB 0~255 uint8 (표시용)
        self.depth_map = None         # Depth 0~1 float
        self.depth_model_loaded = False
    
    def load_image(self, path):
        img_np = COM.load_image_numpy(path)
        if img_np is None:
            return None
        self.original_uint8 = img_np
        self.original_image = (img_np.astype(np.float32) / 255.0)
        return img_np

    def estimate_depth(self, model_id, progress_callback=None, **kwargs):
        if self.original_uint8 is None:
            raise ValueError("Image not loaded")
        
        if progress_callback: progress_callback(10)
        depth_estimator.load_model(model_id)
        
        if progress_callback: progress_callback(50)
        depth = depth_estimator.estimate_depth(self.original_uint8)
        self.depth_map = depth
        
        if progress_callback: progress_callback(100)
        return depth
    
    def run_pipeline(self, params, progress_callback=None, **kwargs):
        if self.original_image is None or self.depth_map is None:
            return None

        # [단계 1] Normal Map
        if progress_callback: progress_callback(10)
        normal_map = DP.depth_to_normal(self.depth_map, normal_scale=params['normal_scale'])

        # [단계 2] Lighting
        if progress_callback: progress_callback(30)
        if params['auto_light']:
            img_gray = cv2.cvtColor(self.original_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            light_dir_vec = DP.estimate_light_vector(self.depth_map, img_gray)
            light_dir_vec = np.array(light_dir_vec, dtype=np.float32)
        else:
            light_dir_vec = np.array(params['light_dir'], dtype=np.float32)
            norm = np.linalg.norm(light_dir_vec)
            if norm > 0: light_dir_vec /= norm
        
        light_dir_str = DP.classify_light_direction(light_dir_vec)

        # [단계 3] Shading
        if progress_callback: progress_callback(50)
        shade_image = DP.phong_shading(
            normal_map, light_dir_vec, 
            ambient=params['ambient'], diffuse_k=params['diffuse'], 
            specular_k=params['specular'], shininess=params['shininess']
        )
        shade_image = np.clip(shade_image, 1e-6, 1.0) ** params['shading_gamma']

        h, w = self.original_image.shape[:2]
        shade_2d = cv2.resize(shade_image, (w, h), interpolation=cv2.INTER_LINEAR)
        shade_3d = np.repeat(shade_2d[:, :, np.newaxis], 3, axis=-1) if shade_2d.ndim == 2 else shade_2d

        # [단계 4] Shadow Diffusion
        if progress_callback: progress_callback(70)
        
        auto = light_dir_str
        shadow_direction = ""
        if "up" in auto: shadow_direction += "down"
        elif "down" in auto: shadow_direction += "up"
        if "left" in auto: shadow_direction += "right"
        elif "right" in auto: shadow_direction += "left"
        if shadow_direction == "": shadow_direction = "center"

        shadow_diffused = DP.depth_shadow_diffusion(
            self.original_image, self.depth_map,
            strength=params['shadow_strength'], grad_sigma=params['shadow_grad_sigma'],
            grad_spread=params['shadow_grad_spread'], depth_spread=params['shadow_depth_spread'],
            offset=params['shadow_pixel_offset'], direction=shadow_direction
        )

        # [단계 5] Blending & Post-process
        if progress_callback: progress_callback(90)
        composite = DP.blend_image(shadow_diffused, shade_3d, mode=params['blend_mode'], alpha=params['blend_alpha'])
        
        if params['contrast_boost'] > 0:
            composite = DP.depth_contrast_boost(composite, self.depth_map, strength=params['contrast_boost'], in_place=True)
        if params['shadow_boost'] > 0:
            composite = DP.depth_shadow_boost(composite, self.depth_map, strength=params['shadow_boost'], in_place=True)
        if params['highlight_boost'] > 0:
            composite = DP.highlight_boost(composite, strength=params['highlight_boost'], in_place=True)

        if progress_callback: progress_callback(100)
        return (composite * 255).astype(np.uint8)


class Effect25DApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2.5D Relief Studio")
        
        # 화면 지원 해상도 분석
        screen_geo = QApplication.primaryScreen().availableGeometry()
        screen_w = screen_geo.width()
        screen_h = screen_geo.height()

        # 기준: 일반적인 노트북 해상도(FHD 미만)나 작은 태블릿, 가로 1400 이하(예: 1366x768)거나 세로 850 이하인 경우
        if screen_w <= 1400 or screen_h <= 850: # [Low-Res 모드] 화면을 꽉 채워 버튼 짤림 방지
            self.resize(screen_w, screen_h) # 혹시 모르니 사이즈 설정
            self.setWindowState(Qt.WindowMaximized) # 최대화 상태로 시작
        else: # [High-Res 모드] 81% 크기로 중앙 정렬
            target_w = int(screen_w * 0.81)
            target_h = int(screen_h * 0.81)
            self.resize(target_w, target_h)
            # 화면 중앙 이동
            frame_geo = self.frameGeometry()
            frame_geo.moveCenter(screen_geo.center())
            self.move(frame_geo.topLeft())

        self.processor = Effect25DProcessor()
        self.worker = None
        self.pipeline_worker = None
        self.is_pipeline_running = False
        self.force_fit_result = False
        
        self.params = {
            'normal_scale': 90.0, 'auto_light': False, 'light_dir': [0.6, -0.25, 0.65],
            'ambient': 0.15, 'diffuse': 0.95, 'specular': 0.12, 'shininess': 28,
            'shading_gamma': 2.75, 'shadow_strength': 0.185, 'shadow_grad_sigma': 1.55,
            'shadow_grad_spread': 7.0, 'shadow_depth_spread': 8.0, 'shadow_pixel_offset': 5.0, 
            'blend_mode': 'overlay', 'blend_alpha': 0.36,
            'contrast_boost': 0.12, 'shadow_boost': 0.05, 'highlight_boost': 0.25
        }
        
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.trigger_pipeline)

        self.init_ui()
        self.setup_floating_toolbar()
        self.setup_shortcuts()
        
        self.toggle_param_controls(False)
        self.loading_overlay = ProcessingOverlay(self.centralWidget())

    def init_ui(self):
        """ UI 초기화 및 위젯 배치
            - 좌측 패널: 설정 컨트롤 (Input, Shading, Shadow, Post-process)
            - 우측 패널: 탭 뷰어 (Result, Depth, Original)
        """
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Left Panel ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        # 고정 너비(380px) 대신 유동적인 너비 설정. 작은 화면에서는 300px까지 줄어들 수 있게 하여 우측 캔버스 공간 확보
        scroll_area.setMinimumWidth(300)
        scroll_area.setMaximumWidth(400)

        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setContentsMargins(10, 10, 10, 10)
        control_layout.setSpacing(15)

        # ----------------------------------------------------------------------
        # 1. Input & Model (안전장치 추가)
        # ----------------------------------------------------------------------
        group_file = QGroupBox("Input & Model")
        layout_file = QVBoxLayout(group_file)
        
        # Load Button
        self.btn_load = QPushButton("Open Image")
        self.btn_load.clicked.connect(self.load_image)
        layout_file.addWidget(self.btn_load)
        
        # Model Combo
        lbl_model = QLabel("Model:")
        lbl_model.setStyleSheet("color: #bdc3c7;")
        layout_file.addWidget(lbl_model)
        
        self.combo_model = QComboBox()

        # Config 로드 중 에러가 발생해도 UI가 깨지지 않도록 try-except 처리
        try:
            # 기존 get_models_by_group 제거하고 현행 config_loader의 get_models 메서드 호출
            depth_models = config.get_models("depth")
            default_idx = 0
            current_idx = 0
            
            if depth_models:
                for key, info in depth_models.items():
                    # 안전하게 값 가져오기
                    name = info.get('short_name', key)
                    rid = info.get('repo_id', key)
                    self.combo_model.addItem(name, rid)
                    
                    if info.get('is_default'):
                        default_idx = current_idx
                    current_idx += 1
            else:
                self.combo_model.addItem("Depth Anything V2 (Fallback)", "depth-anything/Depth-Anything-V2-Large-hf")
            
            self.combo_model.setCurrentIndex(default_idx)
            
        except Exception as e:
            print(f"[UI Error] Config loading failed: {e}")
            self.combo_model.addItem("Depth Anything V2 (Error Fallback)", "depth-anything/Depth-Anything-V2-Large-hf")

        layout_file.addWidget(self.combo_model)
        
        # Run Button
        self.btn_run_depth = QPushButton("Generate Depth Map")
        self.btn_run_depth.clicked.connect(self.run_depth_estimation)
        self.btn_run_depth.setStyleSheet("background-color: #e67e22; color: white; font-weight: bold; padding: 8px;")
        layout_file.addWidget(self.btn_run_depth)
        
        # 그룹박스를 메인 컨트롤 레이아웃에 추가 (이 부분이 실행 보장되어야 함)
        control_layout.addWidget(group_file)

        # ----------------------------------------------------------------------
        # 2. Shading Parameters
        # ----------------------------------------------------------------------
        self.group_shade = QGroupBox("Shading Parameters")
        layout_shade = QVBoxLayout(self.group_shade)
        layout_shade.setSpacing(5)
        
        self.spin_normal = self.add_slider(layout_shade, "Normal Scale", 1, 200, 90, 'normal_scale')
        self.spin_ambient = self.add_slider(layout_shade, "Ambient", 0, 100, 15, 'ambient', scale=0.01)
        self.spin_diffuse = self.add_slider(layout_shade, "Diffuse", 0, 100, 95, 'diffuse', scale=0.01)
        self.spin_specular = self.add_slider(layout_shade, "Specular", 0, 100, 12, 'specular', scale=0.01)
        self.spin_shine = self.add_slider(layout_shade, "Shininess", 1, 100, 28, 'shininess')
        control_layout.addWidget(self.group_shade)

        # ----------------------------------------------------------------------
        # 3. Shadow Effects
        # ----------------------------------------------------------------------
        self.group_shadow = QGroupBox("Shadow Effects")
        layout_shadow = QVBoxLayout(self.group_shadow)
        layout_shadow.setSpacing(5)

        self.spin_shadow_str = self.add_slider(layout_shadow, "Strength", 0, 100, 18, 'shadow_strength', scale=0.01)
        self.spin_shadow_spread = self.add_slider(layout_shadow, "Spread", 1, 30, 7, 'shadow_grad_spread')
        self.spin_shadow_offset = self.add_slider(layout_shadow, "Offset", 0, 50, 5, 'shadow_pixel_offset')
        control_layout.addWidget(self.group_shadow)

        # ----------------------------------------------------------------------
        # 4. Blending & Post-process
        # ----------------------------------------------------------------------
        self.group_post = QGroupBox("Blending & Post-process")
        layout_post = QVBoxLayout(self.group_post)
        layout_post.setSpacing(5)

        lbl_blend = QLabel("Blend Mode:")
        lbl_blend.setStyleSheet("color: #bdc3c7;")
        layout_post.addWidget(lbl_blend)

        self.combo_blend = QComboBox()
        self.combo_blend.addItems(["softlight", "overlay", "multiply", "screen"])
        self.combo_blend.setCurrentText("overlay")
        self.combo_blend.currentTextChanged.connect(lambda t: self.update_param('blend_mode', t))
        layout_post.addWidget(self.combo_blend)
        
        self.spin_blend_alpha = self.add_slider(layout_post, "Blend Alpha", 0, 100, 36, 'blend_alpha', scale=0.01)
        self.spin_contrast = self.add_slider(layout_post, "Contrast Boost", 0, 100, 12, 'contrast_boost', scale=0.01)
        self.spin_highlight = self.add_slider(layout_post, "Highlight Boost", 0, 100, 25, 'highlight_boost', scale=0.01)
        control_layout.addWidget(self.group_post)

        # Save Button
        self.btn_save = QPushButton("Save Result")
        self.btn_save.clicked.connect(self.save_result)
        self.btn_save.setStyleSheet("background-color: #27ae60; color: white; height: 40px; font-weight: bold; font-size: 14px;")
        control_layout.addWidget(self.btn_save)
        
        control_layout.addStretch(1)
        scroll_area.setWidget(control_widget)
        main_layout.addWidget(scroll_area)

        # --- Right Panel ---
        self.tabs = QTabWidget()
        self.canvas_result = ImageCanvas(self)
        self.canvas_depth = ImageCanvas(self)
        self.canvas_original = ImageCanvas(self)

        for canvas in [self.canvas_result, self.canvas_depth, self.canvas_original]:
            canvas.set_mode("view")
            canvas.set_show_crosshair(False)
        
        self.tabs.addTab(self.canvas_depth, "Depth Map")
        self.tabs.addTab(self.canvas_original, "Original")
        self.tabs.addTab(self.canvas_result, "Result (2.5D)")
        
        main_layout.addWidget(self.tabs, 1)

    # 창 크기 변경 이벤트
    def resizeEvent(self, event):
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.resize(self.centralWidget().size())
        super().resizeEvent(event)

    def toggle_param_controls(self, enabled):
        self.group_shade.setEnabled(enabled)
        self.group_shadow.setEnabled(enabled)
        self.group_post.setEnabled(enabled)
        self.btn_save.setEnabled(enabled)

    # 슬라이더 추가 헬퍼 함수
    def add_slider(self, layout, label_text, min_val, max_val, init_val, param_key, scale=1.0):
        lbl = QLabel(label_text)
        layout.addWidget(lbl)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(int(init_val))
        
        val_label = QLabel(f"{init_val * scale:.2f}")
        val_label.setFixedWidth(40)
        val_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        def on_value_changed(v):
            real_val = v * scale
            val_label.setText(f"{real_val:.2f}")
            self.params[param_key] = real_val
            
        slider.valueChanged.connect(on_value_changed)
        slider.sliderReleased.connect(self.trigger_pipeline)
        
        h_layout = QHBoxLayout()
        h_layout.addWidget(slider)
        h_layout.addWidget(val_label)
        layout.addLayout(h_layout)
        return slider

    def update_param(self, key, value):
        self.params[key] = value
        self.trigger_pipeline()

    def setup_floating_toolbar(self):
        """ 2.5D 화면 툴바 설정: 마스킹 화면과 동일한 디자인(반투명/회색) 적용 """
        
        def get_current_canvas():
            current_widget = self.tabs.currentWidget()
            if isinstance(current_widget, ImageCanvas):
                return current_widget
            return None

        self.toolbar = FloatingToolBar(self.tabs)
        
        # [스타일 통일] 마스킹 화면의 툴바와 똑같은 CSS 적용
        self.toolbar.setStyleSheet("""
            /* 1. 패널 배경 (반투명 다크) */
            QWidget {
                background-color: rgba(28, 28, 28, 240);
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
            
            /* 2. 섹션 라벨 */
            QLabel {
                color: #7f8c8d; font-weight: 800; font-size: 10px;
                margin-top: 8px; margin-bottom: 2px; letter-spacing: 1px;
                background: transparent; border: none;
            }
            
            /* 3. 구분선 */
            QFrame[frameShape="4"] {
                color: rgba(255, 255, 255, 0.1);
                margin: 5px 0px;
            }
            
            /* 4. 버튼 공통 스타일 (반투명 회색) */
            QPushButton {
                background-color: rgba(255, 255, 255, 0.06);
                color: #ccc;
                border: none;
                border-radius: 6px;
                padding: 8px;
                font-weight: 600;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.12);
                color: white;
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 0.04);
                padding-top: 9px;
            }
                                   
            /* 5. Pan(Move) 버튼 전용 스타일 (체크 상태) */
            /* 체크되어도 주황색이 아니라, 밝은 회색 테두리로 은은하게 강조 */
            QPushButton[mode="pan"]:checked {
                background-color: rgba(255, 255, 255, 0.15); 
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.4);
            }
        """)

        self.toolbar.add_section_label("VIEW CONTROL")
        
        # 1. Move (Pan) 버튼 - 디자인 변경 (손바닥 -> ✥ Move)
        self.btn_pan = QPushButton("Move  ✥") 
        self.btn_pan.setCheckable(True)
        self.btn_pan.setProperty("mode", "pan") # CSS 선택자용 속성
        self.btn_pan.setToolTip("Pan View (Space or H)")
        self.btn_pan.clicked.connect(self.toggle_pan_mode)
        
        self.toolbar.layout().addWidget(self.btn_pan)
        self.toolbar.add_separator()

        # 2. Fit Window
        btn_fit = QPushButton("Fit Window")
        btn_fit.clicked.connect(lambda: get_current_canvas() and get_current_canvas().fit_to_window())
        self.toolbar.layout().addWidget(btn_fit)

        # 3. 1:1 Actual
        btn_actual = QPushButton("1:1 Actual")
        btn_actual.clicked.connect(lambda: get_current_canvas() and get_current_canvas().set_actual_size())
        self.toolbar.layout().addWidget(btn_actual)
        
        self.toolbar.layout().addStretch(1)
        self.toolbar.show()
        self.toolbar.move(20, 20)
        self.toolbar.raise_()

    def toggle_pan_mode(self, checked):
        """ Pan 모드 전환 (스타일은 CSS가 처리하므로 로직만 남김) """
        mode = "pan" if checked else "view"

        self.canvas_result.set_mode(mode)
        self.canvas_depth.set_mode(mode)
        self.canvas_original.set_mode(mode)

    def setup_shortcuts(self):
        action_pan = QAction(self)
        action_pan.setShortcut(Qt.Key_Space)
        action_pan.triggered.connect(self.btn_pan.click)
        self.addAction(action_pan)
        action_hand = QAction(self)
        action_hand.setShortcut(Qt.Key_H)
        action_hand.triggered.connect(self.btn_pan.click)
        self.addAction(action_hand)

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if fname:
            with COM.SuppressStderr():
                img = self.processor.load_image(fname)
            if img is None:
                QMessageBox.warning(self, "Error", "이미지를 불러올 수 없습니다.")
                return
            self.tabs.setCurrentIndex(1)
            self.toggle_param_controls(False)
            self.processor.depth_map = None
            self.canvas_depth.set_image(None)
            for canvas in [self.canvas_original, self.canvas_result]:
                canvas.set_image(img)
                QTimer.singleShot(50, canvas.fit_to_window)

    def run_depth_estimation(self):
        if self.worker:
            try:
                if self.worker.isRunning(): return
            except RuntimeError: self.worker = None
        
        if self.processor.original_uint8 is None:
            QMessageBox.warning(self, "Error", "이미지를 먼저 로드해주세요.")
            return
        
        self.set_ui_busy(True)
        self.force_fit_result = True
        
        self.loading_overlay.set_message("Generating Depth", "Estimating depth map...\nThis requires heavy calculation.")
        self.loading_overlay.resize(self.centralWidget().size())
        self.loading_overlay.show()
        self.loading_overlay.raise_()

        model_id = self.combo_model.currentData()
        if not model_id: model_id = self.combo_model.currentText()
        
        self.worker = GenericWorker(self.processor.estimate_depth, model_id=model_id)
        self.worker.signal_finished.connect(self.on_depth_finished)
        self.worker.error.connect(self.on_worker_error)
        self.worker.start()
        
    def on_depth_finished(self, depth_map):
        self.set_ui_busy(False)
        self.loading_overlay.hide()
        if self.worker:
            self.worker.deleteLater()
            self.worker = None

        depth_vis = (depth_map * 255).astype(np.uint8)
        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2RGB)
        self.canvas_depth.set_image(depth_vis)
        self.tabs.setCurrentIndex(0)
        QTimer.singleShot(50, self.canvas_depth.fit_to_window) 
        self.toggle_param_controls(True)
        self.trigger_pipeline()
        
    def trigger_pipeline(self):
        if self.processor.depth_map is None or self.is_pipeline_running: return
        self.is_pipeline_running = True
        self.loading_overlay.set_message("Applying Effects", "Rendering 2.5D lighting & shadows...")
        self.loading_overlay.resize(self.centralWidget().size())
        self.loading_overlay.show()
        self.loading_overlay.raise_()
        QTimer.singleShot(50, self._start_pipeline_worker)

    def _start_pipeline_worker(self):
        if self.pipeline_worker:
            try:
                if self.pipeline_worker.isRunning(): return
            except RuntimeError: self.pipeline_worker = None
        
        self.pipeline_worker = GenericWorker(self.processor.run_pipeline, params=self.params.copy())
        self.pipeline_worker.signal_progress.connect(self.on_pipeline_progress)
        self.pipeline_worker.signal_finished.connect(self.on_pipeline_finished)
        self.pipeline_worker.error.connect(self.on_pipeline_error)
        self.pipeline_worker.start()

    def on_pipeline_progress(self, value):
        step_desc = "Processing..."
        if value < 20: step_desc = "Generating Normal Map"
        elif value < 40: step_desc = "Calculating Lighting Direction"
        elif value < 60: step_desc = "Applying Phong Shading"
        elif value < 80: step_desc = "Rendering Soft Shadows"
        elif value < 100: step_desc = "Blending & Finalizing"
        self.loading_overlay.set_message("Applying Effects", f"{step_desc} ({value}%)")
        
    def on_pipeline_finished(self, result_img):
        self.is_pipeline_running = False
        if self.pipeline_worker:
            self.pipeline_worker.deleteLater()
            self.pipeline_worker = None
        self.loading_overlay.hide()
        if result_img is not None:
            if self.force_fit_result:
                self.canvas_result.set_image(result_img)
                QTimer.singleShot(10, self.canvas_result.fit_to_window)
                self.force_fit_result = False
            else:
                self.canvas_result.set_image_keep_view(result_img)
            if self.tabs.currentIndex() != 2: self.tabs.setCurrentIndex(2)

    def on_pipeline_error(self, e):
        self.is_pipeline_running = False
        if self.pipeline_worker:
            self.pipeline_worker.deleteLater()
            self.pipeline_worker = None
        self.loading_overlay.hide()
        print(f"Pipeline Error: {e}")
        QMessageBox.critical(self, "Processing Error", f"효과 적용 중 오류가 발생했습니다.\n{e}")

    def save_result(self):
        if self.canvas_result.image is None:
            QMessageBox.warning(self, "Warning", "저장할 이미지가 없습니다.")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Save Image", "output.png", "Images (*.png)")
        if fname:
            save_img = cv2.cvtColor(self.canvas_result.image, cv2.COLOR_RGB2BGR)
            success = COM.imwrite_unicode(fname, save_img)
            if success: QMessageBox.information(self, "Saved", f"Saved to:\n{fname}")
            else: QMessageBox.critical(self, "Error", "파일 저장에 실패했습니다.")

    def set_ui_busy(self, busy):
        self.btn_run_depth.setEnabled(not busy)
        self.btn_load.setEnabled(not busy)
        self.btn_run_depth.setText("Processing..." if busy else "Generate Depth Map")

    def on_worker_error(self, e):
        self.set_ui_busy(False)
        self.loading_overlay.hide()
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        QMessageBox.critical(self, "Error", f"작업 중 오류 발생:\n{str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 모던 다크 테마 적용 (Launcher와 동일하게 맞춤)
    qdarktheme.setup_theme("dark")
    # 폰트 설정 (선택 사항 - Launcher와 통일감)
    font = QFont("Malgun Gothic", 10)
    app.setFont(font)
    
    win = Effect25DApp()
    win.show()
    sys.exit(app.exec())