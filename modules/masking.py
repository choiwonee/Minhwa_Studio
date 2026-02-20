import json, os, sys
# UI 테마 라이브러리
import qdarktheme

import cv2
import numpy as np

from datetime import datetime
from pathlib import Path
from PIL import Image

# --- PySide6 Imports ---
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QFileDialog, QPushButton, QWidget, QVBoxLayout,
    QHBoxLayout, QComboBox, QMessageBox, QSpinBox, QCheckBox, QStatusBar, QProgressBar,
    QSplitter, QTableWidget, QTableWidgetItem, QAbstractItemView, QTabWidget, QListWidget, QListWidgetItem,
    QButtonGroup, QGroupBox, QSlider, QScrollBar, QGridLayout, QDoubleSpinBox, 
)
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import (QAction, QFont, QIcon, QImage, QPixmap) 

# 상위 폴더(utils) 참조를 위한 경로 설정(중요)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.config_loader import config # Config Loader
from utils.common import qimage_from_ndarray, save_image_file, get_output_dir, SuppressStderr # 공통 유틸 함수
from utils.masking_postprocess import remove_white_noise_component # opencv 노이즈 제거 함수
from models.sam_estimator import SAM2Estimator # SAM 모델 로직
from utils.gui_utils import GenericWorker, ImageCanvas, FloatingToolBar, ProcessingOverlay

# Local Helper Functions (Mask Processing)
def polygon_to_mask(poly_pts, w, h):
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(poly_pts, dtype=np.int32).reshape((-1,1,2))
    cv2.fillPoly(mask, [pts], 1)
    return mask

def sample_points_in_mask(mask, npoints=30):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((0,2), dtype=int)
    if len(xs) <= npoints:
        return np.stack([xs, ys], axis=1)
    idx = np.linspace(0, len(xs)-1, npoints).astype(int)
    return np.stack([xs[idx], ys[idx]], axis=1)


# ==============================================================================
# Main Masking Application Window
# ==============================================================================
class SAMGuiApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # 1. 윈도우 설정
        self.setWindowTitle("SAM2 - Qt Mask Extractor (Tabbed UI)")
        
        # 화면 지원 해상도 분석
        screen_geo = QApplication.primaryScreen().availableGeometry()
        screen_w = screen_geo.width()
        screen_h = screen_geo.height()

        # 기준: 일반적인 노트북 해상도(FHD 미만)나 작은 태블릿
        if screen_w <= 1400 or screen_h <= 850: 
            self.resize(screen_w, screen_h) 
            self.setWindowState(Qt.WindowMaximized) 
        else: 
            target_w = int(screen_w * 0.81)
            target_h = int(screen_h * 0.81)
            self.resize(target_w, target_h)
            frame_geo = self.frameGeometry()
            frame_geo.moveCenter(screen_geo.center())
            self.move(frame_geo.topLeft())

        # 2. 데이터 & 모델 초기화
        self.sam_models_dict = config.get_models("sam2")
        self.sam_estimator = SAM2Estimator()
        self.hw_info = self.sam_estimator.hw_info
        
        self.loaded_model_id = None
        self.img_path = None
        self.image = None
        self.current_mask = None
        self.result_dir = None
        self.save_sub_folder = None
        self.result_counter = 0
        self.worker = None
        self.final_merged_mask_data = None
        self.is_image_set_to_sam = False
        self.candidate_masks = None
        self.candidate_scores = None

        # 3. 공통 UI 요소 (상태바, 타이머) 초기화
        self._init_status_bar()
        self._init_timers()

        # 4. 메인 UI 구성
        self._init_ui()

        # 5. 후처리 (시그널, 단축키, 로그, 오버레이)
        self.canvas.on_selection_done = self.on_selection_done
        self.canvas.on_point_added = self.on_point_added
        self.canvas.on_selection_cancelled = self.on_selection_cancelled
        self.canvas.on_undo = self.on_undo
        
        self.setup_floating_pan_button()
        self.setup_shortcuts()
        
        self.log(f"Device: {self.sam_estimator.device}, Precision: {self.sam_estimator.dtype}")
        self.loading_overlay = ProcessingOverlay(self.centralWidget())

    def _init_status_bar(self):
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        
        self.progress_text_label = QLabel("")
        self.progress_text_label.setStyleSheet("color: #3498db; font-weight: bold; padding-right: 10px;")
        self.progress_text_label.setVisible(False)
        self.status.addPermanentWidget(self.progress_text_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setVisible(False)
        self.status.addPermanentWidget(self.progress_bar)

    def _init_timers(self):
        self.progress_text_timer = QTimer()
        self.progress_text_timer.timeout.connect(self.toggle_progress_text)
        self.progress_text_state = 0

        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self.toggle_button_blink)
        self.blink_state = False
        self.blinking_button = None

    def _init_ui(self):
        """ QSplitter를 사용하여 반응형 레이아웃 구성 """
        root = QWidget()
        self.setCentralWidget(root)
        
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(0)

        # 1. 메인 가로 분할기 (좌측 패널 | 우측 패널)
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(4)

        # 좌측 패널 (Canvas + Controls)
        left_panel = self._setup_left_panel()
        self.main_splitter.addWidget(left_panel)

        # 우측 패널 (Gallery, Log)
        right_panel = self._setup_right_panel()
        self.main_splitter.addWidget(right_panel)

        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setCollapsible(0, False)

        main_layout.addWidget(self.main_splitter)

    def _setup_left_panel(self):
        v_splitter = QSplitter(Qt.Vertical)
        v_splitter.setHandleWidth(4)

        # 1. Canvas Area (상단)
        canvas_wrapper = QWidget()
        canvas_grid = QGridLayout(canvas_wrapper)
        canvas_grid.setContentsMargins(0, 0, 0, 0)
        canvas_grid.setSpacing(0)

        self.canvas = ImageCanvas(self)
        self.canvas.set_mode("box") 
        self.canvas.set_show_crosshair(True)
        self.canvas.sig_view_changed.connect(self.sync_scrollbars_from_canvas)
        
        # 캔버스 최소 높이를 400px -> 200px로 대폭 축소하여 낮은 해상도(768p)에서도 하단 컨트롤 탭이 보일 공간을 확보함
        self.canvas.setMinimumHeight(200)

        self.scroll_h = QScrollBar(Qt.Horizontal)
        self.scroll_v = QScrollBar(Qt.Vertical)
        self.scroll_h.setStyleSheet("QScrollBar:horizontal { height: 14px; }")
        self.scroll_v.setStyleSheet("QScrollBar:vertical { width: 14px; }")
        self.scroll_h.valueChanged.connect(self.on_scrollbar_action)
        self.scroll_v.valueChanged.connect(self.on_scrollbar_action)

        canvas_grid.addWidget(self.canvas, 0, 0)
        canvas_grid.addWidget(self.scroll_v, 0, 1)
        canvas_grid.addWidget(self.scroll_h, 1, 0)
        
        v_splitter.addWidget(canvas_wrapper)
        
        # 툴바 초기화
        self.toolbar = FloatingToolBar(self.canvas)
        self._configure_toolbar()

        # 2. Controls Area (하단) - QTabWidget 적용
        self.controls_tabs = QTabWidget()
    
        # --- Tab 1: Main Action ---
        tab_main = QWidget()
        layout_main = QVBoxLayout(tab_main)
        layout_main.setContentsMargins(10, 10, 10, 10)
        layout_main.setSpacing(10)

        layout_main.addWidget(self._create_input_group())

        self.candidate_widget = QGroupBox("Candidate Masks (Multi-mask)")
        self.candidate_layout = QHBoxLayout(self.candidate_widget)
        self.candidate_layout.setContentsMargins(10, 10, 10, 10)
        
        self.cand_bg = QButtonGroup(self)
        self.cand_btns = []
        for i in range(3):
            btn = QPushButton(f"Mask {i+1}")
            btn.setCheckable(True)
            self.cand_bg.addButton(btn, i)
            self.candidate_layout.addWidget(btn)
            self.cand_btns.append(btn)
        
        self.cand_bg.idClicked.connect(self.on_candidate_selected)
        self.candidate_widget.setVisible(False)
        layout_main.addWidget(self.candidate_widget)
        
        layout_main.addStretch(1)

        # --- Tab 2: Advanced Tools ---
        tab_tools = QWidget()
        layout_tools = QVBoxLayout(tab_tools)
        layout_tools.setContentsMargins(10, 10, 10, 10)
        layout_tools.setSpacing(15)

        layout_tools.addWidget(self._create_noise_group())

        lasso_group = QGroupBox("Lasso Utilities")
        lasso_group.setStyleSheet("QGroupBox { font-weight: bold; color: #ecf0f1; border: 1px solid #555; border-radius: 6px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        lasso_layout = QHBoxLayout(lasso_group)
        
        btn_save_lasso = QPushButton("Save Lasso JSON"); btn_save_lasso.clicked.connect(self.save_lasso)
        btn_save_lasso.setStyleSheet("padding: 6px;")
        btn_load_lasso = QPushButton("Load Lasso JSON"); btn_load_lasso.clicked.connect(self.load_lasso)
        btn_load_lasso.setStyleSheet("padding: 6px;")
        
        lasso_layout.addWidget(btn_save_lasso)
        lasso_layout.addWidget(btn_load_lasso)
        layout_tools.addWidget(lasso_group)

        layout_tools.addStretch(1)

        self.controls_tabs.addTab(tab_main, "Main Actions")
        self.controls_tabs.addTab(tab_tools, "Advanced Tools")

        v_splitter.addWidget(self.controls_tabs)
        
        # 초기 비율 설정 (캔버스에 더 많은 공간 할당)
        v_splitter.setStretchFactor(0, 4)
        v_splitter.setStretchFactor(1, 1)
        v_splitter.setCollapsible(1, True)

        return v_splitter

    def _setup_right_panel(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.right_tabs = QTabWidget()
        layout.addWidget(self.right_tabs)
        
        # Tab 1: Mask Gallery
        self.gallery_tab = QWidget()
        gal_layout = QVBoxLayout(self.gallery_tab)
        
        self.gallery_list = QListWidget()
        self.gallery_list.setViewMode(QListWidget.IconMode)
        self.gallery_list.setIconSize(QSize(100, 100))
        self.gallery_list.setResizeMode(QListWidget.Adjust)
        self.gallery_list.setSpacing(10)
        self.gallery_list.setSelectionMode(QAbstractItemView.NoSelection)
        gal_layout.addWidget(self.gallery_list, 2)
        
        # Gallery Controls
        ctrl_row = QHBoxLayout()
        
        btn_sel_all = QPushButton("Select All")
        btn_sel_all.clicked.connect(lambda: self.set_gallery_check_state(True))
        self._setup_btn_feedback(btn_sel_all)
        
        btn_unsel = QPushButton("Unselect")
        btn_unsel.clicked.connect(lambda: self.set_gallery_check_state(False))
        self._setup_btn_feedback(btn_unsel)
        
        btn_del = QPushButton("Delete Selected")
        btn_del.clicked.connect(self.delete_selected_masks)
        self._setup_btn_feedback(btn_del)
        
        ctrl_row.addWidget(btn_sel_all)
        ctrl_row.addWidget(btn_unsel)
        ctrl_row.addWidget(btn_del)
        gal_layout.addLayout(ctrl_row)
        
        # Merge 저장 버튼 그룹 (Merged Save / Inverted Save)
        save_btn_layout = QHBoxLayout()
        save_btn_layout.setSpacing(5)
        
        # 버튼 1: 기존 Merged Save (배경 검정, 객체 흰색)
        self.btn_merge = QPushButton("Merged Save")
        self.btn_merge.setToolTip("Save merged mask (Object=White, BG=Black)")
        self.btn_merge.setStyleSheet("background-color: #d35400; color: white; font-weight: bold; padding: 8px;")
        self.btn_merge.clicked.connect(self.merge_selected_masks)
        self._setup_btn_feedback(self.btn_merge) # 시각적 색상 변화 피드백 설정
        save_btn_layout.addWidget(self.btn_merge)

        # 버튼 2: Inverted Save (배경 흰색, 객체 검정)
        self.btn_save_inverted = QPushButton("Inverted Save")
        self.btn_save_inverted.setToolTip("Save INVERTED mask (Object=Black, BG=White)")
        self.btn_save_inverted.setStyleSheet("background-color: #8e44ad; color: white; font-weight: bold; padding: 8px;")
        self.btn_save_inverted.clicked.connect(self.save_inverted_mask)
        self._setup_btn_feedback(self.btn_save_inverted) # 시각적 색상 변화 피드백 설정
        save_btn_layout.addWidget(self.btn_save_inverted)

        gal_layout.addLayout(save_btn_layout)
        
        self.preview_canvas = QLabel("Final Merge Preview")
        self.preview_canvas.setAlignment(Qt.AlignCenter)
        self.preview_canvas.setMinimumSize(QSize(150, 150))
        self.preview_canvas.setStyleSheet("QLabel { border: 2px solid #3498db; background-color: #2c3e50; color: #ecf0f1; font-weight: bold; }")
        gal_layout.addWidget(self.preview_canvas, 1) 
        
        self.btn_save_rgba = QPushButton("Save Isolated Object (RGBA)")
        self.btn_save_rgba.setToolTip("Save transparent PNG")
        self.btn_save_rgba.setStyleSheet("QPushButton { background-color: #27ae60; color: white; font-weight: bold; padding: 10px; } QPushButton:disabled { background-color: #95a5a6; }")
        self.btn_save_rgba.clicked.connect(self.save_rgba_result)
        self._setup_btn_feedback(self.btn_save_rgba) # 시각적 색상 변화 피드백 설정
        self.btn_save_rgba.setEnabled(False)
        gal_layout.addWidget(self.btn_save_rgba)
        
        self.right_tabs.addTab(self.gallery_tab, "Mask Gallery")
        
        # Tab 2: Activity Log
        self.log_tab = QWidget()
        log_layout = QVBoxLayout(self.log_tab)
        self.log_table = QTableWidget()
        self.log_table.setColumnCount(2)
        self.log_table.setHorizontalHeaderLabels(["Time", "Event"])
        self.log_table.horizontalHeader().setStretchLastSection(True)
        self.log_table.verticalHeader().setVisible(False)
        self.log_table.setAlternatingRowColors(True)
        log_layout.addWidget(self.log_table)
        self.right_tabs.addTab(self.log_tab, "Activity Log")
        
        return container

    def _create_input_group(self):
        """ [UI] 1. Input & Actions 그룹박스 생성 (Single Row Layout) """
        group = QGroupBox("Input && Actions")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 15, 10, 10)
        layout.setSpacing(5)

        # 시스템 정보 (상단 작게 표시)
        hw_desc = self.hw_info['description']
        info_color = "#ff6b6b" if self.hw_info.get("is_pascal", False) else "#2ecc71"
        self.lbl_hw_info = QLabel(f"🖥 System: {hw_desc}")
        self.lbl_hw_info.setStyleSheet(f"color: {info_color}; font-size: 11px; font-weight: bold; margin-bottom: 2px;")
        layout.addWidget(self.lbl_hw_info)

        # 메인 컨트롤 로우 (한 줄 배치)
        row_main = QHBoxLayout()
        row_main.setSpacing(8) 

        # [Left Group] Open / Model / Load
        self.btn_open = QPushButton("Open")
        self.btn_open.setToolTip("Open Image")
        self.btn_open.setIcon(QIcon.fromTheme("document-open"))
        self.btn_open.setFixedWidth(80)
        self.btn_open.clicked.connect(self.open_image)
        self._setup_btn_feedback(self.btn_open)
        row_main.addWidget(self.btn_open)

        self.model_selector = QComboBox()
        self.model_selector.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.model_selector.setMinimumWidth(120) 
        
        try:
            model_list = list(self.sam_models_dict.values())
            sorted_models = sorted(model_list, key=lambda x: x.get('short_name', ''))
            default_idx = 0
            for i, m in enumerate(sorted_models):
                name = m.get('short_name', 'Unknown')
                rid = m.get('repo_id', '') 
                self.model_selector.addItem(f"{name}", userData=rid)
                if m.get('is_default'): default_idx = i
            self.model_selector.setCurrentIndex(default_idx)
        except Exception:
            self.model_selector.addItem("SAM2 Large", userData="facebook/sam2-hiera-large")

        self.model_selector.currentIndexChanged.connect(self.reset_model_status)
        row_main.addWidget(self.model_selector)

        # Precision (Auto/FP32/FP16)
        self.combo_precision = QComboBox()
        self.combo_precision.addItems(["Auto", "float32", "float16", "bfloat16"])
        self.combo_precision.setFixedWidth(70)
        self.combo_precision.setCurrentIndex(0)
        row_main.addWidget(self.combo_precision)

        self.btn_load_sam = QPushButton("Load")
        self.btn_load_sam.setFixedWidth(60)
        self.btn_load_sam.clicked.connect(self.load_sam_model)
        self._setup_btn_feedback(self.btn_load_sam)
        row_main.addWidget(self.btn_load_sam)
        
        self.lbl_model_status = QLabel("Wait")
        self.lbl_model_status.setAlignment(Qt.AlignCenter)
        self.lbl_model_status.setMinimumWidth(60)
        self.lbl_model_status.setStyleSheet("color: #95a5a6; border: 1px solid #bdc3c7; border-radius: 4px; padding: 4px; background-color: #ecf0f1; font-size: 11px;")
        row_main.addWidget(self.lbl_model_status)
        
        # ---------------------------------------------------------
        # [Spacer] 좌측(로드) 그룹과 우측(액션) 그룹 사이 간격 벌리기
        # ---------------------------------------------------------
        row_main.addStretch(1) 

        # [Right Group] Extract / Save / Actions
        # Manual Mode 체크박스
        self.chk_manual_mode = QCheckBox("Manual")
        self.chk_manual_mode.setToolTip("Check this to bypass SAM AI and use raw shapes as mask (Brush/Box/Lasso).")
        self.chk_manual_mode.setStyleSheet("font-weight: bold; color: #e67e22; margin-right: 5px;")
        row_main.addWidget(self.chk_manual_mode)
        
        self.btn_extract = QPushButton("Extract")
        self.btn_extract.setToolTip("Extract Mask from Selection")
        self.btn_extract.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold; padding: 6px;")
        self.btn_extract.setFixedWidth(80)
        self.btn_extract.clicked.connect(self.extract_mask_current_selection)
        self._setup_btn_feedback(self.btn_extract)
        row_main.addWidget(self.btn_extract)

        self.btn_reset = QPushButton("Clear")
        self.btn_reset.setFixedWidth(60)
        self.btn_reset.clicked.connect(self.clear_selection)
        self._setup_btn_feedback(self.btn_reset)
        row_main.addWidget(self.btn_reset)

        btn_refine = QPushButton("Refine")
        btn_refine.setFixedWidth(60)
        btn_refine.clicked.connect(self.refine_with_points)
        self._setup_btn_feedback(btn_refine)
        row_main.addWidget(btn_refine)

        btn_save = QPushButton("Save")
        btn_save.setFixedWidth(60)
        btn_save.clicked.connect(self.save_mask)
        self._setup_btn_feedback(btn_save)
        row_main.addWidget(btn_save)

        btn_next = QPushButton("Save++")
        btn_next.setToolTip("Save & Clear (Next)")
        btn_next.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 6px;")
        btn_next.setFixedWidth(70)
        btn_next.clicked.connect(self.save_next)
        self._setup_btn_feedback(btn_next)
        row_main.addWidget(btn_next)
        
        layout.addLayout(row_main)
        return group
    
    def _create_noise_group(self):
        """ [UI] Mask Post-Processing 그룹박스 생성 """
        group = QGroupBox("Mask Post-Processing (Clean Noise)")

        layout = QHBoxLayout(group)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(20)
        
        self.chk_pixel_limit = QCheckBox("Use Pixel Limit")
        self.chk_pixel_limit.setChecked(True)
        layout.addWidget(self.chk_pixel_limit)
        
        layout.addWidget(QLabel("Min Ratio:"))
        self.spin_noise_ratio = QDoubleSpinBox()
        self.spin_noise_ratio.setRange(0, 0.1)
        self.spin_noise_ratio.setSingleStep(0.00001)
        self.spin_noise_ratio.setDecimals(5)
        self.spin_noise_ratio.setValue(0.0005)
        self.spin_noise_ratio.setFixedWidth(110)
        layout.addWidget(self.spin_noise_ratio)
        
        self.btn_clean_noise = QPushButton("Clean Noise")
        self.btn_clean_noise.setFixedWidth(130)
        self.btn_clean_noise.setStyleSheet("QPushButton { background-color: #8e44ad; color: white; font-weight: bold; padding: 6px; } QPushButton:hover { background-color: #9b59b6; }")
        self.btn_clean_noise.clicked.connect(self.apply_noise_removal)
        self._setup_btn_feedback(self.btn_clean_noise) # 시각적 색상 변화 피드백 설정
        layout.addWidget(self.btn_clean_noise)
        
        layout.addStretch(1)
        return group

    # -------------------------------------------------------------------------
    # 버튼 클릭 피드백 헬퍼 메서드
    # -------------------------------------------------------------------------
    def _setup_btn_feedback(self, btn: QPushButton):
        """ 버튼 클릭 시 0.4초간 텍스트를 노란색(#f1c40f)으로 변경하여 사용자가 눌렀음을 확실히 인지하게 함 """
        if not btn: return

        def flash_text():
            # 기존 스타일 백업
            original_style = btn.styleSheet()
            
            # 텍스트 색상만 노란색으로 강제 변경 (기존 스타일 뒤에 붙여서 오버라이딩)
            # border나 background 등 다른 속성은 유지됨
            btn.setStyleSheet(original_style + "; color: #f1c40f;")
            
            # 0.4초 후 원복
            QTimer.singleShot(400, lambda: btn.setStyleSheet(original_style))

        btn.clicked.connect(flash_text)

    def _configure_toolbar(self):
        """ 툴바 설정: 툴팁 적용 및 라벨 제거 """
        
        self.toolbar.setObjectName("MainToolbar")
        self.toolbar.setStyleSheet("""
            QWidget#MainToolbar {
                background-color: rgba(28, 28, 28, 240);
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
            QLabel[header="true"] {
                color: #7f8c8d; font-weight: 800; font-size: 10px;
                margin-top: 8px; margin-bottom: 2px; letter-spacing: 1px;
                background: transparent; border: none;
            }
            QFrame[frameShape="4"] {
                color: rgba(255, 255, 255, 0.1);
                margin: 5px 0px;
            }
            QSpinBox {
                background-color: rgba(0, 0, 0, 0.3); color: #eee;
                border: 1px solid #444; border-radius: 6px; padding: 5px;
            }
            QPushButton[simple_btn="true"] {
                background-color: rgba(255, 255, 255, 0.06);
                color: #ccc; border: none; border-radius: 6px;
                padding: 8px; font-weight: 600;
            }
            QPushButton[simple_btn="true"]:hover {
                background-color: rgba(255, 255, 255, 0.12); color: white;
            }
        """)

        def create_custom_option(text):
            container = QWidget()
            container.setObjectName("OptionContainer")
            container.setStyleSheet("""
                QWidget#OptionContainer {
                    background-color: rgba(28, 28, 28, 0.0); /* 투명 배경 */
                    border-radius: 6px;
                    border: none;
                }
                QWidget#OptionContainer:hover {
                    background-color: rgba(255, 255, 255, 0.12);
                }
            """)
            layout = QHBoxLayout(container)
            layout.setContentsMargins(6, 4, 6, 4)
            layout.setSpacing(10)
            
            btn = QPushButton("")
            btn.setCheckable(True)
            btn.setFixedSize(20, 20)
            btn.setCursor(Qt.PointingHandCursor)
            
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #bdc3c7; font-weight: bold; font-size: 11px; border: none; background: transparent;")
            lbl.setCursor(Qt.PointingHandCursor)
            
            layout.addWidget(btn)
            layout.addWidget(lbl)
            layout.addStretch(1)
            
            def label_mouse_press(event):
                if event.button() == Qt.LeftButton:
                    btn.animateClick()
            
            lbl.mousePressEvent = label_mouse_press
            
            def update_visual(checked):
                btn.setText("🗸" if checked else "")
                lbl.setStyleSheet(f"color: {'#fff' if checked else '#bdc3c7'}; border: none; background: transparent; font-weight: bold; font-size: 11px;")
            
            btn.toggled.connect(update_visual)
            update_visual(False)
            
            return container, btn

        # [SECTION 1] MODE
        lbl_mode = QLabel("MODE"); lbl_mode.setProperty("header", "true")
        self.toolbar.layout().addWidget(lbl_mode)
        
        self.mode_bg = QButtonGroup(self)
        
        row_box, self.rb_box = create_custom_option("Box")
        self.mode_bg.addButton(self.rb_box, 1)
        self.toolbar.layout().addWidget(row_box)
        
        row_lasso, self.rb_lasso = create_custom_option("Lasso")
        self.mode_bg.addButton(self.rb_lasso, 2)
        self.toolbar.layout().addWidget(row_lasso)
        
        row_point, self.rb_point = create_custom_option("Point")
        self.mode_bg.addButton(self.rb_point, 3)
        self.toolbar.layout().addWidget(row_point)
        
        # Point 모드 버튼 및 라벨에 컬러 툴팁 적용 (HTML 사용), Green: #2ecc71, Red: #e74c3c (다크 테마에서 잘 보이는 색상)
        tooltip_text = (
            "<b>Point Mode:</b><br>"
            "🖱️ Left Click: Include (<span style='color:#2ecc71; font-weight:bold;'>Green</span>)<br>"
            "🖱️ Right Click: Exclude (<span style='color:#e74c3c; font-weight:bold;'>Red</span>)"
        )
        self.rb_point.setToolTip(tooltip_text)
        row_point.setToolTip(tooltip_text) # 컨테이너에도 툴팁 적용
        
        # --- Brush Mode 버튼 및 슬라이더 ---
        self.toolbar.add_separator() # 구분선 추가

        row_brush, self.rb_brush = create_custom_option("Brush")
        brush_tooltip = (
            "<b>Brush Mode:</b><br>"
            "🖱️ Drag to paint manually.<br>"
            "Good for manual correction."
        )
        self.rb_brush.setToolTip(brush_tooltip)
        row_brush.setToolTip(brush_tooltip)
        
        self.mode_bg.addButton(self.rb_brush, 4) # ID 4
        self.toolbar.layout().addWidget(row_brush)

        # 브러쉬 사이즈 슬라이더 (평소에는 숨김)
        self.brush_size_container = QWidget()
        bs_layout = QHBoxLayout(self.brush_size_container)
        bs_layout.setContentsMargins(6, 0, 6, 0)
        
        lbl_sz = QLabel("Size:")
        lbl_sz.setStyleSheet("color: #bbb; font-size: 10px;")
        
        self.slider_brush = QSlider(Qt.Horizontal)
        self.slider_brush.setRange(5, 100) # 5px ~ 100px
        self.slider_brush.setValue(20)
        self.slider_brush.setFixedWidth(80)
        self.slider_brush.valueChanged.connect(self.canvas.set_brush_size)
        
        bs_layout.addWidget(lbl_sz)
        bs_layout.addWidget(self.slider_brush)
        
        self.brush_size_container.setVisible(False) 
        self.toolbar.layout().addWidget(self.brush_size_container)
        
        self.mode_bg.buttonToggled.connect(self.on_mode_radio_toggled)
        self.rb_box.setChecked(True)
        
        self.toolbar.add_separator()

        # [SECTION 2] OPTIONS
        lbl_opt = QLabel("OPTIONS"); lbl_opt.setProperty("header", "true")
        self.toolbar.layout().addWidget(lbl_opt)
        
        row_guide, self.chk_crosshair = create_custom_option("Show Guide")
        self.chk_crosshair.setChecked(True)
        self.chk_crosshair.toggled.connect(self.canvas.set_show_crosshair)
        self.toolbar.layout().addWidget(row_guide)
        
        row_multi, self.multimask_chk = create_custom_option("Multi-mask")
        self.toolbar.layout().addWidget(row_multi)
        
        self.toolbar.layout().addWidget(QLabel("Lasso Points:"))
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(10, 400)
        self.sample_spin.setValue(120)
        self.toolbar.layout().addWidget(self.sample_spin)
        self.toolbar.add_separator()

        # [SECTION 3] VIEW
        lbl_view = QLabel("VIEW"); lbl_view.setProperty("header", "true")
        self.toolbar.layout().addWidget(lbl_view)
        
        btn_fit = QPushButton("Fit Window")
        btn_fit.setProperty("simple_btn", "true")
        btn_fit.clicked.connect(self.canvas.fit_to_window)
        self.toolbar.layout().addWidget(btn_fit)
        
        btn_actual = QPushButton("1:1 Actual")
        btn_actual.setProperty("simple_btn", "true")
        btn_actual.clicked.connect(self.canvas.set_actual_size)
        self.toolbar.layout().addWidget(btn_actual)
        
        self.toolbar.layout().addStretch(1)
        self.toolbar.show()
        self.toolbar.move(20, 20)

    def setup_shortcuts(self):
        QAction(self, shortcut="1", triggered=lambda: self.rb_box.setChecked(True))
        QAction(self, shortcut="2", triggered=lambda: self.rb_lasso.setChecked(True))
        QAction(self, shortcut="3", triggered=lambda: self.rb_point.setChecked(True))
        QAction(self, shortcut="4", triggered=lambda: self.rb_brush.setChecked(True))
        QAction(self, shortcut="5", triggered=lambda: self.btn_pan_float.click())
        QAction(self, shortcut="H", triggered=lambda: self.btn_pan_float.click())
        QAction(self, shortcut="Space", triggered=self.btn_pan_float.click)
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_1:
            self.rb_box.setChecked(True)
        elif event.key() == Qt.Key_2:
            self.rb_lasso.setChecked(True)
        elif event.key() == Qt.Key_3:
            self.rb_point.setChecked(True)
        elif event.key() == Qt.Key_4:
            self.rb_brush.setChecked(True)
        elif event.key() == Qt.Key_5 or event.key() == Qt.Key_H:
            self.btn_pan_float.click() 
        super().keyPressEvent(event)

    def on_mode_radio_toggled(self, button, checked): 
        if not checked: return
        
        if hasattr(self, 'btn_pan_float') and self.btn_pan_float is not None:
            if self.btn_pan_float.isChecked():
                self.btn_pan_float.blockSignals(True) 
                self.btn_pan_float.setChecked(False)
                self.btn_pan_float.blockSignals(False)
            
        # 모드 변경 로직
        new_mode = "view"
        if button == self.rb_box: new_mode = "box"
        elif button == self.rb_lasso: new_mode = "lasso"
        elif button == self.rb_point: new_mode = "point"
        elif button == self.rb_brush: new_mode = "brush"
        
        self.change_mode(new_mode)

        # Brush 모드일 때만 슬라이더 보이기
        if hasattr(self, 'brush_size_container'):
            self.brush_size_container.setVisible(new_mode == "brush")
            
    def setup_floating_pan_button(self):
        self.btn_pan_float = QPushButton(self.canvas)
        self.btn_pan_float.setCheckable(True)
        self.btn_pan_float.setCursor(Qt.PointingHandCursor)
        self.btn_pan_float.setText("✥") 
        self.btn_pan_float.setToolTip("Pan View (Space or H)")
        self.btn_pan_float.setFixedSize(40, 40)
        self.btn_pan_float.setStyleSheet("""
            QPushButton {
                background-color: rgba(28, 28, 28, 240); 
                color: #bdc3c7; border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 8px; font-weight: bold; font-size: 20px;
                padding-bottom: 2px; outline: none;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.15); color: white;
                border-color: rgba(255, 255, 255, 0.3);
            }
            QPushButton:checked {
                background-color: rgba(255, 255, 255, 0.25); color: white;
                border: 1px solid rgba(255, 255, 255, 0.15); 
            }
            QPushButton:focus { border: 1px solid rgba(255, 255, 255, 0.15); }
        """)
        
        self.btn_pan_float.show()
        self.btn_pan_float.clicked.connect(self.on_floating_pan_toggled)

        original_resize = self.canvas.resizeEvent
        def new_resize_event(event):
            original_resize(event)
            self.update_pan_button_position()
        self.canvas.resizeEvent = new_resize_event
        self.update_pan_button_position()

    def update_pan_button_position(self):
        self.canvas.width()
        h = self.canvas.height()
        margin_left = 20
        margin_bottom = 20
        btn_h = self.btn_pan_float.height()
        self.btn_pan_float.move(margin_left, h - btn_h - margin_bottom)
        
    def on_floating_pan_toggled(self, checked):
        if checked:
            self.mode_bg.setExclusive(False)
            self.rb_box.setChecked(False)
            self.rb_lasso.setChecked(False)
            self.rb_point.setChecked(False)
            self.rb_brush.setChecked(False)
            self.mode_bg.setExclusive(True)
            self.change_mode("pan") 
        else:
            self.rb_box.setChecked(True)
            
    def _task_load_sam_model(self, model_id, precision, **kwargs): 
        if 'progress_callback' in kwargs: kwargs['progress_callback'](10)
        return self.sam_estimator.load_model(model_id, precision=precision) 
    
    def _task_run_prediction(self, predict_kwargs, need_encode_image=False, **kwargs): 
        if 'progress_callback' in kwargs: kwargs['progress_callback'](10)
        if need_encode_image:
            self.sam_estimator.set_image(self.image)
            if 'progress_callback' in kwargs: kwargs['progress_callback'](50)
        result = self.sam_estimator.predict(predict_kwargs)
        if 'progress_callback' in kwargs: kwargs['progress_callback'](100)
        return result
        
    def reset_model_status(self):
        current_selection_id = self.model_selector.currentData()
        if self.loaded_model_id is not None and current_selection_id == self.loaded_model_id:
            short_name = self.model_selector.currentText().split('(')[0].strip()
            self.lbl_model_status.setText(f"✔ Loaded: {short_name}")
            self.lbl_model_status.setStyleSheet("""
                QLabel {
                    color: #27ae60; border: 1px solid #2ecc71; 
                    background-color: #eafaf1; font-weight: bold; 
                    border-radius: 6px; padding: 4px 8px; font-size: 11px;
                }
            """)
        else:
            self.lbl_model_status.setText("Wait")
            self.lbl_model_status.setStyleSheet("""
                QLabel {
                    color: #95a5a6; border: 1px solid #bdc3c7; 
                    border-radius: 6px; padding: 4px 8px; 
                    background-color: #ecf0f1; font-size: 11px;
                }
            """)

    def load_sam_model(self):
        if self.worker is not None:
            QMessageBox.warning(self, "Busy", "Task running."); return
        
        model_id = self.model_selector.currentData()
        short_name = self.model_selector.currentText().split('(')[0].strip()
        if not model_id:
            QMessageBox.warning(self, "Error", "Model ID not found.")
            return

        prec_idx = self.combo_precision.currentIndex()
        selected_prec = ["auto", "float32", "float16", "bfloat16"][prec_idx]

        is_pascal = self.hw_info.get("is_pascal", False)
        if is_pascal and selected_prec in ["float16", "bfloat16"]:
            msg = (f"⚠️ Compatibility Warning\n\nYour GPU (Pascal) requires Float32.\n"
                   f"Proceed with '{selected_prec}'?")
            reply = QMessageBox.warning(self, "Hardware Compatibility", msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                self.combo_precision.setCurrentIndex(1)
                return

        self.start_button_blink(self.btn_load_sam)
        self.show_progress(f"Loading SAM model ({short_name})...")
        self.log(f"Loading SAM model ID: {model_id} (Prec: {selected_prec})...")
        
        self._temp_model_id = model_id
        self._temp_precision = selected_prec 

        self.toggle_loading(True, "Loading Model", f"Loading {short_name}...")
        QTimer.singleShot(50, self._start_sam_load_worker)
        
    def _start_sam_load_worker(self):
        self.worker = GenericWorker(self._task_load_sam_model, model_id=self._temp_model_id, precision=self._temp_precision)
        self.worker.signal_finished.connect(self._on_sam_model_loaded) 
        self.worker.error.connect(self._on_task_error)
        self.worker.start()
    
    def _on_sam_model_loaded(self, success):
        self.toggle_loading(False) 
        self.stop_button_blink(); 
        self.hide_progress(); 
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        if success:
            self.loaded_model_id = self._temp_model_id
            self.is_image_set_to_sam = False
            short_name = self.model_selector.currentText().split('(')[0].strip()
            self.status.showMessage(f"SAM loaded ({short_name}).", 3000)
            self.log(f"[OK] SAM loaded: {short_name}")
            self.lbl_model_status.setText(f"🗸 Loaded: {short_name}")
            self.lbl_model_status.setStyleSheet("QLabel { color: #27ae60; border: 1px solid #2ecc71; background-color: #eafaf1; font-weight: bold; border-radius: 6px; padding: 4px 8px; font-size: 11px; }")
        else:
            self.status.showMessage("Failed to load SAM.", 3000)
    
    def extract_mask_current_selection(self):
        # 1. 기본 상태 체크
        if self.worker is not None: 
            QMessageBox.warning(self, "Busy", "Task running."); return
        if self.image is None: 
            QMessageBox.warning(self, "No image", "Open an image first."); return
        
        # 모드 확인
        mode = self.canvas.mode
        if mode == "pan":
            # Pan 모드라도 기존에 선택된 데이터가 있으면 해당 모드로 간주
            if self.canvas.box_img is not None: mode = "box" 
            elif len(self.canvas.lasso_img) > 0: mode = "lasso"
            elif len(self.canvas.point_list) > 0: mode = "point"
            elif len(self.canvas.brush_strokes) > 0: mode = "brush"
            else:
                QMessageBox.warning(self, "No Selection", "선택된 영역이 없습니다."); return
        
        # 2. Manual Mode (No AI) 처리
        # Box, Lasso, Brush 모두 SAM 없이 즉시 마스크 생성 지원
        if self.chk_manual_mode.isChecked():
            mask = None
            
            if mode == "brush":
                mask = self.generate_brush_mask()
                if mask is None or np.max(mask) == 0:
                     QMessageBox.information(self, "Empty Brush", "브러쉬로 영역을 칠해주세요.")
                     return
            
            elif mode == "box":
                if self.canvas.box_img:
                    l, t, r, b = self.canvas.box_img
                    h, w = self.image.shape[:2]
                    mask = np.zeros((h, w), dtype=np.uint8)
                    # 박스 영역을 흰색(255)으로 채움
                    cv2.rectangle(mask, (int(l), int(t)), (int(r), int(b)), 255, -1)
                else:
                    QMessageBox.information(self, "No Box", "Draw a box first.")
                    return

            elif mode == "lasso":
                if len(self.canvas.lasso_img) >= 3:
                    h, w = self.image.shape[:2]
                    # polygon_to_mask는 0/1을 반환하므로 255를 곱해줌
                    mask = polygon_to_mask(self.canvas.lasso_img, w, h) * 255
                else:
                    QMessageBox.information(self, "No Lasso", "Draw a lasso first.")
                    return

            elif mode == "point":
                # 점(Point)은 AI 없이는 영역을 알 수 없으므로 Manual 지원 불가
                QMessageBox.warning(self, "Manual Mode", 
                                    "Point mode is not supported in Manual (No AI) mode.\n"
                                    "Please use Brush, Box, or Lasso for manual masking.")
                return

            # Manual 마스크 생성 성공 시 결과 처리 후 종료 (SAM 로직 스킵)
            if mask is not None:
                # from_sam=False 전달하여 플래그 오염 방지
                self._on_mask_extracted(mask, from_sam=False)
                self.log(f"[OK] Manual {mode.upper()} mask created (No AI).")
                return 

        # 3. AI 모델 로드 확인 (AI 모드일 경우 필수)
        if not self.sam_estimator.is_ready: 
            QMessageBox.warning(self, "SAM not loaded", 
                                "Please load SAM model first.\n"
                                "Or check 'Manual' to skip AI.")
            self.start_button_blink(self.btn_load_sam)
            return
        
        # 4. SAM 추론 파라미터 구성 (기존 AI 로직)
        predict_kwargs = {}
        try:
            if mode == "box":
                if self.canvas.box_img is None: QMessageBox.information(self, "No box", "Draw a box first."); return
                l,t,r,b = self.canvas.box_img; box = np.array([l,t,r,b])
                predict_kwargs = { "box": box.reshape(1, 4), "multimask_output": self.multimask_chk.isChecked() }
                
            elif mode == "lasso":
                poly = self.canvas.lasso_img[:]
                if not poly or len(poly)<3: QMessageBox.information(self, "No lasso", "Draw lasso."); return
                h, w = self.image.shape[:2]; poly_mask = polygon_to_mask(poly, w, h)
                pts = sample_points_in_mask(poly_mask, int(self.sample_spin.value()))
                predict_kwargs = { "point_coords": pts.astype(float), "point_labels": np.ones(len(pts), dtype=np.int32), "multimask_output": self.multimask_chk.isChecked() }
                
            elif mode == "point":
                pts = np.array([[x,y] for (x,y,l) in self.canvas.point_list], dtype=float)
                labels = np.array([l for (x,y,l) in self.canvas.point_list], dtype=int)
                predict_kwargs = { "point_coords": pts, "point_labels": labels, "multimask_output": self.multimask_chk.isChecked() }

            elif mode == "brush":
                # 브러쉬 AI 모드: 궤적을 힌트로 사용
                all_points = []
                for stroke in self.canvas.brush_strokes:
                    pts = stroke['points']
                    if len(pts) > 0:
                        step = max(1, len(pts) // 20)
                        sampled = pts[::step]
                        all_points.extend(sampled)
                        if pts[-1] not in sampled: all_points.append(pts[-1])

                if not all_points: QMessageBox.information(self, "Empty Brush", "브러쉬로 영역을 칠해주세요."); return

                pts_array = np.array(all_points, dtype=float)
                labels_array = np.ones(len(pts_array), dtype=np.int32)
                predict_kwargs = { "point_coords": pts_array, "point_labels": labels_array, "multimask_output": self.multimask_chk.isChecked() }

        except Exception as e:
            self.log(f"[ERROR] Prep failed: {e}"); return

        # 5. 워커 시작
        need_encode = not self.is_image_set_to_sam
        msg = "Encoding & Extracting..." if need_encode else "Extracting Mask..."
        self._temp_predict_kwargs = predict_kwargs
        self._temp_need_encode = need_encode
        self.toggle_loading(True, "Processing", f"{msg}")
        QTimer.singleShot(50, self._start_extract_worker)
    
    def _start_extract_worker(self):
        self.worker = GenericWorker(self._task_run_prediction, predict_kwargs=self._temp_predict_kwargs, need_encode_image=self._temp_need_encode)
        # 시그널 연결 시 from_sam=True 전달
        self.worker.signal_finished.connect(lambda r: self._on_mask_extracted(r, from_sam=True)) 
        self.worker.error.connect(self._on_task_error)
        self.worker.start()
    
    def refine_with_points(self):
        if self.worker:
            QMessageBox.warning(self, "Busy", "Task running."); return
        if not self.sam_estimator.is_ready:
            QMessageBox.warning(self, "SAM not loaded", "Load SAM first."); return
        if self.image is None:
            QMessageBox.warning(self, "No image", "Open image first."); return
        if not self.canvas.point_list:
            QMessageBox.information(self, "No points", "Click points."); return

        pts = np.array([[x,y] for (x,y,l) in self.canvas.point_list], dtype=float)
        labels = np.array([l for (x,y,l) in self.canvas.point_list], dtype=int)
        predict_kwargs = { "point_coords": pts, "point_labels": labels, "multimask_output": False }
        
        self.show_progress("Refining mask...")
        need_encode = not self.is_image_set_to_sam
        self.worker = GenericWorker(self._task_run_prediction, predict_kwargs=predict_kwargs, need_encode_image=need_encode)
        # 시그널 연결 시 from_sam=True 전달
        self.worker.signal_finished.connect(lambda r: self._on_mask_extracted(r, from_sam=True))
        self.worker.error.connect(self._on_task_error)
        self.worker.start()
        
    def _on_mask_extracted(self, mask_result, from_sam=False):
        self.toggle_loading(False)
        self.stop_button_blink()
        self.hide_progress()
        
        # SAM 모델을 통해 추출된 경우에만 '이미지 세팅됨' 플래그를 True로 설정
        if from_sam:
            self.is_image_set_to_sam = True
        
        if isinstance(mask_result, tuple):
            masks, scores = mask_result
            self.candidate_masks = masks
            self.candidate_scores = scores
            for i, btn in enumerate(self.cand_btns):
                if i < len(scores):
                    score_pct = scores[i] * 100
                    btn.setText(f"Mask {i+1} ({score_pct:.1f}%)")
                    btn.setVisible(True)
                else:
                    btn.setVisible(False)
            best_idx = int(np.argmax(scores))
            self.cand_btns[best_idx].setChecked(True)
            self.candidate_widget.setVisible(True) 
            self.current_mask = (masks[best_idx] > 0).astype(np.uint8) * 255
            self.log(f"[OK] Multi-mask candidates ready. Best: Mask {best_idx+1}")
        else:
            self.current_mask = mask_result
            self.candidate_widget.setVisible(False)
            self.candidate_masks = None
        
        self.canvas.set_overlay_mask(self.current_mask)
        self.status.showMessage("Mask extracted.", 3000)
        self.right_tabs.setCurrentIndex(1)
        
        if self.worker:
            self.worker.deleteLater()
            self.worker = None

    def on_candidate_selected(self, btn_id):
        if self.candidate_masks is not None and 0 <= btn_id < len(self.candidate_masks):
            self.current_mask = (self.candidate_masks[btn_id] > 0).astype(np.uint8) * 255
            self.canvas.set_overlay_mask(self.current_mask)
            self.log(f"Selected candidate: Mask {btn_id+1}")

    def update_mask_gallery(self):
        self.gallery_list.clear()
        if not self.result_dir or not self.result_dir.exists():
            return
        files = sorted(list(self.result_dir.glob("*.png")))
        for fpath in files:
            item = QListWidgetItem(fpath.name)
            item.setData(Qt.UserRole, str(fpath))
            pix = QPixmap(str(fpath))
            if not pix.isNull():
                icon = QIcon(pix.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                item.setIcon(icon)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.gallery_list.addItem(item)
        self.log(f"Gallery updated: {len(files)} files.")

    def set_gallery_check_state(self, checked: bool):
        state = Qt.Checked if checked else Qt.Unchecked
        for i in range(self.gallery_list.count()):
            self.gallery_list.item(i).setCheckState(state)

    def delete_selected_masks(self):
        items_to_delete = []
        for i in range(self.gallery_list.count()):
            item = self.gallery_list.item(i)
            if item.checkState() == Qt.Checked:
                items_to_delete.append(item)
                
        if not items_to_delete:
            QMessageBox.information(self, "Info", "No masks selected to delete.")
            return
            
        ret = QMessageBox.question(self, "Confirm Delete", f"Delete {len(items_to_delete)} files permanently?", QMessageBox.Yes | QMessageBox.No)
        
        if ret == QMessageBox.Yes:
            count = 0
            is_merged_mask_deleted = False
            for item in items_to_delete:
                fpath = Path(item.data(Qt.UserRole))
                try:
                    if fpath.exists():
                        os.remove(fpath)
                        count += 1
                        if fpath.name.startswith("final_merged_mask") or fpath.name.startswith("inverted_merged_mask"):
                            is_merged_mask_deleted = True
                except Exception as e:
                    self.log(f"[Err] Deleting {fpath.name}: {e}")
            
            self.log(f"[Delete] Removed {count} masks.")
            self.update_mask_gallery()
            
            if is_merged_mask_deleted:
                self.preview_canvas.clear()
                self.preview_canvas.setText("Final Merge Preview")
                self.final_merged_mask_data = None
                self.btn_save_rgba.setEnabled(False)
                self.btn_save_rgba.setText("Save Isolated Object (RGBA)")
                self.current_mask = None
                self.canvas.set_overlay_mask(None)
                self.log("[Info] Merged mask deleted -> Preview & Overlay cleared.")

    # [Helper] 마스크 병합 로직 분리 (중복 제거)
    def _calculate_merged_mask_from_selection(self):
        if self.image is None:
            return None
        selected_paths = []
        for i in range(self.gallery_list.count()):
            item = self.gallery_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_paths.append(Path(item.data(Qt.UserRole)))
        if not selected_paths:
            QMessageBox.information(self, "Info", "Please select at least one mask to merge.")
            return None

        h, w = self.image.shape[:2]
        final_mask = np.zeros((h, w), dtype=np.uint8)
        
        for path in selected_paths:
            try:
                m_pil = Image.open(path).convert("L")
                m_np = np.array(m_pil)
                if m_np.shape != final_mask.shape:
                    m_np = cv2.resize(m_np, (w, h), interpolation=cv2.INTER_NEAREST)
                m_np = (m_np > 127).astype(np.uint8) * 255
                final_mask = cv2.bitwise_or(final_mask, m_np)
            except Exception as e:
                self.log(f"[Error] Reading mask {path.name}: {e}")
                
        return final_mask

    # Merged Save (배경:검정, 객체:흰색)
    def merge_selected_masks(self):
        final_mask = self._calculate_merged_mask_from_selection()
        if final_mask is None: return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"final_merged_mask_{timestamp}.png"
            saved_path = save_image_file(final_mask, filename, self.save_sub_folder)

            if saved_path:
                self._post_save_merge_action(saved_path, final_mask, "Merged Mask")

        except Exception as e:
            self.log(f"[Error] Merge failed: {e}")
            QMessageBox.critical(self, "Merge Error", str(e))

    # Inverted Save (배경:흰색, 객체:검정)
    def save_inverted_mask(self):
        final_mask = self._calculate_merged_mask_from_selection()
        if final_mask is None: return
        
        try:
            # 마스크 반전 (255 - mask)
            inverted_mask = cv2.bitwise_not(final_mask)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"inverted_merged_mask_{timestamp}.png"
            saved_path = save_image_file(inverted_mask, filename, self.save_sub_folder)
            
            if saved_path:
                # 미리보기 및 오버레이는 원본(흰색=객체)을 보여주는 것이 직관적이므로 final_mask 사용
                # 저장만 Inverted로 수행됨
                self._post_save_merge_action(saved_path, final_mask, "Inverted Mask")
                self.log(f"[SAVE] Inverted Mask Saved: {filename}")

        except Exception as e:
            self.log(f"[Error] Inverted Save failed: {e}")
            QMessageBox.critical(self, "Save Error", str(e))

    # 저장 후 공통 처리 (UI 갱신 등) - 자동 닫힘 팝업 적용
    def _post_save_merge_action(self, saved_path, display_mask, title):
        out_name = Path(saved_path)
        self.log(f"[MERGE] Saved {title}: {out_name.name}")
    
        # 기존의 차단형 msg box 팝업(exec)을 제거하고 자동 닫힘 팝업 사용
        msg_html = f"<h3 style='color: #2ecc71;'>{title} Saved!</h3><p><b>File:</b> {out_name.name}</p>"
        self.show_auto_close_popup("Save Successful", msg_html, duration=1350) # 1.35초 후 자동 닫힘
        
        self.update_mask_gallery()
        self.current_mask = (display_mask > 0).astype(np.uint8)
        self.canvas.set_overlay_mask(self.current_mask)
        
        # 병합 데이터 저장 (항상 원본 기준=흰색 객체)
        self.final_merged_mask_data = display_mask
        self.btn_save_rgba.setEnabled(True)
        self.btn_save_rgba.setText(f"Save Isolated Object (RGBA)")
        
        preview_pixmap = self.create_preview_pixmap(display_mask)
        if preview_pixmap:
            self.preview_canvas.setPixmap(preview_pixmap)
            self.preview_canvas.setText("")

    def log(self, txt):
        if not hasattr(self, 'log_table'):
            print(f"[Pre-Init Log] {txt}")
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_table.insertRow(0)
        time_item = QTableWidgetItem(timestamp)
        time_item.setFont(QFont("Consolas", 9))
        self.log_table.setItem(0, 0, time_item)
        event_item = QTableWidgetItem(txt)
        self.log_table.setItem(0, 1, event_item)
        self.log_table.resizeRowToContents(0)
        if self.log_table.rowCount() > 100:
            self.log_table.removeRow(100)
    
    # 자동으로 닫히는 안내 팝업 (버튼 없음, 1초 후 종료)
    def show_auto_close_popup(self, title, html_message, duration=1000):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(html_message)
        msg_box.setStandardButtons(QMessageBox.NoButton) # 확인 버튼 제거
        
        # (선택 사항) 타이틀바 없는 깔끔한 토스트 스타일로 하려면 아래 주석 해제
        msg_box.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Dialog)
        msg_box.setStyleSheet("QMessageBox { border: 2px solid #3498db; background-color: #2b2b2b; } QLabel { color: #ecf0f1; }")

        msg_box.show() # exec() 대신 show()를 사용하여 비차단(Non-blocking) 실행
        
        # duration(ms) 후에 팝업 닫기
        QTimer.singleShot(duration, msg_box.accept)
    
    def toggle_progress_text(self):
        if self.progress_text_state == 0:
            self.progress_text_label.setText("Progressing...")
            self.progress_text_state = 1
        else:
            self.progress_text_label.setText("Waiting...")
            self.progress_text_state = 0

    def show_progress(self, message):
        self.progress_bar.setFormat(message)
        self.progress_bar.setVisible(True)
        self.status.showMessage(message)
        self.progress_text_label.setVisible(True)
        self.progress_text_label.setText("Progressing...")
        self.progress_text_state = 1
        self.progress_text_timer.start(600)
    
    def hide_progress(self):
        self.progress_bar.setVisible(False)
        self.status.clearMessage()
        self.progress_text_timer.stop()
        self.progress_text_label.setVisible(False)
    
    def start_button_blink(self, button):
        self.blinking_button = button
        self.blink_state = False
        self.blink_timer.start(500)
    
    def stop_button_blink(self):
        self.blink_timer.stop()
        if self.blinking_button:
            self.blinking_button.setStyleSheet("")
            self.blinking_button = None
    
    def toggle_button_blink(self):
        if self.blinking_button:
            if self.blink_state:
                self.blinking_button.setStyleSheet("QPushButton { background-color: #2ecc71; color: white; font-weight: bold; }")
            else:
                self.blinking_button.setStyleSheet("")
            self.blink_state = not self.blink_state
            
    def open_image(self):
        if self.worker is not None:
            if self.worker.isRunning():
                QMessageBox.warning(self, "Busy", "Another task is still running. Please wait.")
                return
            else:
                self.worker.deleteLater()
                self.worker = None

        fname, _ = QFileDialog.getOpenFileName(self, "Open image", os.getcwd(), "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not fname: return
        
        self.start_button_blink(self.btn_open)        
        self.show_progress("Loading image file...")
        self.img_path = fname
        self.toggle_loading(True, "Loading Image", "Reading high-res image data...")
        QTimer.singleShot(50, self._start_load_image_worker)
        
    def _start_load_image_worker(self):
        self.worker = GenericWorker(self._task_load_image_only, img_path=self.img_path)
        self.worker.signal_finished.connect(self._on_image_file_loaded) 
        self.worker.error.connect(self._on_task_error)
        self.worker.start()
    
    def _task_load_image_only(self, img_path, **kwargs):
        try:
            path_obj = Path(img_path)
            if not path_obj.exists(): raise FileNotFoundError(f"File not found: {img_path}")
            with SuppressStderr():
                pil = Image.open(img_path).convert("RGB")
                pil.load() 
            if 'progress_callback' in kwargs: kwargs['progress_callback'](50)
            image_np = np.array(pil)
            return image_np
        except OSError as e:
            print(f"[Warning] Corrupt or invalid image file: {e}")
            raise RuntimeError(f"Cannot load image (Corrupt file): {e}")
        except Exception as e:
            raise RuntimeError(f"Image load failed: {e}")

    def _on_image_file_loaded(self, image_np):
        self.toggle_loading(False) 
        if self.worker:
            self.worker.deleteLater() 
            self.worker = None 
        self.stop_button_blink() 
        self.hide_progress()     
        if image_np is None:
            self.status.showMessage("Failed to load image data.", 3000)
            return

        self.image = image_np
        self.is_image_set_to_sam = False 
        base = Path(self.img_path).stem
        default_save_path = config.get_config_value('Settings', 'default_save_path', './outputs')
        if os.path.isabs(default_save_path): save_root = Path(default_save_path)
        else: save_root = Path(project_root) / default_save_path
            
        self.result_dir = save_root / f"{base}_masks"
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.save_sub_folder = str(self.result_dir)
        self.result_counter = len(list(self.result_dir.glob("mask_*.png")))

        self.canvas.set_image(self.image)
        QTimer.singleShot(100, self.canvas.fit_to_window)
        
        self.update_mask_gallery()
        self.candidate_widget.setVisible(False)
        self.candidate_masks = None
        self.final_merged_mask_data = None
        self.btn_save_rgba.setEnabled(False)
        self.preview_canvas.setText("Final Merge Preview")
        self.preview_canvas.clear()
        self.right_tabs.setCurrentIndex(0)

        self.log(f"[OK] Image file loaded: {Path(self.img_path).name}")
        self.log(f"[Info] Save path set to: {self.result_dir}") 
        self.status.showMessage("Image loaded. Ready to select object.", 3000)
        
    def _on_task_error(self, e: Exception):
        # 1. 무한 로딩 창 즉시 끄기 (가장 중요)
        self.toggle_loading(False)
        self.stop_button_blink()
        self.hide_progress()
        
        # 2. 동작하던 백그라운드 Worker 안전하게 종료
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
            
        err_msg = str(e)
        self.log(f"[ERROR] Task failed: {err_msg}")
        self.status.showMessage("Error occurred.", 5000)
        
        # 3. 에러 내용 분석: VRAM 메모리 부족 오류일 경우 친절한 안내 제공
        if "could not create a memory" in err_msg or "out of memory" in err_msg.lower():
            friendly_msg = (
                "그래픽카드 VRAM(메모리)이 부족하여 작업을 완료할 수 없습니다.\n\n"
                "💡 [해결 방법]\n"
                "좌측 모델 선택 메뉴에서 'SAM2 Small' 또는 'SAM2 Tiny' 등\n"
                "더 가벼운 모델로 변경한 뒤 [Load] 버튼을 누르고 다시 시도해 주세요."
            )
            QMessageBox.critical(self, "Memory Error (VRAM 부족)", friendly_msg)
        else:
            # 기타 일반 에러
            QMessageBox.warning(self, "Error", f"작업 중 오류가 발생했습니다:\n{err_msg}")
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.resize(self.centralWidget().size())
        if self.preview_canvas.pixmap() is not None and self.current_mask is not None:
            preview_pixmap = self.create_preview_pixmap(self.current_mask * 255)
            if preview_pixmap:
                self.preview_canvas.setPixmap(preview_pixmap)
        if hasattr(self, 'canvas'):
             self.canvas.sig_view_changed.emit()
    
    def toggle_loading(self, show, title="Processing", desc="Please wait..."):
        if show:
            self.loading_overlay.set_message(title, desc)
            self.loading_overlay.resize(self.centralWidget().size())
            self.loading_overlay.show()
            self.loading_overlay.raise_() 
            self.set_ui_enabled(False) 
        else:
            self.loading_overlay.hide()
            self.set_ui_enabled(True) 

    def set_ui_enabled(self, enabled):
        self.btn_open.setEnabled(enabled)
        self.btn_extract.setEnabled(enabled)
        self.btn_load_sam.setEnabled(enabled)
        self.btn_reset.setEnabled(enabled)
        self.canvas.setEnabled(enabled)

    def change_mode(self, mode):
        if mode == "pan":
            self.canvas.set_mode(mode)
            if hasattr(self, 'btn_extract'):
                self.btn_extract.setEnabled(True)
            self.log(f"Mode changed -> PAN (Selection Kept)")
            return

        has_conflict = False
        has_brush = len(self.canvas.brush_strokes) > 0 # 충돌 조건에 'brush_strokes' 추가
        if mode == "box":
            if len(self.canvas.lasso_img) > 0 or len(self.canvas.point_list) > 0 or has_brush:
                has_conflict = True
        elif mode == "lasso":
            if self.canvas.box_img is not None or len(self.canvas.point_list) > 0 or has_brush:
                has_conflict = True
        elif mode == "point":
            if self.canvas.box_img is not None or len(self.canvas.lasso_img) > 0 or has_brush:
                has_conflict = True
        elif mode == "brush":
            if self.canvas.box_img is not None or len(self.canvas.lasso_img) > 0 or len(self.canvas.point_list) > 0:
                has_conflict = True
            
        if has_conflict:
            self.canvas.reset_selection()
            self.log(f"Mode changed -> {mode.upper()} (Selection Cleared due to conflict)")
        else:
            self.log(f"Mode changed -> {mode.upper()} (Selection Kept)")

        self.canvas.set_mode(mode)
        if hasattr(self, 'btn_extract'):
            self.btn_extract.setEnabled(True)
            self.btn_extract.setToolTip("Extract object based on current selection")    
    
    def on_selection_done(self, sel_type, data):
        if sel_type == "box":
            l, t, r, b = data
            self.log(f"BBOX selected: ({l},{t}) -> ({r},{b}) [{r-l}x{b-t}]")
        elif sel_type == "lasso":
            self.log(f"Lasso polygon completed: {len(data)} points")
            
    def on_point_added(self, pt_label):
        x,y,label = pt_label
        label_str = "POSITIVE" if label==1 else "NEGATIVE"
        self.log(f"Point added: ({x},{y}) [{label_str}]")
    
    def on_selection_cancelled(self, message):
        self.log(f"[CANCEL] {message}")
        self.status.showMessage(message, 2000)
    
    def on_undo(self, success, message):
        if success:
            self.log(f"[UNDO] {message}")
            self.status.showMessage(message, 2000)
        else:
            self.status.showMessage(message, 2000)
            
    def sync_scrollbars_from_canvas(self):
        if self.canvas.image is None:
            self.scroll_h.setEnabled(False)
            self.scroll_v.setEnabled(False)
            return

        view_w = self.canvas.width()
        view_h = self.canvas.height()
        content_w = self.canvas.img_w * self.canvas.scale
        content_h = self.canvas.img_h * self.canvas.scale

        if content_w > view_w:
            self.scroll_h.setEnabled(True)
            max_scroll_x = int(content_w - view_w)
            self.scroll_h.setRange(0, max_scroll_x)
            self.scroll_h.setPageStep(int(view_w)) 
            current_val = -int(self.canvas.offset_x)
            self.scroll_h.blockSignals(True)
            self.scroll_h.setValue(current_val)
            self.scroll_h.blockSignals(False)
        else:
            self.scroll_h.blockSignals(True)
            self.scroll_h.setEnabled(False)
            self.scroll_h.setRange(0, 0)
            self.scroll_h.blockSignals(False)

        if content_h > view_h:
            self.scroll_v.setEnabled(True)
            max_scroll_y = int(content_h - view_h)
            self.scroll_v.setRange(0, max_scroll_y)
            self.scroll_v.setPageStep(int(view_h))
            current_val = -int(self.canvas.offset_y)
            self.scroll_v.blockSignals(True)
            self.scroll_v.setValue(current_val)
            self.scroll_v.blockSignals(False)
        else:
            self.scroll_v.blockSignals(True)
            self.scroll_v.setEnabled(False)
            self.scroll_v.setRange(0, 0)
            self.scroll_v.blockSignals(False)

    def on_scrollbar_action(self):
        if self.canvas.image is None: return
        val_x = self.scroll_h.value()
        val_y = self.scroll_v.value()
        if self.scroll_h.isEnabled(): self.canvas.offset_x = -float(val_x)
        if self.scroll_v.isEnabled(): self.canvas.offset_y = -float(val_y)
        self.canvas.update()
        
    def save_lasso(self):
        if self.canvas.lasso_img is None or len(self.canvas.lasso_img) == 0:
            QMessageBox.information(self, "No lasso", "No lasso polygon to save.")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Save Lasso", os.getcwd(), "JSON (*.json)")
        if not fname: return
        data = {"poly": self.canvas.lasso_img}
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.log(f"[OK] Lasso saved: {Path(fname).name}")

    def load_lasso(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load Lasso", os.getcwd(), "JSON (*.json)")
        if not fname: return
        try:
            with open(fname, "r", encoding="utf-8") as f: data = json.load(f)
            poly = data.get("poly", None)
            if poly is None or len(poly) < 3:
                QMessageBox.information(self, "Invalid file", "No valid polygon found in JSON.")
                return
            self.canvas.lasso_img = [(int(p[0]), int(p[1])) for p in poly]
            self.canvas.update()
            self.log(f"[OK] Lasso loaded: {Path(fname).name} ({len(self.canvas.lasso_img)} pts)")
        except Exception as e:
            QMessageBox.warning(self, "Load error", f"Failed to load lasso: {e}")
    
    def create_preview_pixmap(self, final_mask: np.ndarray) -> QPixmap | None:
        if self.image is None: return None
        h, w = self.image.shape[:2]
        img_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        img_rgba[..., :3] = self.image
        alpha_channel = (final_mask > 0).astype(np.uint8) * 255 
        img_rgba[..., 3] = alpha_channel
        qimg = QImage(img_rgba.data, w, h, img_rgba.strides[0], QImage.Format_RGBA8888).copy()
        pixmap = QPixmap.fromImage(qimg)
        target_size = self.preview_canvas.size()
        if target_size.isEmpty(): target_size = QSize(300, 300)
        scaled_pixmap = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return scaled_pixmap
    
    def generate_brush_mask(self):
        """ 브러쉬 스트로크를 마스크 이미지로 변환 """
        if self.image is None: return None
        h, w = self.image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if self.canvas.brush_strokes:
            for stroke in self.canvas.brush_strokes:
                points = stroke['points']
                size = stroke['size']
                
                if len(points) > 1:
                    pts = np.array(points, dtype=np.int32)
                    for i in range(len(points) - 1):
                        pt1 = (int(points[i][0]), int(points[i][1]))
                        pt2 = (int(points[i+1][0]), int(points[i+1][1]))
                        cv2.line(mask, pt1, pt2, 255, thickness=int(size))
                        cv2.circle(mask, pt1, int(size/2), 255, -1)
                    cv2.circle(mask, (int(points[-1][0]), int(points[-1][1])), int(size/2), 255, -1)
                elif len(points) == 1:
                    cv2.circle(mask, (int(points[0][0]), int(points[0][1])), int(size/2), 255, -1)
        return mask
        
    def save_mask(self):
        if self.current_mask is None or self.save_sub_folder is None: 
            self.status.showMessage("No mask to save!", 2000)
            return
        self.result_counter += 1
        filename = f"mask_{self.result_counter:03d}.png"
        saved_path = save_image_file(self.current_mask, filename, self.save_sub_folder)
        if saved_path:
            self.update_mask_gallery()
            file_name = Path(saved_path).name
            self.log(f"[SAVE] Mask Saved: {file_name}")
            self.status.showMessage(f"Saved successfully! Opening Gallery...", 1000)
            QTimer.singleShot(700, lambda: self.right_tabs.setCurrentIndex(0))

    def save_next(self):
        self.save_mask()
        self.clear_selection(silent=True)

    # RGBA 저장 로직 - 갤러리 즉시 갱신 추가
    def save_rgba_result(self):
        if self.image is None or self.final_merged_mask_data is None:
            QMessageBox.warning(self, "Error", "No merged mask available. Please 'Merge' masks first.")
            return
        try:
            h, w = self.image.shape[:2]
            rgba_img = np.zeros((h, w, 4), dtype=np.uint8)
            rgba_img[..., :3] = self.image 
            mask_uint8 = (self.final_merged_mask_data > 0).astype(np.uint8) * 255
            rgba_img[..., 3] = mask_uint8 

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"object_rgba_{timestamp}.png" 
            save_path = os.path.join(self.save_sub_folder, filename)

            pil_img = Image.fromarray(rgba_img)
            pil_img.save(save_path, format="PNG")

            self.log(f"[SAVE RGBA] Saved transparent object: {filename}")
            
            # 파일 저장 후 갤러리 목록 UI 즉시 갱신
            self.update_mask_gallery()
            
            # 자동 닫힘 팝업
            msg_html = (f"<h3 style='color: #27ae60;'>Object Saved!</h3><p><b>File:</b> {filename}</p>"
                        f"<p style='color: gray; font-size: 11px;'>Ready for Background Synthesis (Step 3)</p>")
            self.show_auto_close_popup("Saved RGBA Image", msg_html, duration=1500)

        except Exception as e:
            self.log(f"[ERROR] Failed to save RGBA: {e}")
            QMessageBox.critical(self, "Save Error", f"Could not save RGBA image:\n{e}")
    
    def apply_noise_removal(self):
        if self.current_mask is None:
            QMessageBox.warning(self, "No Mask", "Please extract a mask first.")
            return
        use_limit = self.chk_pixel_limit.isChecked()
        ratio_val = self.spin_noise_ratio.value()
        self.log(f"[Processing] Cleaning noise (Limit={use_limit}, Ratio={ratio_val:.4f})...")
        try:
            cleaned_mask = remove_white_noise_component(
                self.current_mask, invert=False, 
                pixel_limit=use_limit, min_area_ratio=ratio_val, debug=True
            )
            if cleaned_mask is None or cleaned_mask.size == 0:
                self.log("[Warning] Noise removal returned empty mask.")
                return
            self.current_mask = cleaned_mask
            self.canvas.set_overlay_mask(self.current_mask)
            self.status.showMessage("White noise removed.", 2000)
            self.log("[OK] Noise removal applied.")
        except Exception as e:
            self.log(f"[Error] Noise removal failed: {e}")
            QMessageBox.warning(self, "Error", f"Failed to clean noise:\n{e}")
               
    def clear_selection(self, silent=False):
        self.stop_button_blink()
        nothing_to_clear = (
            self.image is None or (
                self.current_mask is None and
                self.canvas.box_img is None and
                len(self.canvas.lasso_img) == 0 and
                len(self.canvas.point_list) == 0 and
                len(self.canvas.current_lasso_preview) == 0
            )
        )
        if nothing_to_clear:
            if not silent: self.status.showMessage("Nothing to clear.", 2000)
            return
        
        self.canvas.reset_selection()
        self.current_mask = None
        self.candidate_widget.setVisible(False)
        self.candidate_masks = None
        
        self.log("Selection cleared.")
        if not silent: self.status.showMessage("Selection cleared.", 2000)
            
def main():
    app = QApplication(sys.argv)
    qdarktheme.setup_theme("dark")
    font = QFont("Malgun Gothic", 10)
    app.setFont(font)
    win = SAMGuiApp()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()