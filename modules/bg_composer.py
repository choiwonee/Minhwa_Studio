import gc
import os
import re
import sys
import time
import tempfile
import threading
from datetime import datetime
from io import BytesIO
from pathlib import Path

# UI 테마 라이브러리
import qdarktheme

# PyTorch 메모리 단편화 방지 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import cv2

from diffusers.utils.logging import disable_progress_bar
disable_progress_bar()

from gradio_client import Client, handle_file
from PIL import Image, ImageOps

# GUI Framework
from PySide6.QtWidgets import (
    QApplication, QDialog, QFormLayout, QFrame, QMainWindow, QLabel, QFileDialog, QPushButton, QWidget,
    QVBoxLayout, QHBoxLayout, QComboBox, QMessageBox, QGroupBox,
    QRadioButton, QButtonGroup, QDoubleSpinBox, QScrollBar, QScrollArea, QGridLayout, QTextEdit,
    QCheckBox, QStatusBar, QListWidget, QListWidgetItem, QSizePolicy, QSlider, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QSpinBox, QListWidget
)
from PySide6.QtCore import Qt, QSize, QTimer, QThread, Signal
from PySide6.QtGui import QFont, QIcon, QImage, QImageReader, QPixmap, QAction

# Local Module Imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.config_loader import config
from utils import token_key
from utils.common import save_image_file, get_output_dir, qimage_from_ndarray, SuppressStderr
from utils.translator import translator
from models.diffusion_estimator import DiffusionEstimator
from utils.gui_utils import GenericWorker, FloatingToolBar, ImageCanvas, ProcessingOverlay, VisualCameraWidget, KeySettingsDialog

# HF Client 캐시
HF_CLIENT_CACHE = {}

# =============================================================================
# Background Load Worker (Gallery)
# =============================================================================
class GalleryLoadWorker(QThread):
    signal_progress = Signal(int, int)
    signal_finished = Signal(list) 
    
    def __init__(self, output_dir, max_items, cached_meta):
        super().__init__()
        self.output_dir = output_dir
        self.max_items = max_items
        self.cached_meta = cached_meta
        self._results = []
        self.finished.connect(self._emit_results)

    def run(self):
        try:
            entries = sorted(
                [e for e in os.scandir(self.output_dir) if e.is_file() and e.name.lower().endswith(('.png','.jpg','.jpeg'))],
                key=lambda e: e.stat().st_mtime,
                reverse=True
            )[:self.max_items]
            
            self.signal_progress.emit(0, len(entries))
            
            for idx, entry in enumerate(entries):
                if self.isInterruptionRequested():
                    break
                path = entry.path
                mtime = entry.stat().st_mtime
                qimg = None
                
                if path not in self.cached_meta or self.cached_meta[path] != mtime:
                    reader = QImageReader(path)
                    if reader.size().isValid():
                        reader.setScaledSize(reader.size().scaled(QSize(100,100), Qt.KeepAspectRatio))
                    img = reader.read()
                    if not img.isNull():
                        qimg = img
                
                self._results.append((path, entry.name, mtime, qimg))
                self.signal_progress.emit(idx+1, len(entries))
        except:
            pass
            
    def _emit_results(self):
        self.signal_finished.emit(self._results)

# =============================================================================
# Upscale Dialog
# =============================================================================
class UpscaleSettingsDialog(QDialog):
    def __init__(self, parent=None, current_settings=None):
        super().__init__(parent)
        self.setWindowTitle("Upscale Settings")
        self.setFixedWidth(300)
        self.setStyleSheet("background-color: #2b2b2b; color: #ddd;")
        
        self.settings = current_settings or {"scale": 4.0, "tile": 512, "resize_back": True}
        
        layout = QVBoxLayout(self)
        form = QFormLayout()
        
        self.combo_scale = QComboBox()
        self.combo_scale.addItems(["2x", "3x", "4x"])
        idx = {2.0:0, 3.0:1, 4.0:2}.get(self.settings["scale"], 2)
        self.combo_scale.setCurrentIndex(idx)
        form.addRow("Scale Factor:", self.combo_scale)
        
        self.combo_tile = QComboBox()
        for t, v in [("Auto (Safe)", 512), ("High Perf (Large)", 0), ("Balanced", 512), ("Low VRAM", 256)]:
            self.combo_tile.addItem(t, v)
        
        self.combo_tile.setCurrentIndex(max(0, self.combo_tile.findData(self.settings["tile"])))
        form.addRow("VRAM Strategy:", self.combo_tile)
        
        self.chk_resize = QCheckBox("Fit to Original Size")
        self.chk_resize.setChecked(self.settings["resize_back"])
        form.addRow(self.chk_resize)
        
        layout.addLayout(form)
        btns = QHBoxLayout()
        
        btn_ok = QPushButton("Apply")
        btn_ok.clicked.connect(self.accept)
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        
        btns.addWidget(btn_ok)
        btns.addWidget(btn_cancel)
        layout.addLayout(btns)

    def get_settings(self):
        return {
            "scale": [2.0,3.0,4.0][self.combo_scale.currentIndex()],
            "tile": self.combo_tile.currentData(),
            "resize_back": self.chk_resize.isChecked()
        }

# =============================================================================
# Main Application Class
# =============================================================================
class BgComposerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Heritage Composer (Lite Edition)")
        
        screen = QApplication.primaryScreen().availableGeometry()
        if screen.width() <= 1400 or screen.height() <= 850:
            self.resize(screen.width(), screen.height())
            self.setWindowState(Qt.WindowMaximized)
        else:
            self.resize(int(screen.width() * 0.81), int(screen.height() * 0.81))
            frame_geo = self.frameGeometry()
            frame_geo.moveCenter(screen.center())
            self.move(frame_geo.topLeft())
            
        self.scenarios = config.get_scenarios()
        self.generation_models_dict = config.get_models("generation")
        
        self.hf_token = token_key.get_valid_hf_token()
        self.api_key = token_key.get_valid_api_key()
        
        if self.hf_token:
             try: 
                 from huggingface_hub import login
                 login(token=self.hf_token)
             except:
                 pass

        self.diffusion_estimator = DiffusionEstimator()
        self.hw_info = self.diffusion_estimator.hw_info

        # Canvas & Data Init
        self.input_canvas = None
        self.mask_canvas = None 
        self.result_canvas = None
        
        self.image = None
        self.multi_images = []
        self.result_image = None        
        self.external_mask = None
        
        self.worker = None
        self.preview_worker = None
        self.gallery_worker = None
        
        self.output_dir = get_output_dir("bg_local")
        self.current_model_mode = "local"
        self._active_model_config = None
        self._user_touched_steps = False
        self._user_touched_cfg = False
        self._cancel_event = threading.Event()
        
        self.init_ui()
        self.loading_overlay = ProcessingOverlay(self.centralWidget())
        self.loading_overlay.sig_cancel_requested.connect(self.cancel_generation)

        self.log(f"System Ready. Models: {len(self.generation_models_dict)}")
        self.update_gallery()
        self.apply_smart_defaults()
        
        self.blink_timer = QTimer(self)
        self.blink_timer.setInterval(600)
        self.blink_timer.timeout.connect(self._update_blinking_message)
        self._blink_state = True

    def _stop_worker(self, worker_attr_name):
        worker = getattr(self, worker_attr_name, None)
        if worker is None:
            return
        
        try:
            worker.signal_finished.disconnect()
            worker.error.disconnect()
        except:
            pass
            
        if worker.isRunning():
            worker.requestInterruption()
            worker.quit()
            worker.finished.connect(lambda: self._cleanup_worker_ref(worker, worker_attr_name))
            worker.finished.connect(worker.deleteLater)
        else:
            worker.deleteLater()
            setattr(self, worker_attr_name, None)

    def _cleanup_worker_ref(self, worker_obj, attr_name):
        if getattr(self, attr_name, None) is worker_obj:
            setattr(self, attr_name, None)

    def init_ui(self):
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        
        root = QWidget()
        self.setCentralWidget(root)
        
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        self.vertical_splitter = QSplitter(Qt.Vertical)
        self.vertical_splitter.setHandleWidth(4)

        # --- 상단 영역 ---
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(4)
        
        self.main_splitter.addWidget(self._setup_input_panel())
        
        center_splitter = QSplitter(Qt.Vertical)
        center_splitter.setHandleWidth(4)
        center_splitter.addWidget(self._setup_mask_panel())
        center_splitter.addWidget(self._setup_result_panel())
        center_splitter.setStretchFactor(0, 1)
        center_splitter.setStretchFactor(1, 2)
        
        self.main_splitter.addWidget(center_splitter)
        self.main_splitter.addWidget(self._setup_right_panel())
        
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 3)
        self.main_splitter.setStretchFactor(2, 2)
        
        self.vertical_splitter.addWidget(self.main_splitter)

        # --- 하단 영역 ---
        control_scroll = QScrollArea()
        control_scroll.setWidget(self._create_control_panel())
        control_scroll.setWidgetResizable(True)
        control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        control_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        control_scroll.setFrameShape(QFrame.NoFrame)
        control_scroll.setMinimumHeight(180)

        self.vertical_splitter.addWidget(control_scroll)
        self.vertical_splitter.setStretchFactor(0, 7)
        self.vertical_splitter.setStretchFactor(1, 3)

        main_layout.addWidget(self.vertical_splitter)
        
        self.update_model_list()
        self.setup_shortcuts()

    def _create_view_control_bar(self, canvas, extra_widgets=None, show_pan=True):
        container = QWidget()
        container.setStyleSheet("background-color: #2b2b2b; border-top: 1px solid #3d3d3d;")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(4)

        if show_pan:
            btn_pan = QPushButton("✋")
            btn_pan.setCheckable(True)
            btn_pan.setFixedSize(28, 24)
            btn_pan.setToolTip("Toggle Pan Mode")
            btn_pan.setStyleSheet("""
                QPushButton { border: none; border-radius: 3px; background: transparent; color: #bbb; }
                QPushButton:checked { background-color: #444; color: #fff; border: 1px solid #555; }
                QPushButton:hover { background-color: #3d3d3d; }
            """)
            btn_pan.toggled.connect(lambda c: canvas.set_mode("pan" if c else "view"))
            if canvas.mode == "pan": btn_pan.setChecked(True)
            layout.addWidget(btn_pan)

        btn_fit = QPushButton("Fit")
        btn_fit.setFixedSize(40, 24)
        btn_fit.clicked.connect(canvas.fit_to_window)
        btn_fit.setStyleSheet("QPushButton { border: none; background: #34495e; color: white; border-radius: 3px; font-size: 12px; }")

        btn_act = QPushButton("1:1")
        btn_act.setFixedSize(40, 24)
        btn_act.clicked.connect(canvas.set_actual_size)
        btn_act.setStyleSheet("QPushButton { border: none; background: #34495e; color: white; border-radius: 3px; font-size: 12px; }")
        
        layout.addWidget(btn_fit)
        layout.addWidget(btn_act)
        
        if extra_widgets:
            layout.addStretch(1)
            for w in extra_widgets:
                layout.addWidget(w)
        else:
            layout.addStretch(1)

        return container

    def setup_floating_pan_button(self, canvas, margin=(20, 20)):
        """ 캔버스 위에 플로팅 팬 버튼을 생성 (우측 하단) """
        btn = QPushButton(canvas)
        btn.setCheckable(True)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setText("✥") 
        btn.setFixedSize(40, 40)
        btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(28, 28, 28, 240); 
                color: #bdc3c7; border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 8px; font-weight: bold; font-size: 20px;
                padding-bottom: 2px; outline: none;
            }
            QPushButton:hover { background-color: rgba(255, 255, 255, 0.1); color: white; }
            QPushButton:checked { background-color: rgba(255, 255, 255, 0.25); color: white; }
        """)
        
        # 버튼 토글과 캔버스 모드 동기화
        btn.toggled.connect(lambda c: canvas.set_mode("pan" if c else "view"))
        
        # 캔버스 크기 변경 시 위치 재조정
        def update_pos():
            # 우측 하단 배치
            x = canvas.width() - btn.width() - margin[0]
            y = canvas.height() - btn.height() - margin[1]
            btn.move(x, y)
            
        canvas.resizeEvent = (lambda orig_ev: lambda e: (orig_ev(e), update_pos()))(canvas.resizeEvent)
        # 초기 위치 설정
        QTimer.singleShot(0, update_pos)
        
        btn.show()
        return btn

    def _create_styled_panel(self, title, widget):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        lbl = QLabel(title)
        lbl.setFixedHeight(26)
        lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        lbl.setStyleSheet("QLabel { background-color: #2b2b2b; color: #bdc3c7; font-weight: bold; font-size: 11px; padding-left: 8px; border-bottom: 1px solid #3d3d3d; }")
        
        layout.addWidget(lbl)
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(widget, 1)
        return container

    def _setup_input_panel(self):
        self.input_canvas = ImageCanvas(self)
        self.input_canvas.set_mode("box")
        self.input_canvas.on_selection_done = self.on_input_selection_changed
        self.input_canvas.sig_view_changed.connect(self.sync_scrollbars)
        
        canvas_wrapper = self._wrap_canvas_with_scrollbars(self.input_canvas)
        
        # 도구 툴바 및 팬 버튼 설정
        self.setup_floating_toolbar()
        self.btn_pan_float = self.setup_floating_pan_button(self.input_canvas)
        
        return self._create_styled_panel("1. INPUT IMAGE (Draw Mask)", canvas_wrapper)

    def _setup_mask_panel(self):
        content_container = QWidget()
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self.mask_canvas = ImageCanvas(self)
        self.mask_canvas.set_mode("view")
        self.mask_canvas.setStyleSheet("background-color: #141414;")
        
        content_layout.addWidget(self.mask_canvas, 1)
        ctrl_bar = self._create_view_control_bar(self.mask_canvas, show_pan=False)
        content_layout.addWidget(ctrl_bar, 0)
        self.btn_pan_mask = self.setup_floating_pan_button(self.mask_canvas)
        return self._create_styled_panel("2. MASK PREVIEW", content_container)

    def _setup_result_panel(self):
        content_container = QWidget()
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        self.result_canvas = ImageCanvas(self)
        self.result_canvas.set_mode("pan") 
        content_layout.addWidget(self.result_canvas, 1) 
        
        btn_save = QPushButton("Save Image")
        btn_save.setFixedWidth(100)
        btn_save.setFixedSize(100, 24)
        btn_save.setStyleSheet("background: #d35400; color: white; border: none; border-radius: 4px; font-weight: bold; font-size: 11px;")
        btn_save.clicked.connect(self.save_result)
        self._setup_btn_feedback(btn_save) 
        
        ctrl_bar = self._create_view_control_bar(self.result_canvas, extra_widgets=[btn_save], show_pan=False)
        content_layout.addWidget(ctrl_bar, 0)
        self.btn_pan_result = self.setup_floating_pan_button(self.result_canvas)
        return self._create_styled_panel("3. GENERATION RESULT", content_container)

    def _setup_right_panel(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #444; border-top: 1px solid #3d3d3d; }
            QTabBar::tab { background: #2b2b2b; color: #888; padding: 5px 10px; }
            QTabBar::tab:selected { background: #3d3d3d; color: #ecf0f1; font-weight: bold; }
        """)
        
        self.camera_controller = VisualCameraWidget()
        if hasattr(self.camera_controller, 'sig_prompt_changed'):
            self.camera_controller.sig_prompt_changed.connect(self.on_camera_prompt_update)
            
        self.gallery_list = QListWidget()
        self.gallery_list.setViewMode(QListWidget.IconMode)
        self.gallery_list.setIconSize(QSize(80, 80))
        self.gallery_list.setResizeMode(QListWidget.Adjust)
        self.gallery_list.itemDoubleClicked.connect(self.on_gallery_double_clicked)
        self.gallery_list.setStyleSheet("background: #202020; border: none;")
        self.tabs.addTab(self.gallery_list, "Gallery")

        self.log_table = QTableWidget()
        self.log_table.setColumnCount(2)
        self.log_table.setHorizontalHeaderLabels(["Time", "Msg"])
        self.log_table.horizontalHeader().setStretchLastSection(True)
        self.log_table.verticalHeader().setVisible(False)
        self.log_table.setStyleSheet("background: #1e1e1e; border: none; gridline-color: #333;")
        self.tabs.addTab(self.log_table, "Log")
        
        layout.addWidget(self.tabs)
        return container

    def _create_control_panel(self):
        # [타이머 복구]
        if not hasattr(self, 'auto_trans_timer'):
            self.auto_trans_timer = QTimer(self)
            self.auto_trans_timer.setSingleShot(True)
            self.auto_trans_timer.setInterval(3000) 
            self.auto_trans_timer.timeout.connect(self.on_retranslate_requested)

        # 메인 패널 스타일
        panel = QGroupBox("Control Panel")
        panel.setStyleSheet("""
            QGroupBox {
                font-weight: bold; border: 1px solid #444; border-radius: 4px;
                margin-top: 8px; padding-top: 4px; padding-bottom: 2px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }
            QTabWidget::pane { border: 1px solid #3d3d3d; border-radius: 2px; }
            QTabBar::tab { 
                background: #252525; color: #888; padding: 4px 10px; 
                border-top-left-radius: 4px; border-top-right-radius: 4px;
                min-width: 60px; font-size: 11px;
            }
            QTabBar::tab:selected { background: #3d3d3d; color: #ecf0f1; font-weight: bold; border-bottom: 2px solid #3498db; }
            QLabel { color: #ccc; font-size: 11px; }
            QCheckBox { font-size: 11px; color: #ddd; spacing: 4px; }
            QPushButton {
                background-color: #3e3e3e; color: #ccc; border: 1px solid #555;
                border-radius: 3px; font-size: 11px; padding: 2px 6px;
            }
            QPushButton:hover { background-color: #4e4e4e; color: white; border-color: #777; }
            QPushButton:pressed { background-color: #2e2e2e; padding-top: 3px; }
        """)
        
        # 메인 레이아웃 (가로 분할)
        main_layout = QHBoxLayout(panel)
        main_layout.setContentsMargins(6, 10, 6, 6)
        main_layout.setSpacing(10)

        # ==========================================================
        # [LEFT] 설정 탭 (Settings)
        # ==========================================================
        setting_tabs = QTabWidget()
        setting_tabs.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        
        # --- Tab 1: 기본 설정 (Basic) ---
        tab_basic = QWidget()
        lb = QVBoxLayout(tab_basic)
        lb.setContentsMargins(8, 8, 8, 8)
        lb.setSpacing(8)

        # 1. 모델 모드
        grp_mode = QWidget()
        l_mode = QHBoxLayout(grp_mode)
        l_mode.setContentsMargins(0,0,0,0)
        l_mode.setSpacing(12)
        
        self.bg_mode = QButtonGroup(self)
        self.rb_local = QRadioButton("Local")
        self.rb_remote = QRadioButton("Remote")
        self.bg_mode.addButton(self.rb_local)
        self.bg_mode.addButton(self.rb_remote)
        self.rb_local.setChecked(True)
        self.bg_mode.buttonClicked.connect(self.update_model_list)
        
        l_mode.addWidget(QLabel("Mode:"))
        l_mode.addWidget(self.rb_local)
        l_mode.addWidget(self.rb_remote)
        l_mode.addStretch()
        
        self.btn_token_conf = QPushButton("API Key")
        self.btn_token_conf.setCursor(Qt.PointingHandCursor)
        self.btn_token_conf.setToolTip("HuggingFace / Gemini API 키 설정")
        self.btn_token_conf.clicked.connect(self.open_token_settings)
        self.btn_token_conf.setStyleSheet("color: #ecc058; border: 1px solid #777; padding: 2px 8px;") 
        l_mode.addWidget(self.btn_token_conf)
        lb.addWidget(grp_mode)

        # 2. 모델 선택 및 비율 설정 (Grid)
        grp_model = QGridLayout()
        grp_model.setSpacing(6)
        
        self.combo_model = QComboBox()
        self.combo_model.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.combo_model.currentIndexChanged.connect(self.on_model_combo_changed)
        
        self.btn_load = QPushButton("Load Model")
        self.btn_load.setCursor(Qt.PointingHandCursor)
        self.btn_load.setFixedHeight(24)
        self.btn_load.setStyleSheet("background-color: #34495e; color: white; border-radius: 3px; font-weight: bold;")
        self.btn_load.clicked.connect(self.on_load_model_request)
        self._setup_btn_feedback(self.btn_load)
        
        self.lbl_model_status = QLabel("Not Loaded")
        self.lbl_model_status.setAlignment(Qt.AlignCenter)
        self.lbl_model_status.setStyleSheet("background: #222; border: 1px solid #444; border-radius: 3px; color: #888; font-size: 10px;")
        
        # 추가: Ratio를 Basic으로 이동
        self.combo_res_mode = QComboBox()
        for t, d in [("Match Input","match_input"),("1:1 Square","1:1"),("16:9 Wide","16:9"),("9:16 Portrait","9:16"),("4:3 Standard","4:3")]:
            self.combo_res_mode.addItem(t, d)

        grp_model.addWidget(QLabel("Model:"), 0, 0)
        grp_model.addWidget(self.combo_model, 0, 1, 1, 2)
        grp_model.addWidget(self.btn_load, 1, 0, 1, 2)
        grp_model.addWidget(self.lbl_model_status, 1, 2)
        
        grp_model.addWidget(QLabel("Ratio:"), 2, 0)
        grp_model.addWidget(self.combo_res_mode, 2, 1, 1, 2)
        
        lb.addLayout(grp_model)

        # 3. 입력 소스
        grp_src = QHBoxLayout()
        grp_src.setSpacing(6)
        
        # Image Source
        self.chk_use_image = QCheckBox("Image")
        self.chk_use_image.setToolTip("Enable Input Image")
        btn_img = QPushButton("Open")
        btn_img.clicked.connect(self.open_image)
        btn_clear_img = QPushButton("Clear")
        btn_clear_img.setStyleSheet("color: #e74c3c; font-weight: bold;")
        btn_clear_img.clicked.connect(self.clear_image)

        # Mask Source
        self.chk_use_mask = QCheckBox("Mask")
        self.chk_use_mask.setToolTip("Enable Mask")
        self.chk_use_image.toggled.connect(lambda c: self.chk_use_mask.setEnabled(c))
        
        btn_msk = QPushButton("Open")
        btn_msk.clicked.connect(self.open_external_mask)
        btn_clr = QPushButton("Clear")
        btn_clr.setStyleSheet("color: #e74c3c; font-weight: bold;")
        btn_clr.clicked.connect(self.clear_external_mask)

        grp_src.addWidget(self.chk_use_image)
        grp_src.addWidget(btn_img)
        grp_src.addWidget(btn_clear_img)
        
        grp_src.addSpacing(5)
        v_sep = QFrame(); v_sep.setFrameShape(QFrame.VLine); v_sep.setFrameShadow(QFrame.Sunken); v_sep.setStyleSheet("color:#444")
        grp_src.addWidget(v_sep)
        grp_src.addSpacing(5)
        
        grp_src.addWidget(self.chk_use_mask)
        grp_src.addWidget(btn_msk)
        grp_src.addWidget(btn_clr)
        grp_src.addStretch()
        lb.addLayout(grp_src)
        
        # Multi-Image
        self.multi_image_group = QGroupBox("Multi-Image"); self.multi_image_group.setVisible(False)
        m_lay = QHBoxLayout(self.multi_image_group); m_lay.setContentsMargins(4,4,4,4)
        self.list_multi_imgs = QListWidget(); self.list_multi_imgs.setFixedHeight(40)
        
        self.btn_add_mi = QPushButton("Add")
        self.btn_del_mi = QPushButton("Del")
        self.btn_add_mi.clicked.connect(self.add_multi_images)
        self.btn_del_mi.clicked.connect(self.del_multi_images)
        
        m_lay.addWidget(self.list_multi_imgs)
        m_lay.addWidget(self.btn_add_mi)
        m_lay.addWidget(self.btn_del_mi)
        lb.addWidget(self.multi_image_group)
        
        lb.addStretch()

        # --- Tab 2: 상세 조정 (Detail) ---
        tab_params = QWidget()
        lp = QGridLayout(tab_params)
        lp.setContentsMargins(8, 12, 8, 8)
        lp.setSpacing(10)

        # Row 0: Steps & CFG
        lp.addWidget(QLabel("Steps:"), 0, 0)
        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(1, 100); self.spin_steps.setValue(30)
        lp.addWidget(self.spin_steps, 0, 1)
        
        lp.addWidget(QLabel("CFG Scale:"), 0, 2)
        self.spin_cfg = QDoubleSpinBox()
        self.spin_cfg.setRange(0.0, 50.0); self.spin_cfg.setValue(7.5); self.spin_cfg.setSingleStep(0.5)
        lp.addWidget(self.spin_cfg, 0, 3)

        # Row 1: Img Guide & Preset
        self.lbl_img_guidance = QLabel("Img CFG:") 
        self.spin_img_guidance = QDoubleSpinBox()
        self.spin_img_guidance.setRange(1.0, 10.0); self.spin_img_guidance.setSingleStep(0.1)
        self.lbl_img_guidance.setVisible(False); self.spin_img_guidance.setVisible(False)
        lp.addWidget(self.lbl_img_guidance, 1, 0)
        lp.addWidget(self.spin_img_guidance, 1, 1)

        lp.addWidget(QLabel("Preset:"), 2, 0)
        self.combo_scenario = QComboBox()
        self.combo_scenario.addItem("- Select -")
        self.combo_scenario.addItems(list(self.scenarios.keys()))
        self.combo_scenario.currentIndexChanged.connect(self.on_scenario_changed)
        lp.addWidget(self.combo_scenario, 2, 1, 1, 3)

        # Row 3: Multi-Angle & Upscale
        h_line = QFrame(); h_line.setFrameShape(QFrame.HLine); h_line.setFrameShadow(QFrame.Sunken); h_line.setStyleSheet("color:#444")
        lp.addWidget(h_line, 3, 0, 1, 4)
        
        self.chk_multi_angle = QCheckBox("Multi-Angle")
        self.chk_multi_angle.setToolTip("다각도 카메라 제어 활성화 (지원 모델 전용)")
        self.chk_multi_angle.setEnabled(False)
        self.chk_multi_angle.toggled.connect(self.on_multi_angle_toggled)

        self.chk_upscale = QCheckBox("Upscale")
        self.chk_upscale.setToolTip("고해상도 업스케일링 적용")
        
        self.btn_upscale_conf = QPushButton("Config")
        self.btn_upscale_conf.setToolTip("업스케일링 설정")
        self.btn_upscale_conf.clicked.connect(self.open_upscale_settings)
        self.btn_upscale_conf.setEnabled(False)
        self.chk_upscale.toggled.connect(self.btn_upscale_conf.setEnabled)
        
        row_tools = QHBoxLayout()
        row_tools.addWidget(self.chk_multi_angle)
        row_tools.addWidget(self.chk_upscale)
        row_tools.addWidget(self.btn_upscale_conf)
        row_tools.addStretch()
        lp.addLayout(row_tools, 4, 0, 1, 4)

        # 더미 위젯 (코드 호환성 유지용)
        self.chk_manual_prompt = QCheckBox("Manual"); self.chk_manual_prompt.setVisible(False)
        self.combo_precision = QComboBox(); self.combo_quant = QComboBox()
        self.upscale_settings = {"scale": 4.0, "tile": 512, "resize_back": True}
        
        lp.setRowStretch(5, 1) # 빈 공간 채우기

        setting_tabs.addTab(tab_basic, "Basic")
        setting_tabs.addTab(tab_params, "Detail")

        # ==========================================================
        # [RIGHT] 프롬프트 & 생성
        # ==========================================================
        right_panel = QWidget()
        rp = QVBoxLayout(right_panel)
        rp.setContentsMargins(0, 0, 0, 0)
        rp.setSpacing(6)

        # 1. 헤더
        row_head = QHBoxLayout()
        lbl_title = QLabel("PROMPT INPUT")
        lbl_title.setStyleSheet("color: #bbb; font-weight: bold; font-size: 11px;")
        
        self.lbl_p_count = QLabel("P: 0/75"); self.lbl_n_count = QLabel("N: 0/75")
        self.lbl_p_count.setStyleSheet("color:#2ecc71; font-family: Consolas; font-size:10px;")
        self.lbl_n_count.setStyleSheet("color:#e74c3c; font-family: Consolas; font-size:10px;")
        
        row_head.addWidget(lbl_title)
        row_head.addStretch()
        row_head.addWidget(self.lbl_p_count)
        row_head.addSpacing(8)
        row_head.addWidget(self.lbl_n_count)
        rp.addLayout(row_head)

        # 2. 텍스트 에디터 및 버튼 레이아웃 병합
        input_gen_layout = QHBoxLayout()
        
        # 좌측: 프롬프트 및 번역창
        text_layout = QVBoxLayout()
        text_layout.setSpacing(6)
        
        self.txt_prompt = QTextEdit()
        self.txt_prompt.setPlaceholderText("Positive Prompt (한글 가능)")
        self.txt_prompt.setMinimumHeight(80) # 입력창 높이 대폭 확대
        self.txt_prompt.setStyleSheet("background: #252525; border: 1px solid #555; border-radius: 4px; padding: 6px;")
        
        self.txt_negative = QTextEdit()
        self.txt_negative.setPlaceholderText("Negative Prompt")
        self.txt_negative.setMaximumHeight(45) # 부정 프롬프트 창 높이 조절
        self.txt_negative.setStyleSheet("background: #252525; border: 1px solid #555; border-radius: 4px; padding: 4px;")

        self.txt_prompt.textChanged.connect(self.update_word_counts)
        self.txt_negative.textChanged.connect(self.update_word_counts)

        text_layout.addWidget(self.txt_prompt)
        text_layout.addWidget(self.txt_negative)

        # 하단: 번역 결과 및 Run Trans 버튼
        trans_layout = QHBoxLayout()
        trans_layout.setSpacing(8)
        
        self.btn_retrans = QPushButton("Run Trans")
        self.btn_retrans.setFixedSize(80, 45) # 버튼 크기 정렬
        self.btn_retrans.setStyleSheet("background-color: #34495e; color: white; border-radius: 4px; font-weight: bold;")
        self.btn_retrans.clicked.connect(self.on_retranslate_requested)
        
        self.txt_trans_result = QTextEdit()
        self.txt_trans_result.setReadOnly(True)
        self.txt_trans_result.setFixedHeight(45)
        self.txt_trans_result.setPlaceholderText("수동 번역 결과가 여기에 표시됩니다.")
        self.txt_trans_result.setStyleSheet("background: #1e1e1e; color: #aaa; border: 1px solid #3d3d3d; font-size: 11px; padding: 4px;")
        
        trans_layout.addWidget(self.btn_retrans)
        trans_layout.addWidget(self.txt_trans_result)
        text_layout.addLayout(trans_layout)

        # 우측: Generate 버튼 (세로로 길게 배치)
        self.btn_gen = QPushButton("GENERATE")
        self.btn_gen.setMinimumHeight(150)
        self.btn_gen.setFixedWidth(120)
        self.btn_gen.setCursor(Qt.PointingHandCursor)
        self.btn_gen.setStyleSheet("""
            QPushButton { 
                background-color: #d35400; color: white; 
                font-weight: bold; font-size: 14px;
                border-radius: 6px; border: 1px solid #e67e22;
            }
            QPushButton:hover { background-color: #e67e22; border: 1px solid #f39c12; }
            QPushButton:pressed { background-color: #a84300; margin-top: 2px; }
            QPushButton:disabled { background-color: #555; color: #888; border: 1px solid #444; }
        """)
        self.btn_gen.clicked.connect(self.run_generation)
        self._setup_btn_feedback(self.btn_gen)

        input_gen_layout.addLayout(text_layout)
        input_gen_layout.addWidget(self.btn_gen)
        
        rp.addLayout(input_gen_layout)

        # 메인 레이아웃 비율 적용
        main_layout.addWidget(setting_tabs, 45)
        main_layout.addWidget(right_panel, 55)

        return panel

    def add_multi_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png *.jpg *.jpeg)")
        if files:
            for f in files:
                self.list_multi_imgs.addItem(f)
                self.multi_images.append(f)

    def del_multi_images(self):
        for item in self.list_multi_imgs.selectedItems():
            row = self.list_multi_imgs.row(item)
            self.list_multi_imgs.takeItem(row)
            if row < len(self.multi_images):
                self.multi_images.pop(row)

    def _setup_btn_feedback(self, btn: QPushButton):
        if not btn: return
        def flash_text():
            original_style = btn.styleSheet()
            btn.setStyleSheet(original_style + "; color: #f1c40f;")
            QTimer.singleShot(400, lambda: btn.setStyleSheet(original_style))
        btn.clicked.connect(flash_text)

    def _wrap_canvas_with_scrollbars(self, canvas):
        w = QWidget()
        g = QGridLayout(w)
        g.setContentsMargins(0,0,0,0)
        g.setSpacing(0)
        self.scroll_h = QScrollBar(Qt.Horizontal)
        self.scroll_v = QScrollBar(Qt.Vertical)
        self.scroll_h.valueChanged.connect(self.on_scrollbar_action)
        self.scroll_v.valueChanged.connect(self.on_scrollbar_action)
        g.addWidget(canvas,0,0)
        g.addWidget(self.scroll_v,0,1)
        g.addWidget(self.scroll_h,1,0)
        return w

    def setup_floating_toolbar(self):
        """ 플로팅 도구 툴바 UI 설정 (오류 수정 및 로직 복구) """
        self.toolbar = FloatingToolBar(self.input_canvas)
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
            QPushButton[simple_btn="true"] {
                background-color: rgba(255, 255, 255, 0.06);
                color: #ccc; border: none; border-radius: 6px;
                padding: 8px; font-weight: 600;
            }
            QPushButton[simple_btn="true"]:hover {
                background-color: rgba(255, 255, 255, 0.12); color: white;
            }
        """)

        # 기존 레이아웃 재사용 (FloatingToolBar가 이미 레이아웃을 가지고 있음)
        layout = self.toolbar.layout()
        if layout is None:
            layout = QVBoxLayout(self.toolbar)

        # 헬퍼 함수: 커스텀 옵션(라디오 + 텍스트) 생성
        def create_custom_option(text):
            container = QWidget()
            container.setObjectName("OptionContainer")
            container.setStyleSheet("""
                QWidget#OptionContainer {
                    background-color: rgba(28, 28, 28, 0.0);
                    border-radius: 6px; border: none;
                }
                QWidget#OptionContainer:hover {
                    background-color: rgba(255, 255, 255, 0.12);
                }
            """)
            h_layout = QHBoxLayout(container)
            h_layout.setContentsMargins(6, 4, 6, 4)
            h_layout.setSpacing(10)
            
            btn = QPushButton("")
            btn.setCheckable(True)
            btn.setFixedSize(20, 20)
            btn.setCursor(Qt.PointingHandCursor)
            
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #bdc3c7; font-weight: bold; font-size: 11px; border: none; background: transparent;")
            lbl.setCursor(Qt.PointingHandCursor)
            
            h_layout.addWidget(btn)
            h_layout.addWidget(lbl)
            h_layout.addStretch(1)
            
            # 라벨 클릭 시 버튼 클릭 효과
            def label_mouse_press(event):
                if event.button() == Qt.LeftButton:
                    btn.animateClick()
            lbl.mousePressEvent = label_mouse_press
            
            # 체크 상태 시각화 업데이트
            def update_visual(checked):
                btn.setText("🗸" if checked else "")
                lbl.setStyleSheet(f"color: {'#fff' if checked else '#bdc3c7'}; border: none; background: transparent; font-weight: bold; font-size: 11px;")
            
            btn.toggled.connect(update_visual)
            update_visual(False)
            
            return container, btn

        # 1. MODE 섹션
        lbl_mode = QLabel("MODE"); lbl_mode.setProperty("header", "true")
        layout.addWidget(lbl_mode)
        
        self.mode_bg = QButtonGroup(self) # bg_tools 대신 mode_bg로 명칭 통일

        # Box
        row_box, self.rb_box = create_custom_option("Box")
        self.mode_bg.addButton(self.rb_box, 1)
        layout.addWidget(row_box)
        
        # Lasso
        row_lasso, self.rb_lasso = create_custom_option("Lasso")
        self.mode_bg.addButton(self.rb_lasso, 2)
        layout.addWidget(row_lasso)

        # Brush
        row_brush, self.rb_brush = create_custom_option("Brush")
        self.mode_bg.addButton(self.rb_brush, 3) 
        layout.addWidget(row_brush)

        # Brush Size Slider (초기엔 숨김)
        self.brush_size_container = QWidget()
        bs_layout = QHBoxLayout(self.brush_size_container)
        bs_layout.setContentsMargins(6, 0, 6, 0)
        
        lbl_sz = QLabel("Size:")
        lbl_sz.setStyleSheet("color: #bbb; font-size: 10px;")
        
        self.slider_brush = QSlider(Qt.Horizontal)
        self.slider_brush.setRange(5, 100)
        self.slider_brush.setValue(20)
        self.slider_brush.setFixedWidth(80)
        self.slider_brush.valueChanged.connect(self.input_canvas.set_brush_size)
        
        bs_layout.addWidget(lbl_sz)
        bs_layout.addWidget(self.slider_brush)
        self.brush_size_container.setVisible(False) 
        layout.addWidget(self.brush_size_container)
        
        # 버튼 그룹 이벤트 연결
        self.mode_bg.buttonToggled.connect(self.on_mode_radio_toggled)

        self.toolbar.add_separator()
        
        # 2. OPTIONS 섹션
        lbl_opt = QLabel("OPTIONS"); lbl_opt.setProperty("header", "true")
        layout.addWidget(lbl_opt)
        
        row_guide, self.chk_crosshair = create_custom_option("Show Guide")
        self.chk_crosshair.setChecked(True)
        self.chk_crosshair.toggled.connect(self.input_canvas.set_show_crosshair)
        layout.addWidget(row_guide)
        
        btn_clr = QPushButton("Clear Mask")
        btn_clr.setProperty("simple_btn", "true")
        btn_clr.clicked.connect(self.clear_input_mask)
        layout.addWidget(btn_clr)

        self.toolbar.show()
        self.toolbar.move(20, 20)
        
        # 기본값 설정
        self.rb_box.setChecked(True)

    def on_input_selection_changed(self, tool_type, data):
        if self.image is None:
            return
            
        mask = self.generate_mask_from_canvas()
        
        # 이미지 채널 수(RGB 또는 RGBA)에 따른 분기 처리
        if self.image.shape[2] == 4:
            preview = np.zeros((self.image.shape[0], self.image.shape[1], 4), dtype=np.uint8)
            preview[:, :, 3] = 255 
        else:
            preview = np.zeros_like(self.image)
            
        # 채널에 맞게 색상 배열 할당 (4채널이면 RGBA, 3채널이면 RGB)
        preview[mask == 255] = [255, 255, 255, 255] if preview.shape[2] == 4 else [255, 255, 255]
        
        self.mask_canvas.set_image(preview)
        self.mask_canvas.fit_to_window()
        self.status.showMessage(f"Mask Updated ({tool_type})", 1000)
        
    def clear_image(self):
        self.image = None
        self.external_mask = None
        if hasattr(self, 'input_canvas') and self.input_canvas:
            self.input_canvas.set_image(None)
            self.input_canvas.reset_selection()
        if hasattr(self, 'mask_canvas') and self.mask_canvas:
            self.mask_canvas.set_image(None)
        self.chk_use_image.setChecked(False)
        self.chk_use_mask.setChecked(False)
        gc.collect()
    
    def clear_input_mask(self):
        self.input_canvas.reset_selection()
        if self.image is not None:
            self.mask_canvas.set_image(np.zeros_like(self.image))
    
    def clear_all_masks(self):
        self.external_mask = None
        self.input_canvas.clear_all_overlays() 
        self.input_canvas.set_overlay_mask(None) 
        self.input_canvas.repaint()
        if self.image is not None:
            self.on_input_selection_changed("Clear All", None)
        else:
            self.mask_canvas.set_image(None)
        self.chk_use_mask.setChecked(False)

    # -----------------------------------------------------------
    # [누락된 함수 추가] 이 부분을 BgComposerApp 클래스 안에 넣어주세요
    # -----------------------------------------------------------
    def clear_external_mask(self):
        """외부에서 불러온 마스크만 제거"""
        self.external_mask = None
        # 마스크 사용 체크 해제
        self.chk_use_mask.setChecked(False)
        # 화면 갱신
        self.on_input_selection_changed("Clear External", None)
        
    def open_upscale_settings(self):
        dlg = UpscaleSettingsDialog(self, self.upscale_settings)
        if dlg.exec(): 
            self.upscale_settings = dlg.get_settings()

    def open_token_settings(self):
        old_hf_token = self.hf_token
        dlg = KeySettingsDialog(self)
        if dlg.exec():
            new_hf_token = token_key.get_valid_hf_token()
            new_api_key = token_key.get_valid_api_key()
            if new_hf_token != old_hf_token:
                self.hf_token = new_hf_token
                if self.hf_token:
                    try:
                        from huggingface_hub import login
                        login(token=self.hf_token)
                    except: pass
            if new_api_key != getattr(self, 'api_key', None):
                self.api_key = new_api_key
                if hasattr(self, 'diffusion_estimator') and self.diffusion_estimator:
                    self.diffusion_estimator.api_key = new_api_key
                
    def on_mode_radio_toggled(self, button, checked):
        """ 도구 모드 변경 핸들러 (복구됨) """
        if not checked: return
        
        # 팬 버튼이 켜져있다면 끄기
        if hasattr(self, 'btn_pan_float') and self.btn_pan_float.isChecked():
            self.btn_pan_float.blockSignals(True)
            self.btn_pan_float.setChecked(False)
            self.btn_pan_float.blockSignals(False)
            
        mode_map = {self.rb_box: "box", self.rb_lasso: "lasso", self.rb_brush: "brush"}
        
        if button in mode_map:
            new_mode = mode_map[button]
            self.input_canvas.set_mode(new_mode)
            
            # 브러시 모드일 때만 슬라이더 보이기
            if hasattr(self, 'brush_size_container'):
                self.brush_size_container.setVisible(new_mode == "brush")
                if hasattr(self, 'toolbar'):
                    self.toolbar.adjustSize()
                
            self.status.showMessage(f"Mode changed: {new_mode.upper()}", 1000)

    def toggle_pan(self, checked):
        if checked: 
            self.input_canvas.set_mode("pan")
            # 라디오 버튼 배타성 일시 해제 후 체크 해제
            self.mode_bg.setExclusive(False)
            self.rb_box.setChecked(False)
            self.rb_lasso.setChecked(False)
            self.rb_brush.setChecked(False)
            self.mode_bg.setExclusive(True)
        else:
            # 팬 모드 해제 시 박스 모드로 복귀
            self.rb_box.setChecked(True)
            self.input_canvas.set_mode("box")

    def sync_scrollbars(self):
        cv = self.input_canvas
        if cv.image is None: return
        vw, vh = cv.width(), cv.height()
        cw, ch = cv.img_w * cv.scale, cv.img_h * cv.scale
        self.scroll_h.setEnabled(cw > vw)
        self.scroll_v.setEnabled(ch > vh)
        if cw > vw:
            self.scroll_h.setRange(0, int(cw-vw))
            self.scroll_h.setPageStep(int(vw))
            self.scroll_h.setValue(-int(cv.offset_x))
        if ch > vh:
            self.scroll_v.setRange(0, int(ch-vh))
            self.scroll_v.setPageStep(int(vh))
            self.scroll_v.setValue(-int(cv.offset_y))

    def on_scrollbar_action(self):
        if self.input_canvas.image is None: return
        if self.scroll_h.isEnabled(): self.input_canvas.offset_x = -float(self.scroll_h.value())
        if self.scroll_v.isEnabled(): self.input_canvas.offset_y = -float(self.scroll_v.value())
        self.input_canvas.update()

    def setup_shortcuts(self):
        act_space = QAction(self)
        act_space.setShortcut("Space")
        act_space.setShortcutContext(Qt.WindowShortcut) 
        act_space.triggered.connect(self.on_pan_shortcut_triggered)
        self.addAction(act_space)

    def on_pan_shortcut_triggered(self):
        """ 스페이스바 단축키 핸들러 (복구됨) """
        # 마우스가 어느 캔버스 위에 있는지 확인하고 해당 캔버스의 팬 모드 토글
        if self.input_canvas and self.input_canvas.underMouse():
            if hasattr(self, 'btn_pan_float'):
                self.btn_pan_float.animateClick()
        elif self.mask_canvas and self.mask_canvas.underMouse():
            if hasattr(self, 'btn_pan_mask'):
                self.btn_pan_mask.animateClick()
        elif self.result_canvas and self.result_canvas.underMouse():
            if hasattr(self, 'btn_pan_result'):
                self.btn_pan_result.animateClick()

    @staticmethod
    def _task_load_image(path, abort_check, shared_buffer=None, **kwargs):
        if abort_check(): raise RuntimeError("USER_CANCEL")
        pil = Image.open(path).convert("RGBA")
        pil = ImageOps.exif_transpose(pil)
        max_d = 3840
        w, h = pil.size
        if max(w, h) > max_d:
            pil = pil.resize((int(w * max_d / max(w, h)), int(h * max_d / max(w, h))), Image.BILINEAR)
        try:
            qimage = QImage(pil.tobytes("raw", "RGBA"), pil.width, pil.height, QImage.Format_RGBA8888).copy()
        except:
            qimage = qimage_from_ndarray(np.array(pil)).copy()
        if shared_buffer is not None:
            shared_buffer['img_np'] = np.array(pil)
            shared_buffer['qimage'] = qimage
        return True

    def _on_image_loaded(self, success, fpath):
        self.toggle_loading(False)
        self.worker = None
        if not success: return
        img, qimg = self._image_shared_buffer.get('img_np'), self._image_shared_buffer.get('qimage')
        self._image_shared_buffer = {}
        self._apply_loaded_image(fpath, img, qimg)

    def _apply_loaded_image(self, fpath, img, qimg):
        self.image = img
        self.input_canvas.set_image(self.image, preloaded_qimage=qimg)
        self.external_mask = None
        self.input_canvas.fit_to_window()
        self.output_dir = get_output_dir(f"bg_local_{Path(fpath).stem}")
        if hasattr(self, 'camera_controller'): self.camera_controller.set_thumbnail(self.image)
        QTimer.singleShot(0, self.update_gallery)
        self.toolbar.show()
        self.log(f"Opened: {Path(fpath).name}")

    def open_image(self):
        if self.worker and self.worker.isRunning(): return
        with SuppressStderr():
            fpath, _ = QFileDialog.getOpenFileName(self, "Open", "", "Images (*.png *.jpg *.jpeg)")
        if fpath:
            self._cancel_event.clear()
            self.toggle_loading(True, "Loading", "Reading file...")
            self._image_shared_buffer = {}
            self.worker = GenericWorker(lambda p, **k: self._task_load_image(p, shared_buffer=self._image_shared_buffer, **k), fpath, abort_check=self._should_cancel)
            self.worker.signal_finished.connect(lambda r: self._on_image_loaded(r, fpath))
            self.worker.finished.connect(lambda: self._stop_worker('worker'))
            self.worker.start()

    def apply_smart_defaults(self):
        if self.hw_info.get("is_pascal"): self.chk_upscale.setChecked(False)

    def update_model_list(self):
        prev = self.combo_model.currentData()
        self.combo_model.clear()
        mode = "remote" if self.rb_remote.isChecked() else "local"
        
        models = []
        for k, v in self.generation_models_dict.items():
            if not v.get("use", True): continue
            if mode == "remote" and not v.get("remote_url"): continue
            
            imode = v.get("mode", "local")
            if (mode=="local" and imode in ["local","both"]) or (mode=="remote" and imode in ["remote","both"]):
                models.append(v)
        
        models.sort(key=lambda x: (not x.get('is_default', False), x['short_name']))
        for m in models:
            self.combo_model.addItem(m['short_name'], userData=m['key'])
        if prev: 
            idx = self.combo_model.findData(prev)
            if idx >= 0: self.combo_model.setCurrentIndex(idx)
        self.on_model_combo_changed()

    @staticmethod
    def _task_load_model(model_key, precision, quantization, estimator, abort_check, model_config=None, **kwargs):
        if estimator.is_ready: estimator.unload_model()
        return estimator.load_model(model_key, precision=precision, quantization=quantization, abort_check=abort_check, model_config=model_config)

    def on_load_model_request(self):
        key = self.combo_model.currentData()
        if not key or (self.worker and self.worker.isRunning()): return
        mi = self.generation_models_dict.get(key)
        self._active_model_config = mi

        # --- [예외 처리 추가] API Key 및 토큰 사전 검증 ---
        is_remote = self.rb_remote.isChecked()
        provider = mi.get("provider", "")
        
        if is_remote:
            # Remote API일 경우 API Key 필수 체크
            if provider == "google_genai" and not token_key.get_valid_api_key():
                QMessageBox.warning(self, "API Key 누락", "Google API Key가 설정되지 않았습니다.\n상단의 [API Key] 버튼을 눌러 키를 설정해주세요.")
                return
        else:
            # Local 모델일 경우 HuggingFace 토큰 체크 (다운로드 시 터미널 멈춤 방지)
            if not token_key.get_valid_hf_token():
                reply = QMessageBox.question(self, "HuggingFace Token 누락", 
                    "HuggingFace 토큰이 설정되지 않았습니다.\n"
                    "가중치 파일 다운로드가 필요한 경우, 터미널에서 입력을 기다리며 UI가 무한 로딩에 빠질 수 있습니다.\n\n"
                    "그래도 다운로드가 이미 되어있다고 가정하고 계속 진행하시겠습니까?", 
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.No:
                    self.open_token_settings() # API 설정창 바로 띄워주기
                    return
        # ------------------------------------------------
        
        if "Qwen" in mi.get("pipeline_type", "") and not self.rb_remote.isChecked():
            if self.hw_info.get("is_pascal") or self.hw_info.get("vram_gb", 0) <= 8.5:
                return QMessageBox.critical(self, "Incompatible", "Local Qwen requires RTX GPU with 12GB+ VRAM.")

        if self.rb_remote.isChecked():
            self.current_model_mode = "remote"
            if self.diffusion_estimator.is_ready: self.diffusion_estimator.unload_model()
            self.on_load_finished(True)
            return

        self.current_model_mode = "local"
        if self.diffusion_estimator.is_ready:
            self.lbl_model_status.setText("Cleaning...")
            QApplication.processEvents()
            self.diffusion_estimator.unload_model()

        self._stop_worker('worker')
        self._cancel_event.clear()
        self.toggle_loading(True, "Loading Model", f"Loading {mi['short_name']}...")
        self._t_load_start = time.time()
        
        self.worker = GenericWorker(
            self._task_load_model, model_key=key, precision="auto", quantization="none",
            estimator=self.diffusion_estimator, abort_check=self._should_cancel, model_config=mi
        )
        self.worker.signal_finished.connect(self.on_load_finished)
        self.worker.finished.connect(lambda: self._stop_worker('worker'))
        self.worker.start()
        
    def on_load_finished(self, success):
        self.toggle_loading(False)
        self.lbl_model_status.setText("✔ Ready" if success else "❌ Failed")
        self.lbl_model_status.setStyleSheet(f"QLabel {{ border: 1px solid {'#555' if success else '#c0392b'}; color: {'#2ecc71' if success else '#e74c3c'}; background-color: #2c3e50; font-weight: bold; padding: 6px 24px; }}")

    def on_camera_prompt_update(self, camera_text):
        if not self.chk_multi_angle.isChecked(): return
        current_text = self.txt_prompt.toPlainText()
        pattern = r"\s*<camera>.*?</camera>\s*"
        injection_text = f" {camera_text} " 
        if "<camera>" in current_text:
            new_text = re.sub(pattern, injection_text, current_text, flags=re.DOTALL)
        else:
            new_text = f"{current_text.strip()} {injection_text.strip()}"
        self.txt_prompt.setPlainText(new_text)

    def on_retranslate_requested(self):
        p = self.txt_prompt.toPlainText()
        n = self.txt_negative.toPlainText()
        if p or n: self._run_preview_translation(p, n)

    def _run_preview_translation(self, p, n):
        self._blink_state = True
        self.blink_timer.start()
        self.txt_trans_result.setPlainText("Translating...")
        self._stop_worker('preview_worker')
        self.preview_worker = GenericWorker(lambda p, n, **k: (translator.translate(p), translator.translate(n)), p, n, abort_check=lambda: False)
        self.preview_worker.signal_finished.connect(self._on_preview_translation_done)
        self.preview_worker.finished.connect(lambda: self._stop_worker('preview_worker'))
        self.preview_worker.start()

    def _on_preview_translation_done(self, res):
        self.blink_timer.stop()
        if not res:
            self.txt_trans_result.setPlainText("Failed.")
            return
        p_en, n_en = res
        self.chk_manual_prompt.isChecked()
        
        # 간소화된 프롬프트 뷰어
        self.txt_trans_result.setHtml(f"<b>P:</b> {p_en}<br><b>N:</b> {n_en}")

    def update_word_counts(self, *args):
        model_key = self.combo_model.currentData()
        model_config = self.generation_models_dict.get(model_key, {})
        is_qwen_or_gemini = "qwen" in str(model_config).lower() or "gemini" in str(model_config).lower()
        limit = 2000 if is_qwen_or_gemini else 75
        
        for txt, lbl, pre in [(self.txt_prompt, self.lbl_p_count, "P"), (self.txt_negative, self.lbl_n_count, "N")]:
            cnt = len(txt.toPlainText().strip().split())
            color = "#ff6b6b" if cnt > limit else "#ecf0f1"
            lbl.setText(f"<b style='color:#74b9ff'>{pre}:</b> <span style='color:{color}'>{cnt}/{limit}</span>")
    
    def on_scenario_changed(self):
        k = self.combo_scenario.currentText()
        if k in self.scenarios:
            d = self.scenarios[k]
            self.txt_prompt.setText(d.get("prompt",""))
            self.txt_negative.setText(d.get("negative",""))

    def generate_mask_from_canvas(self):
        if self.image is None: return None
        h, w = self.image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if self.image.shape[2] == 4:
            mask = cv2.bitwise_or(mask, (255 - self.image[:,:,3]).astype(np.uint8))
        if self.input_canvas.box_img:
            l, t, r, b = self.input_canvas.box_img
            cv2.rectangle(mask, (int(l), int(t)), (int(r), int(b)), 255, -1)
        if len(self.input_canvas.lasso_img) > 0:
            cv2.fillPoly(mask, [np.array(self.input_canvas.lasso_img, dtype=np.int32)], 255)
        if hasattr(self.input_canvas, 'brush_strokes') and self.input_canvas.brush_strokes:
            for stroke in self.input_canvas.brush_strokes:
                points = stroke['points']
                size = stroke['size']
                if len(points) > 1:
                    pts = np.array(points, dtype=np.int32)
                    for i in range(len(points) - 1):
                        cv2.line(mask, (int(points[i][0]), int(points[i][1])), (int(points[i+1][0]), int(points[i+1][1])), 255, thickness=int(size))
                        cv2.circle(mask, (int(points[i][0]), int(points[i][1])), int(size/2), 255, -1)
                    cv2.circle(mask, (int(points[-1][0]), int(points[-1][1])), int(size/2), 255, -1)
                elif len(points) == 1:
                    cv2.circle(mask, (int(points[0][0]), int(points[0][1])), int(size/2), 255, -1)

        if self.external_mask is not None:
            mask = cv2.bitwise_or(mask, self.external_mask)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask

    def open_external_mask(self):
        if self.image is None: return
        with SuppressStderr():
            fname, _ = QFileDialog.getOpenFileName(self, "Open Mask", "", "Images (*.png *.jpg *.bmp)")
        if fname:
            m = Image.open(fname).convert("L").resize((self.image.shape[1], self.image.shape[0]), Image.NEAREST)
            _, self.external_mask = cv2.threshold(np.array(m), 127, 255, cv2.THRESH_BINARY)
            self.chk_use_mask.setChecked(True)
            self.on_input_selection_changed("External", None)

    def cancel_generation(self):
        self._cancel_event.set()

    def _should_cancel(self):
        return self._cancel_event.is_set()

    def _task_smart_generation(self, model_cfg, mode_ui, is_remote_forced, **kwargs):
        execution_mode = "remote" if is_remote_forced or model_cfg.get("mode") == "remote" else "local"
        
        if execution_mode == "local":
            try:
                return self.diffusion_estimator.predict(**kwargs)
            except Exception as e:
                print(f"Local failed: {e}. Switching to Remote.")
                execution_mode = "remote"
        
        if execution_mode == "remote":
            gen_pil = self.call_remote_api(model_cfg=model_cfg, **kwargs)
            # kwargs['image']는 numpy 배열, Manual Post Process를 위해 전달
            # use_input_image가 False여도 합성할 배경이 있으면 전달 필요할 수 있으나 기본 로직 유지
            post_mask = kwargs.get("mask") if kwargs.get("use_mask") else None
            return self.diffusion_estimator.manual_post_process(
                gen_pil, kwargs.get("image"), post_mask, upscale_opts=kwargs.get("upscale_opts")
            )
        return None

    def on_model_combo_changed(self, index=None):
        model_key = self.combo_model.currentData()
        if not model_key: return
        model_config = self.generation_models_dict.get(model_key, {})
        self._active_model_config = model_config
        
        options = [x.strip().lower() for x in model_config.get("options", [])]
        
        # Multi-Angle 체크
        has_multi_angle = "multi-angle" in options or "multi_angle" in options
        self.chk_multi_angle.setEnabled(has_multi_angle)
        self.chk_multi_angle.setChecked(has_multi_angle)
        
        # Multi-Image 체크 (Qwen)
        has_multi_image = "multi-image" in options
        self.multi_image_group.setVisible(has_multi_image)
        
        # 이미지 필수 여부
        should_check_image = True # Qwen, DreamShaper 모두 이미지 권장
        self.chk_use_image.setChecked(should_check_image)
        self.chk_use_mask.setEnabled(should_check_image)
        
        self.update_word_counts()
        self.lbl_model_status.setText("Load Needed")

    def on_manual_prompt_toggled(self, checked):
        self.chk_translate.setEnabled(not checked)
        self.btn_retrans.setEnabled(True)

    def on_multi_angle_toggled(self, state):
        is_checked = (state == Qt.Checked) if isinstance(state, int) else state
        idx = self.tabs.indexOf(self.camera_controller)
        if is_checked and idx == -1: self.tabs.insertTab(0, self.camera_controller, "Camera")
        elif not is_checked and idx != -1: self.tabs.removeTab(idx)

    def run_generation(self):
        if self.worker and self.worker.isRunning(): return
        if not self._active_model_config: return QMessageBox.warning(self, "Not Ready", "Model not loaded")

        # --- [예외 처리 추가] 생성 전 API Key / 토큰 다시 검증 ---
        is_remote = self.rb_remote.isChecked()
        provider = self._active_model_config.get("provider", "")
        
        if is_remote:
            if provider == "google_genai" and not token_key.get_valid_api_key():
                QMessageBox.warning(self, "API Key 누락", "Google API Key가 설정되지 않았습니다.\n[API Key] 버튼을 눌러 키를 설정해주세요.")
                self.open_token_settings()
                return
            elif provider == "fal_ai" and not token_key.get_valid_api_key():
                QMessageBox.warning(self, "API Key 누락", "Fal AI API Key가 설정되지 않았습니다.\n[API Key] 버튼을 눌러 키를 설정해주세요.")
                self.open_token_settings()
                return
        # --------------------------------------------------------

        p_raw = self.txt_prompt.toPlainText().strip()
        n_raw = self.txt_negative.toPlainText().strip()
        
        # Manual Mode면 번역 스킵 (기존 호환성 유지)
        if hasattr(self, 'chk_manual_prompt') and self.chk_manual_prompt.isChecked():
            p_txt, n_txt = p_raw, n_raw
        else:
            # === 변경된 부분: 무조건 번역 수행 ===
            p_txt = translator.translate(p_raw) if p_raw else ""
            n_txt = translator.translate(n_raw) if n_raw else ""
                
            # 태그 변환
            if hasattr(self, 'chk_multi_angle') and self.chk_multi_angle.isChecked() and "<camera>" in p_raw:
                p_txt = self._convert_camera_tag_to_sks(p_txt)

        final_rgb = self.image[:,:,:3] if (self.chk_use_image.isChecked() and self.image is not None) else None
        final_mask = self.generate_mask_from_canvas() if (final_rgb is not None and self.chk_use_mask.isChecked()) else None
        
        target_imgs = []
        if final_rgb is not None: target_imgs.append(final_rgb)
        for f in self.multi_images:
            try: target_imgs.append(np.array(Image.open(f).convert("RGB")))
            except: pass

        self._stop_worker('worker')
        self._cancel_event.clear()
        self.toggle_loading(True, "Generating", "Processing...")
        
        self.worker = GenericWorker(
            self._task_smart_generation,
            model_cfg=self._active_model_config,
            mode_ui="remote" if self.rb_remote.isChecked() else "local",
            is_remote_forced=self.rb_remote.isChecked(),
            image=final_rgb, mask=final_mask,
            pil_images=[Image.fromarray(i) for i in target_imgs],
            prompt=p_txt, negative_prompt=n_txt,
            num_inference_steps=self.spin_steps.value(),
            guidance_scale=self.spin_cfg.value(),
            upscale_opts=self.upscale_settings if self.chk_upscale.isChecked() else None,
            resolution_mode=self.combo_res_mode.currentData(),
            use_input_image=(final_rgb is not None),
            use_mask=(final_mask is not None),
            abort_check=self._should_cancel,
            hf_token=token_key.get_valid_hf_token(),
            api_key=token_key.get_valid_api_key()
        )
        self.worker.signal_progress.connect(self.on_progress_update)
        self.worker.signal_finished.connect(self.on_gen_finished)
        self.worker.finished.connect(lambda: self._stop_worker('worker'))
        self.worker.start()
    
    def on_progress_update(self, percent):
        self.loading_overlay.set_message("GENERATING", f"Processing... ({percent}%)")

    def on_gen_finished(self, result):
        self.toggle_loading(False)
        if result is not None:
            self.result_image = result
            self.result_canvas.set_image(result)
            self.result_canvas.fit_to_window()
            save_image_file(result, f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", str(self.output_dir))
            self.update_gallery()
        else:
            QMessageBox.warning(self, "Failed", "Generation returned no image.")
        gc.collect()

    def on_gallery_double_clicked(self, item):
        self.result_canvas.load_from_file(item.data(Qt.UserRole))

    def update_gallery(self, max_items=50):
        if self.gallery_worker and self.gallery_worker.isRunning(): return
        self.gallery_worker = GalleryLoadWorker(self.output_dir, max_items, {})
        self.gallery_worker.signal_finished.connect(self._on_gallery_updated)
        self.gallery_worker.start()

    def _on_gallery_updated(self, results):
        self.gallery_list.clear()
        for path, name, _, qimg in results:
            it = QListWidgetItem(QIcon(QPixmap.fromImage(qimg)) if qimg else QIcon(), name)
            it.setData(Qt.UserRole, path)
            self.gallery_list.addItem(it)

    def log(self, msg, switch_tab=True):
        self.log_table.insertRow(0)
        self.log_table.setItem(0,0,QTableWidgetItem(datetime.now().strftime("%H:%M:%S")))
        self.log_table.setItem(0,1,QTableWidgetItem(str(msg)))
        if switch_tab: self.tabs.setCurrentWidget(self.log_table)
        self.status.showMessage(msg, 5000)

    def toggle_loading(self, show, title="Processing", desc="Wait..."):
        if show:
            self.loading_overlay.set_message(title, desc)
            self.loading_overlay.show()
            self.set_ui_enabled(False)
        else:
            self.loading_overlay.hide()
            self.set_ui_enabled(True)

    def set_ui_enabled(self, enabled):
        self.btn_load.setEnabled(enabled)
        self.btn_gen.setEnabled(enabled)
        self.input_canvas.setEnabled(enabled)

    def _on_steps_manually_changed(self): self._user_touched_steps = True
    def _on_cfg_manually_changed(self): self._user_touched_cfg = True
    def _update_blinking_message(self): pass

    def on_worker_error(self, e):
        self.toggle_loading(False)
        QMessageBox.critical(self, "Error", str(e))

    def save_result(self):
        if self.result_image is None: return
        f, _ = QFileDialog.getSaveFileName(self, "Save", str(self.output_dir), "PNG (*.png)")
        if f: Image.fromarray(self.result_image).save(f)

    def closeEvent(self, e):
        if self.diffusion_estimator.is_ready: self.diffusion_estimator.unload_model()
        e.accept()
    
    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, 'loading_overlay'): self.loading_overlay.resize(self.centralWidget().size())

    def _parse_camera_tag(self, prompt):
        match = re.search(r"<camera>(.*?)</camera>", prompt, re.DOTALL)
        if not match: return None, (0.0, 0.0, 1.0)
        h_match = re.search(r"horizontal\s*[=:]\s*([-\d\.]+)", match.group(1), re.I)
        v_match = re.search(r"vertical\s*[=:]\s*([-\d\.]+)", match.group(1), re.I)
        z_match = re.search(r"zoom\s*[=:]\s*([-\d\.]+)", match.group(1), re.I)
        h = float(h_match.group(1)) if h_match else 0.0
        v = float(v_match.group(1)) if v_match else 0.0
        z = float(z_match.group(1)) if z_match else 1.0
        return match, (h, v, z)

    def _convert_camera_tag_to_sks(self, prompt):
        match, (h, v, z) = self._parse_camera_tag(prompt)
        if not match: return prompt
        # Simplified LoRA Trigger Mapping
        return f"<sks> front view, eye-level shot, medium shot, {prompt.replace(match.group(0), '')}"

    def _convert_camera_tag_for_gemini(self, prompt):
        match, (h, v, z) = self._parse_camera_tag(prompt)
        if not match: return prompt
        return f"Front view camera at eye level. {prompt.replace(match.group(0), '')}"

    def get_hf_client(self, space_url, hf_token=None):
        if not space_url: return None
        key = (space_url, bool(hf_token))
        if key not in HF_CLIENT_CACHE:
            HF_CLIENT_CACHE[key] = Client(space_url, token=hf_token)
        return HF_CLIENT_CACHE[key]

    def call_remote_api(self, *, model_cfg, **kwargs):
        provider = model_cfg.get("provider", "hf_space")
        if provider == "google_genai": return self.call_gemini_api(model_cfg=model_cfg, **kwargs)
        elif provider == "fal_ai":
            return self.call_fal_ai_api(
                model_id=model_cfg.get("repo_id"), 
                token=kwargs.get("api_key") if "fal" in model_cfg.get("remote_url") else kwargs.get("hf_token"),
                model_cfg=model_cfg, **kwargs
            )
        elif provider == "hf_space": return self.call_hf_space_api(model_cfg=model_cfg, **kwargs)
        raise ValueError(f"Unknown provider: {provider}")

    def call_gemini_api(self, *, model_cfg, pil_images, prompt, **kwargs):
        """ Gemini API 호출 (에러 핸들링 추가됨) """
        try:
            from google.genai import Client
            from google.genai.errors import ServerError, ClientError
        except ImportError:
            raise ImportError("Google GenAI SDK가 설치되지 않았습니다. (pip install google-genai)")

        api_key = kwargs.get("api_key")
        if not api_key: 
            raise ValueError("Google API Key가 설정되지 않았습니다. [API Key] 버튼을 눌러 키를 입력해주세요.")

        client = Client(api_key=api_key)
        
        # 프롬프트 구성
        contents = [prompt]
        if pil_images:
            contents.extend(pil_images)

        target_model = model_cfg.get("api_model_uri", "gemini-1.5-pro")
        
        try:
            print(f"[Gemini] Requesting to {target_model}...")
            resp = client.models.generate_content(
                model=target_model,
                contents=contents
            )
            
            # 응답 처리 및 이미지 추출
            if resp.candidates and resp.candidates[0].content.parts:
                for part in resp.candidates[0].content.parts:
                    if part.inline_data:
                        return Image.open(BytesIO(part.inline_data.data)).convert("RGB")
            
            print("[Gemini] No image found in response.")
            return None

        except ServerError as e:
            # 503 등 서버 측 에러 처리
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                raise RuntimeError("Google 서버가 현재 혼잡하여 요청을 처리할 수 없습니다 (503). 잠시 후(1~2분 뒤) 다시 시도해주세요.")
            else:
                raise RuntimeError(f"Google 서버 오류: {e}")

        except ClientError as e:
            # 400 등 클라이언트 요청 에러
            raise RuntimeError(f"요청이 거부되었습니다 (설정/쿼터 확인): {e}")

        except Exception as e:
            raise RuntimeError(f"Gemini API 호출 실패: {e}")
        
    def call_fal_ai_api(self, *, model_id, pil_images, prompt, num_inference_steps, guidance_scale, token, **kwargs):
        # Fal-AI Logic
        pass 

    def call_hf_space_api(self, *, model_cfg, pil_images, mask, prompt, negative_prompt, num_inference_steps=30, hf_token=None, abort_check=None, **kwargs):
        client = self.get_hf_client(model_cfg.get("remote_url"), hf_token=hf_token)
        pipeline_type = model_cfg.get("pipeline_type", "").lower()
        api_name = model_cfg.get("api_model_uri", "/infer")
        
        # 1. Qwen-Edit-2511
        if "qwen" in pipeline_type:
            if not pil_images: raise ValueError("Image required")
            # Qwen Payload Placeholder
            return Image.new("RGB", (1024, 1024))

        # 2. General SD Inpaint
        elif "inpaint" in pipeline_type:
            # DreamShaper Remote Fallback
            predict_kwargs = {
                "prompt": prompt, "negative_prompt": negative_prompt, "num_inference_steps": num_inference_steps,
                "guidance_scale": kwargs.get("guidance_scale", 7.5), "width": kwargs.get("width", 512), "height": kwargs.get("height", 512),
                "api_name": api_name
            }
            if pil_images:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                    pil_images[0].save(tf.name)
                    predict_kwargs["image"] = handle_file(tf.name)
            if mask is not None:
                 with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                    Image.fromarray(mask).save(tf.name)
                    predict_kwargs["mask_image"] = handle_file(tf.name)
                    
            try:
                res = client.predict(**predict_kwargs)
                if isinstance(res, str): return Image.open(res).convert("RGB")
                if isinstance(res, dict) and "image" in res: return Image.open(res["image"]).convert("RGB")
                return Image.open(res).convert("RGB")
            except Exception as e:
                print(f"HF Space Error: {e}")
                return Image.new("RGB", (512, 512))

        return Image.new("RGB", (512, 512)) # Fallback

if __name__ == "__main__":
    app = QApplication(sys.argv)
    qdarktheme.setup_theme("dark")
    app.setFont(QFont("Malgun Gothic", 10))
    win = BgComposerApp()
    win.show()
    sys.exit(app.exec())