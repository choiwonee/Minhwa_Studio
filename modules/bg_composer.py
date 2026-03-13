import base64
import gc
import httpx
import os
import re
import sys
import time
import random
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

from google import genai
from google.genai import types
from gradio_client import Client, handle_file
from PIL import Image, ImageOps

# GUI Framework
from PySide6.QtWidgets import (
    QApplication, QDialog, QFormLayout, QFrame, QMainWindow, QLabel, QFileDialog, QPushButton, QWidget,
    QVBoxLayout, QHBoxLayout, QComboBox, QMessageBox, QGroupBox,
    QRadioButton, QButtonGroup, QDoubleSpinBox, QScrollBar, QScrollArea, QGridLayout, QTextEdit,
    QCheckBox, QStatusBar, QListWidget, QListWidgetItem, QSizePolicy, QSlider, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QSpinBox
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
from models.diffusion_estimator import DiffusionEstimator
from utils.gui_utils import GenericWorker, FloatingToolBar, ImageCanvas, ProcessingOverlay, VisualCameraWidget, KeySettingsDialog
from utils.rag_prompter import RAGPrompter
from utils.prompt_engine import patch_bg_composer

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

        # --- RAG 초기화 추가 ---
        self.rag_prompter = RAGPrompter()
        threading.Thread(target=self.rag_prompter.build_index, daemon=True).start()
        # -----------------------
        
        # PromptEngine 단일 인스턴스로 통일:
        # patch_bg_composer()가 내부에서 PromptEngine()을 새로 생성해 self._prompt_engine에 저장하므로,
        # 먼저 patch를 실행한 뒤 self.prompt_engine이 동일 인스턴스를 참조하도록 한다.
        patch_bg_composer(self)                        # → self._prompt_engine 생성
        self.prompt_engine = self._prompt_engine       # 동일 인스턴스 참조 (중복 생성 방지)
        
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
        """ 1. INPUT IMAGE 패널 UI 설정
            - 이미지 캔버스, 스크롤바 래퍼 및 플로팅 도구 모음 구성
            - 캔버스 하단에 Fit, 1:1 뷰어 제어 바(Control Bar)
        """
        content_container = QWidget()
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        self.input_canvas = ImageCanvas(self)
        self.input_canvas.set_mode("box")
        self.input_canvas.on_selection_done = self.on_input_selection_changed
        self.input_canvas.sig_view_changed.connect(self.sync_scrollbars)
        
        canvas_wrapper = self._wrap_canvas_with_scrollbars(self.input_canvas)
        content_layout.addWidget(canvas_wrapper, 1)
        
        # 캔버스 하단 Fit, 1:1 버튼이 포함된 컨트롤 바
        ctrl_bar = self._create_view_control_bar(self.input_canvas, show_pan=False)
        content_layout.addWidget(ctrl_bar, 0)
        
        # 도구 툴바 및 팬 버튼 설정
        self.setup_floating_toolbar()
        self.btn_pan_float = self.setup_floating_pan_button(self.input_canvas)
        
        return self._create_styled_panel("1. INPUT IMAGE (Draw Mask)", content_container)

    def _setup_mask_panel(self):
        content_container = QWidget()
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self.mask_canvas = ImageCanvas(self)
        self.mask_canvas.set_mode("view")
        self.mask_canvas.setStyleSheet("background-color: #141414;")

        self.mask_canvas.set_show_crosshair(False)
        
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

        self.result_canvas.set_show_crosshair(False)

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

        self.camera_scroll_area = QScrollArea()
        self.camera_scroll_area.setWidget(self.camera_controller)
        self.camera_scroll_area.setWidgetResizable(True) # 크기 유동적 조절 허용
        self.camera_scroll_area.setFrameShape(QFrame.NoFrame)
            
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
        """ 하단 제어 패널(Control Panel) 생성 및 UI 레이아웃 구성
            - 좌측: 기본 설정 및 상세 파라미터 탭
            - 우측: 프롬프트 입력, 번역 제어 및 생성 액션 버튼
            - 수동 프롬프트(Manual Prompt) 활성화 UI를 프롬프트 입력 영역 상단에 노출
        """
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

        # [LEFT] 설정 탭 (Settings)
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
        
        # Mode
        self.bg_mode = QButtonGroup(self)
        self.rb_local = QRadioButton("Local")
        self.rb_remote = QRadioButton("Remote")
        self.bg_mode.addButton(self.rb_local); self.bg_mode.addButton(self.rb_remote)
        self.rb_local.setChecked(True)
        l_mode.addWidget(QLabel("Mode:"))
        l_mode.addWidget(self.rb_local)
        l_mode.addWidget(self.rb_remote)
        l_mode.addStretch()
        
        self.btn_token_conf = QPushButton("API Key")
        self.btn_token_conf.setCursor(Qt.PointingHandCursor)
        self.btn_token_conf.setToolTip("HuggingFace / Gemini API 키 설정")
        self.btn_token_conf.clicked.connect(self.open_token_settings)
        self.btn_token_conf.setStyleSheet("color: #ecc058; border: 1px solid #777;") 
        l_mode.addWidget(self.btn_token_conf)

        self.bg_mode.buttonClicked.connect(self.update_model_list)
        lb.addWidget(grp_mode)

        # 2. 모델 선택 (Grid)
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
        
        grp_model.addWidget(QLabel("Model:"), 0, 0)
        grp_model.addWidget(self.combo_model, 0, 1, 1, 2)
        grp_model.addWidget(self.btn_load, 1, 0, 1, 2)
        grp_model.addWidget(self.lbl_model_status, 1, 2)
        lb.addLayout(grp_model)

        # 3. 입력 소스
        grp_src = QHBoxLayout()
        grp_src.setSpacing(6)
        
        # Image Source
        self.chk_use_image = QCheckBox("Image")
        self.chk_use_image.setToolTip("Enable Input Image")
        
        btn_image = QPushButton("Open")
        btn_image.clicked.connect(self.open_image)
        
        self.btn_image_clear = QPushButton("Clear")
        self.btn_image_clear.setCursor(Qt.PointingHandCursor)
        self.btn_image_clear.setToolTip("입력 이미지 초기화")
        self.btn_image_clear.setStyleSheet("color: #e74c3c; font-weight: bold; border: 1px solid #555;")
        self.btn_image_clear.clicked.connect(self.clear_input_image)
        
        # Mask Source
        self.chk_use_mask = QCheckBox("Mask")
        self.chk_use_mask.setToolTip("Enable Mask")
        self.chk_use_image.toggled.connect(lambda c: self.chk_use_mask.setEnabled(c))
        
        btn_mask = QPushButton("Open")
        btn_mask.clicked.connect(self.open_external_mask)
        
        btn_mask_clear = QPushButton("Clear")
        btn_mask_clear.setToolTip("입력 마스크 초기화")
        btn_mask_clear.setStyleSheet("color: #e74c3c; font-weight: bold;")
        btn_mask_clear.clicked.connect(self.clear_external_mask)

        grp_src.addWidget(self.chk_use_image)
        grp_src.addWidget(btn_image)
        grp_src.addWidget(self.btn_image_clear)
        grp_src.addSpacing(5)
        
        v_sep = QFrame(); v_sep.setFrameShape(QFrame.VLine); v_sep.setFrameShadow(QFrame.Sunken); v_sep.setStyleSheet("color:#444")
        grp_src.addWidget(v_sep)
        grp_src.addSpacing(5)
        
        grp_src.addWidget(self.chk_use_mask)
        grp_src.addWidget(btn_mask)
        grp_src.addWidget(btn_mask_clear)
        grp_src.addStretch()
        lb.addLayout(grp_src)
        
        # Multi-Image
        self.multi_image_group = QGroupBox("Multi-Image"); self.multi_image_group.setVisible(False)
        m_lay = QHBoxLayout(self.multi_image_group); m_lay.setContentsMargins(4,4,4,4)
        
        # --- (수정) 멀티이미지 리스트를 갤러리 썸네일 스타일로 변경 ---
        self.list_multi_imgs = QListWidget()
        self.list_multi_imgs.setFixedHeight(75) # 목록 높이 넉넉하게 확장
        self.list_multi_imgs.setViewMode(QListWidget.IconMode) # 아이콘(썸네일) 뷰 모드
        self.list_multi_imgs.setIconSize(QSize(50, 50)) # 썸네일 크기 설정
        self.list_multi_imgs.setResizeMode(QListWidget.Adjust)
        self.list_multi_imgs.setSpacing(5)
        self.list_multi_imgs.setStyleSheet("background: #252525; border: 1px solid #444;")
        # --------------------------------------------------------------
        
        # 버튼을 위아래로 배치하기 위해 세로 레이아웃(VBox) 사용
        btn_vlay = QVBoxLayout()
        self.btn_add_mi = QPushButton("Add")
        self.btn_del_mi = QPushButton("Del")
        self.btn_add_mi.clicked.connect(self.add_multi_images)
        self.btn_del_mi.clicked.connect(self.del_multi_images)
        btn_vlay.addWidget(self.btn_add_mi)
        btn_vlay.addWidget(self.btn_del_mi)
        
        m_lay.addWidget(self.list_multi_imgs)
        m_lay.addLayout(btn_vlay)
        lb.addWidget(self.multi_image_group)
        
        lb.addStretch()

        # --- Tab 2: 상세 조정 (Detail) ---
        tab_params = QWidget()
        lp = QGridLayout(tab_params)
        lp.setContentsMargins(8, 12, 8, 8)
        lp.setSpacing(8)

        # Row 0: Steps & CFG
        lp.addWidget(QLabel("Steps:"), 0, 0)
        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(1, 100); self.spin_steps.setValue(30)
        lp.addWidget(self.spin_steps, 0, 1)
        
        lp.addWidget(QLabel("CFG:"), 0, 2)
        self.spin_cfg = QDoubleSpinBox()
        self.spin_cfg.setRange(0.0, 50.0); self.spin_cfg.setValue(7.5); self.spin_cfg.setSingleStep(0.5)
        lp.addWidget(self.spin_cfg, 0, 3)

        # Row 1: Ratio & Img Guide
        lp.addWidget(QLabel("Ratio:"), 1, 0)
        self.combo_res_mode = QComboBox()
        for t, d in [("Match Input","match_input"),("1:1 Square","1:1"),("16:9 Wide","16:9"),("9:16 Portrait","9:16"),("4:3 Standard","4:3")]:
            self.combo_res_mode.addItem(t, d)
        lp.addWidget(self.combo_res_mode, 1, 1)

        self.lbl_img_guidance = QLabel("ImgCFG:") 
        self.spin_img_guidance = QDoubleSpinBox()
        self.spin_img_guidance.setRange(1.0, 10.0); self.spin_img_guidance.setSingleStep(0.1)
        self.lbl_img_guidance.setVisible(False); self.spin_img_guidance.setVisible(False)
        lp.addWidget(self.lbl_img_guidance, 1, 2)
        lp.addWidget(self.spin_img_guidance, 1, 3)

        # Row 2: Preset
        lp.addWidget(QLabel("Preset:"), 2, 0)
        self.combo_scenario = QComboBox()
        self.combo_scenario.addItem("- Select -")
        self.combo_scenario.addItems(list(self.scenarios.keys()))
        self.combo_scenario.currentIndexChanged.connect(self.on_scenario_changed)
        lp.addWidget(self.combo_scenario, 2, 1, 1, 2)
        
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
      
        self.combo_precision = QComboBox(); self.combo_quant = QComboBox()
        self.upscale_settings = {"scale": 4.0, "tile": 512, "resize_back": True}

        setting_tabs.addTab(tab_basic, "Basic")
        setting_tabs.addTab(tab_params, "Detail")

        # [RIGHT] 프롬프트 & 생성
        right_panel = QWidget()
        rp = QVBoxLayout(right_panel)
        rp.setContentsMargins(0, 0, 0, 0)
        rp.setSpacing(6)

        row_head = QHBoxLayout()
        lbl_title = QLabel("PROMPT INPUT")
        lbl_title.setStyleSheet("color: #bbb; font-weight: bold; font-size: 11px;")
        
        # 수동 프롬프트 활성화 위젯 생성
        self.chk_manual_prompt = QCheckBox("enable manual prompt")
        self.chk_manual_prompt.setStyleSheet("color: #e67e22; font-weight: bold;")
        self.chk_manual_prompt.setToolTip("체크 시 번역 및 태그 변환 등을 모두 생략하고 프롬프트 원본을 그대로 전달합니다.")
        self.chk_manual_prompt.toggled.connect(self.on_manual_prompt_toggled)
        self.chk_manual_prompt.setChecked(False) # 기본값 선택 해제
        
        self.lbl_p_count = QLabel("P: 0/75"); self.lbl_n_count = QLabel("N: 0/75")
        self.lbl_p_count.setStyleSheet("color:#2ecc71; font-family: Consolas; font-size:10px;")
        self.lbl_n_count.setStyleSheet("color:#e74c3c; font-family: Consolas; font-size:10px;")
        
        row_head.addWidget(lbl_title)
        row_head.addSpacing(15)
        row_head.addWidget(self.chk_manual_prompt) # 헤더 영역에 배치 완료
        row_head.addStretch()
        row_head.addWidget(self.lbl_p_count)
        row_head.addSpacing(8)
        row_head.addWidget(self.lbl_n_count)
        rp.addLayout(row_head)
        
        # 2. 텍스트 에디터
        self.txt_prompt = QTextEdit()
        self.txt_prompt.setPlaceholderText("Positive Prompt (한글 가능)")
        self.txt_prompt.setFixedHeight(45)
        self.txt_prompt.setStyleSheet("background: #252525; border: 1px solid #555; border-radius: 3px;")
        
        self.txt_negative = QTextEdit()
        self.txt_negative.setPlaceholderText("Negative Prompt (부정 프롬프트)")
        self.txt_negative.setFixedHeight(30)
        self.txt_negative.setStyleSheet("background: #252525; border: 1px solid #555; border-radius: 3px;")

        self.txt_prompt.textChanged.connect(self.update_word_counts)
        self.txt_negative.textChanged.connect(self.update_word_counts)

        rp.addWidget(self.txt_prompt)
        rp.addWidget(self.txt_negative)

        # 3. 하단 액션 바
        action_box = QWidget()
        # (수정) 전체 뼈대를 세로(QVBoxLayout)로 변경하여 상단(체크박스) / 하단(텍스트창+생성버튼)으로 분리
        action_layout = QVBoxLayout(action_box)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(4)
        
        # 상단: 번역/RAG 컨트롤 툴바
        trans_tool = QHBoxLayout()
        self.chk_translate = QCheckBox("use Trans(Ko→En)")
        self.chk_translate.setChecked(True)
        
        self.chk_use_rag = QCheckBox("민화 최적화 프롬프트(RAG)")
        self.chk_use_rag.setStyleSheet("color: #f39c12; font-weight: bold;")
        self.chk_use_rag.setToolTip("한글 입력 후 'view Result'를 누르면 AI가 민화풍 프롬프트로 자동 최적화합니다.")

        self.btn_view_trans = QPushButton("view Result")
        self.btn_view_trans.setToolTip("입력된 프롬프트를 지금 바로 번역/최적화해서 결과 보기")
        self.btn_view_trans.clicked.connect(self.on_translate_requested)
        
        trans_tool.addWidget(self.chk_translate)
        trans_tool.addSpacing(10)
        trans_tool.addWidget(self.chk_use_rag) 
        trans_tool.addSpacing(15)
        trans_tool.addWidget(self.btn_view_trans)
        trans_tool.addStretch()
        
        # 하단: 텍스트 창 + GENERATE 버튼 가로 정렬
        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(0, 0, 0, 0)
        bottom_row.setSpacing(8)

        self.txt_trans_result = QTextEdit()
        self.txt_trans_result.setReadOnly(True)
        self.txt_trans_result.setFixedHeight(55) # 높이 살짝 조정
        self.txt_trans_result.setPlaceholderText("변환 결과가 여기에 표시됩니다.")
        self.txt_trans_result.setStyleSheet("background: #1e1e1e; color: #888; border: 1px solid #3d3d3d; font-size: 10px; font-family: Consolas;")
        
        self.chk_translate.toggled.connect(self.txt_trans_result.setEnabled)
        self.chk_translate.toggled.connect(self.update_word_counts)
        
        self.btn_gen = QPushButton("GENERATE")
        self.btn_gen.setFixedSize(100, 55) # 텍스트 창과 높이(55) 맞춤
        self.btn_gen.setCursor(Qt.PointingHandCursor)
        self.btn_gen.setStyleSheet("""
            QPushButton { 
                background-color: #d35400; color: white; 
                font-weight: bold; font-size: 14px;
                border-radius: 4px; border: 1px solid #e67e22;
            }
            QPushButton:hover { background-color: #e67e22; border: 1px solid #f39c12; }
            QPushButton:pressed { background-color: #a84300; margin-top: 1px; }
            QPushButton:disabled { background-color: #555; color: #888; border: 1px solid #444; }
        """)
        self.btn_gen.clicked.connect(self.run_generation)
        self._setup_btn_feedback(self.btn_gen)

        bottom_row.addWidget(self.txt_trans_result, 1) # 텍스트 창은 길게 늘어남
        bottom_row.addWidget(self.btn_gen, 0)          # 버튼은 고정 크기 유지
        
        action_layout.addLayout(trans_tool)
        action_layout.addLayout(bottom_row)
        
        rp.addWidget(action_box)

        # 메인 레이아웃
        main_layout.addWidget(setting_tabs, 45)
        main_layout.addWidget(right_panel, 55)

        return panel

    def add_multi_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png *.jpg *.jpeg)")
        if files:
            for f in files:
                item = QListWidgetItem(QIcon(f), os.path.basename(f))
                item.setData(Qt.UserRole, f)
                self.list_multi_imgs.addItem(item)
                self.multi_images.append(f)

    def del_multi_images(self):
        """ 다중 이미지 리스트에서 선택된 항목들을 안전하게 삭제함
            배열 인덱스가 당겨져서 엉뚱한 값이 삭제되는 현상을 방지하기 위해 뒷쪽(역순) 인덱스부터 삭제를 진행함
            - 추가: 항목을 선택하지 않고 삭제 시도 시 사용자에게 경고 팝업 제공 """
        selected_items = self.list_multi_imgs.selectedItems()
        
        # 선택된 아이템이 없을 경우 경고 메시지 팝업 출력 후 종료
        if not selected_items:
            QMessageBox.warning(self, "선택 오류", "삭제할 이미지를 먼저 리스트에서 선택해 주세요.")
            return
            
        # 1. UI에서 선택된 아이템의 행(Row) 인덱스를 추출하고 내림차순(역순)으로 정렬
        rows_to_delete = sorted([self.list_multi_imgs.row(item) for item in selected_items], reverse=True)
        
        # 2. 역순으로 UI 목록 위젯 및 백엔드 데이터(self.multi_images)에서 동시 제거
        for row in rows_to_delete:
            self.list_multi_imgs.takeItem(row)
            if 0 <= row < len(self.multi_images):
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

        row_view_btns = QHBoxLayout()
        row_view_btns.setContentsMargins(0, 4, 0, 0)
        row_view_btns.setSpacing(6)
        
        btn_fit_t = QPushButton("Fit")
        btn_fit_t.setProperty("simple_btn", "true")
        btn_fit_t.setToolTip("창 크기에 맞춤")
        btn_fit_t.clicked.connect(self.input_canvas.fit_to_window)
        
        btn_act_t = QPushButton("1:1")
        btn_act_t.setProperty("simple_btn", "true")
        btn_act_t.setToolTip("실제 크기로 보기")
        btn_act_t.clicked.connect(self.input_canvas.set_actual_size)
        
        row_view_btns.addWidget(btn_fit_t)
        row_view_btns.addWidget(btn_act_t)
        layout.addLayout(row_view_btns)

        self.toolbar.show()
        self.toolbar.move(20, 20)
        
        # 기본값 설정
        self.rb_box.setChecked(True)

    def on_input_selection_changed(self, tool_type, data):
        """ 입력 캔버스의 선택 영역 변경 시 마스크 생성 및 프리뷰 갱신
            모델 추론 시 전달되는 실제 마스크 데이터와 동일하게, 완전한 검정 배경에 선택 영역만 흰색으로 칠해 직관적으로 표시함 """
        if self.image is None:
            return
            
        # 1. 캔버스 데이터로부터 1채널 마스크 (0 or 255) 생성
        mask = self.generate_mask_from_canvas()
        
        # 2. 프리뷰 시각화를 위해 항상 순수 검정색의 3채널(RGB) 도화지 생성
        # (원본 이미지의 투명도나 채널 수에 영향을 받지 않고 가장 확실한 마스크 형태를 보장함)
        h, w = self.image.shape[:2]
        preview = np.zeros((h, w, 3), dtype=np.uint8)
            
        # 3. 마스크 값이 255(흰색)인 영역만 프리뷰에 순백색 오버레이
        if mask is not None:
            preview[mask == 255] = [255, 255, 255]
        
        # 4. 마스크 프리뷰 캔버스 갱신
        self.mask_canvas.set_image(preview)
        self.mask_canvas.fit_to_window()
        self.status.showMessage(f"Mask Updated ({tool_type})", 1000)
        
    def clear_image(self):
        """ 단일 이미지 및 마스크 기본 초기화
            - 원본 이미지 및 캔버스 화면 제거
            - 다각도 카메라 제어(Camera 탭)의 썸네일 이미지 초기화 추가
        """
        self.image = None
        self.external_mask = None
        
        if hasattr(self, 'input_canvas') and self.input_canvas:
            self.input_canvas.set_image(None)
            self.input_canvas.reset_selection()
            
        if hasattr(self, 'mask_canvas') and self.mask_canvas:
            self.mask_canvas.set_image(None)
            
        # 카메라 썸네일 초기화
        if hasattr(self, 'camera_controller') and self.camera_controller:
            self.camera_controller.set_thumbnail(None)
            
        if hasattr(self, 'chk_use_image'):
            self.chk_use_image.setChecked(False)
            
        if hasattr(self, 'chk_use_mask'):
            self.chk_use_mask.setChecked(False)
            
        gc.collect()
    
    def clear_input_image(self):
        """ 입력 이미지 및 종속된 시각적 데이터 완전 초기화
            - 원본 이미지 데이터 및 캔버스 오버레이 제거
            - 다각도 카메라 제어(Camera 탭)의 썸네일 이미지 동기화 초기화
            - 마스크 데이터 및 UI 체크박스 상태 동기화 처리
        """
        # 1. 논리 데이터 초기화
        self.image = None
       
        # 2. 캔버스 시각적 요소 제거
        if self.input_canvas:
            self.input_canvas.set_image(None)
            self.input_canvas.clear_all_overlays() 
            
        # 3. 카메라 컨트롤러 썸네일 초기화
        if hasattr(self, 'camera_controller') and self.camera_controller:
            self.camera_controller.set_thumbnail(None)
            
        # 4. 이미지 종속 데이터 정리
        self.clear_all_masks()
        
        # 5. UI 상태 동기화
        if hasattr(self, "chk_use_image"):
            self.chk_use_image.setChecked(False)
            
        # 6. 메모리 정리 및 로그 기록
        gc.collect()
        self.log("Input image and related data have been cleared.")
    
    def clear_all_images(self):
        """ 모든 입력 및 결과 이미지 데이터 통합 초기화
            - 원본, 멀티 이미지 리스트, 결과 이미지를 모두 제거
            - 카메라 컨트롤러 썸네일 이미지 초기화
            - UI 시각적 상태와 완전히 동기화함
        """
        # 1. 논리 데이터 초기화
        self.image = None
        self.multi_images = []
        self.result_image = None
        
        # 2. 캔버스 시각적 요소 제거
        if self.input_canvas:
            self.input_canvas.set_image(None)
            self.input_canvas.clear_all_overlays() 
            
        if self.result_canvas:
            self.result_canvas.set_image(None)
            
        # 3. 카메라 컨트롤러 썸네일 초기화
        if hasattr(self, 'camera_controller') and self.camera_controller:
            self.camera_controller.set_thumbnail(None)
            
        # 4. Multi-Image UI 리스트 완전 삭제
        if hasattr(self, 'list_multi_imgs'):
            self.list_multi_imgs.clear()
            
        # 5. 마스크 종속 데이터 정리
        self.clear_all_masks()
        
        # 6. UI 상태 동기화
        if hasattr(self, "chk_use_image"):
            self.chk_use_image.setChecked(False)
            
        # 7. 메모리 정리 및 로그 기록
        gc.collect()
        self.log("All input and result images have been cleared.")
        
    def clear_all_inputs(self):
        """ [통합] 입력 이미지, 마스크 및 결과 화면 전체 초기화
            - 내부 데이터 변수 및 캔버스 이미지 일괄 제거
            - 카메라 컨트롤러 썸네일 포함 UI 상태 초기화
            - 체크박스 상태 초기화 및 관련 로그 기록
        """
        # 1. 내부 데이터 초기화
        self.image = None
        self.multi_images = []
        self.result_image = None
        self.external_mask = None
        
        # 2. 카메라 컨트롤러 썸네일 초기화
        if hasattr(self, 'camera_controller') and self.camera_controller:
            self.camera_controller.set_thumbnail(None)
            
        # 3. Multi-Image UI 리스트 완전 삭제
        if hasattr(self, 'list_multi_imgs'):
            self.list_multi_imgs.clear()
        
        # 4. 각 캔버스 이미지 클리어
        if self.input_canvas: 
            self.input_canvas.set_image(None)
            self.input_canvas.clear_selection()
            
        if self.mask_canvas: 
            self.mask_canvas.set_image(None)
            
        if self.result_canvas: 
            self.result_canvas.set_image(None)
            
        # 5. UI 체크박스 상태 초기화
        if hasattr(self, "chk_use_image"):
            self.chk_use_image.setChecked(False)
            
        if hasattr(self, "chk_use_mask"):
            self.chk_use_mask.setChecked(False)
            self.chk_use_mask.setEnabled(False)
        
        # 6. 메모리 정리 및 로그 기록
        gc.collect()
        self.log("All input data and canvases have been cleared.")
    
    def clear_input_mask(self):
        """ 사용자가 캔버스에 그린 마스크(Box, Lasso, Brush) 선택 영역을 초기화하고 프리뷰를 갱신함
            단순 검정 화면을 덮는 대신 갱신 로직을 호출하여 외부 마스크 데이터가 있다면 보존되도록 함 """
        self.input_canvas.reset_selection()
        if self.image is not None:
            self.on_input_selection_changed("Clear Canvas", None)
            
    def clear_all_masks(self):
        """ 모든 마스크(내부 드로잉 + 외부 파일) 및 프리뷰 초기화
            UI 이벤트 루프의 충돌을 방지하며 모든 캔버스 데이터를 완전 삭제 상태로 되돌림 """
        self.current_mask = None
        self._last_generated_mask = None
        self.external_mask = None
        
        self.input_canvas.blockSignals(True)
        try:
            self.input_canvas.clear_all_overlays() 
            self.input_canvas.set_overlay_mask(None) 
            self.input_canvas.reset_selection() # [추가] 선택 영역 변수 완벽 초기화
            self.input_canvas.repaint() 
        finally:
            self.input_canvas.blockSignals(False)

        if self.image is not None:
            # 갱신 로직을 타게 하여 깨끗한 검은 배경이 렌더링 되도록 유도
            self.on_input_selection_changed("Clear All", None)
        else:
            self.mask_canvas.set_image(None)
            
        self.mask_canvas.clear_all_overlays()
        self.mask_canvas.repaint()

        if hasattr(self, "chk_use_mask"):
            self.chk_use_mask.setChecked(False)

        self.log("All mask and overlay visuals cleared.")

    def clear_external_mask(self):
        """ 외부에서 불러온 마스크 데이터만 제거하고 프리뷰 화면을 초기화함 """
        self.external_mask = None
        
        # UI 체크 해제 및 프리뷰 갱신 로직 실행
        if hasattr(self, "chk_use_mask"):
            self.chk_use_mask.setChecked(False)
            
        # "Clear External" 모드로 갱신하여 현재 남은(Box, Brush 등) 마스크만 다시 그리도록 함
        self.on_input_selection_changed("Clear External", None)
        self.log("External mask has been removed.")
    
    def open_external_mask(self):
        """ 외부 마스크 이미지를 로드하고 투명도(Alpha)를 명확히 처리하여 마스크로 변환
            투명한 배경이 흰색 마스크로 잘못 변질되어 화면을 덮는 치명적 렌더링 오류를 차단함 """
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please load the Input Image first.")
            return

        with SuppressStderr():
            fname, _ = QFileDialog.getOpenFileName(self, "Open Mask", "", "Images (*.png *.jpg *.bmp)")

        if fname:
            try:
                m_pil = Image.open(fname)
                
                # 투명 픽셀(Alpha=0)이 흑백 변환 시 흰색으로 왜곡되는 현상 방지
                if m_pil.mode in ('RGBA', 'LA') or (m_pil.mode == 'P' and 'transparency' in m_pil.info):
                    m_pil = m_pil.convert("RGBA")
                    # 투명 영역은 온전한 검은색(0)으로, 마스크 영역은 원래 밝기로 보존하기 위한 배경 합성
                    bg = Image.new("RGB", m_pil.size, (0, 0, 0))
                    bg.paste(m_pil, mask=m_pil.split()[3])
                    m_pil = bg.convert("L")
                else:
                    m_pil = m_pil.convert("L")

                # 원본 이미지 크기에 맞춰 크기 조정
                h, w = self.image.shape[:2]
                if m_pil.size != (w, h):
                    m_pil = m_pil.resize((w, h), Image.NEAREST)
                
                mask_np = np.array(m_pil)
                # 이진화 처리 (완전한 흑백 분리)
                _, self.external_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

                self.chk_use_mask.setChecked(True)
                self.on_input_selection_changed("External File", None)

            except Exception as e:
                print(f"[Error] Failed to load mask: {e}")
                QMessageBox.critical(self, "Error", f"Failed to load mask file.\n\nDetails: {e}")
       
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
                # --- (수정) mtype(T2I/I2I) 검사 로직 삭제, 모든 모델 표시 ---
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
        
        # 🚨 [안전장치] 만약 카메라 텍스트에 <camera> 껍데기가 없다면 강제로 씌워줍니다.
        if not camera_text.strip().startswith("<camera>"):
            camera_text = f"<camera>{camera_text}</camera>"
            
        pattern = r"\s*<camera>.*?</camera>\s*"
        injection_text = f" {camera_text} " 
        if "<camera>" in current_text:
            new_text = re.sub(pattern, injection_text, current_text, flags=re.DOTALL)
        else:
            new_text = f"{current_text.strip()} {injection_text.strip()}"
        self.txt_prompt.setPlainText(new_text)

    def on_translate_requested(self):
        p = self.txt_prompt.toPlainText()
        n = self.txt_negative.toPlainText()
        if p or n: self._run_preview_translation(p, n)

    def _run_preview_translation(self, p, n):
        self._blink_state = True
        self.blink_timer.start()
        self.txt_trans_result.setPlainText("Processing...")
        self._stop_worker('preview_worker')
        
        # --- RAG 모드가 켜져있을 때 ---
        if self.chk_use_rag.isChecked():
            api_key = token_key.get_valid_api_key()
            if not api_key:
                self.txt_trans_result.setPlainText("Error: RAG 기능을 사용하려면 Google API Key를 먼저 설정해 주세요.")
                self.blink_timer.stop()
                return
            # RAG 인덱스 준비 여부 사전 확인
            if not self.rag_prompter.is_ready:
                self.txt_trans_result.setPlainText("RAG 인덱스 로딩 중... 잠시 후 다시 시도해주세요.")
                self.blink_timer.stop()
                return

            # I2I 모드 판별: 입력 이미지가 있고 instruction-following 모델일 때만 I2I RAG 사용
            # DreamShaper(SDInpaint) 는 prompt_style = instruction 이 아니므로 자동으로 T2I RAG 경로 유지
            _model_cfg = self._active_model_config or {}
            _is_instruction_model = _model_cfg.get("prompt_style", "") == "instruction"
            _has_input_image = self.chk_use_image.isChecked() and self.image is not None
            _use_i2i_rag = _has_input_image and _is_instruction_model

            if _use_i2i_rag:
                self.txt_trans_result.setPlainText("Processing... (I2I 이미지 분석 중)")
                self.preview_worker = GenericWorker(
                    self.rag_prompter.generate_i2i_instruction,
                    user_input=p,
                    image=self.image.copy(),   # 복사본 전달 (스레드 안전)
                    api_key=api_key,
                    abort_check=lambda: False
                )
            else:
                self.txt_trans_result.setPlainText("Processing... (RAG 스타일 최적화 중)")
                self.preview_worker = GenericWorker(
                    self.rag_prompter.generate_enhanced_prompt,
                    user_input=p,
                    api_key=api_key,
                    abort_check=lambda: False
                )
        else:
            def preview_process(**kwargs):
                # 실제 GENERATE 버튼을 눌렀을 때와 똑같은 과정을 거쳐 미리보기를 생성합니다.
                p_txt, n_txt = self.prompt_engine.process(
                    p_raw=p,
                    n_raw=n,
                    manual_mode=self.chk_manual_prompt.isChecked(),
                    multi_angle=self.chk_multi_angle.isChecked(),
                    model_cfg=self._active_model_config,
                    use_translator=True
                )
                return p_txt, n_txt
                
            self.preview_worker = GenericWorker(preview_process, abort_check=lambda: False)
            
        self.preview_worker.signal_finished.connect(self._on_preview_translation_done)
        self.preview_worker.finished.connect(lambda: self._stop_worker('preview_worker'))
        self.preview_worker.start()

    def _on_preview_translation_done(self, res):
        """ Preview (view Result) 처리 콜백 """
        self.blink_timer.stop()
        if not res:
            self.txt_trans_result.setPlainText("Failed.")
            return
            
        if isinstance(res, dict):
            # RAG 데이터 캐싱 (Generate 시 활용)
            self._cached_rag_data = res

            # 원문 단순 번역본 확보 (UI 표시용)
            p_en = self.prompt_engine.translate_only(self.txt_prompt.toPlainText())

            if res.get("mode") == "i2i":
                # I2I RAG 결과: 이미지 분석 기반 경량 instruction 표시
                html  = f"<b style='color:#27ae60'>[🖼️ I2I 이미지 분석 완료]</b><br>"
                html += f"<b>원문 번역:</b> {p_en}<br><hr>"
                html += f"<b>[이미지 분석 결과]</b><br>"
                html += f"<b>CHANGE:</b> {res.get('change', '')}<br>"
                html += f"<b>KEEP:</b> {res.get('keep', '')}<br>"
                html += f"<small style='color:#95a5a6'>※ I2I 모드: 스타일 앵커 없이 편집 지시문만 생성됩니다.</small>"
            else:
                # T2I RAG 결과: 기존 전체 레시피 표시
                html  = f"<b style='color:#f39c12'>[🪄 RAG 템플릿 적용 준비 완료]</b><br>"
                html += f"<b>원문 번역:</b> {p_en}<br><hr>"
                html += f"<b>[RAG 분석 결과]</b><br>"
                html += f"<b>KEEP:</b> {res.get('keep', '')}<br>"
                html += f"<b>CHANGE:</b> {res.get('change', '')}<br>"
                html += f"<b>ADD:</b> {res.get('add', '')}<br>"
                html += f"<b>STYLE:</b> {res.get('style_anchors', '')}<br>"
                html += f"<b>NEG:</b> {res.get('negative', '')}<br>"

            self.txt_trans_result.setHtml(html)
        else:
            self._cached_rag_data = None
            p_en, n_en = res
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
        """ 캔버스에 그려진 데이터 및 외부 마스크를 취합하여 최종 흑백 마스크 생성
            원본 이미지의 형태나 투명도(Alpha)가 마스크로 변질되는 현상을 막기 위해 순수 0(Black) 배열에서 시작함 """
        if self.image is None:
            return None
        
        h, w = self.image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
            
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
                        pt1 = (int(points[i][0]), int(points[i][1]))
                        pt2 = (int(points[i+1][0]), int(points[i+1][1]))
                        cv2.line(mask, pt1, pt2, 255, thickness=int(size))
                        cv2.circle(mask, pt1, int(size/2), 255, -1)
                    cv2.circle(mask, (int(points[-1][0]), int(points[-1][1])), int(size/2), 255, -1)
                elif len(points) == 1:
                    cv2.circle(mask, (int(points[0][0]), int(points[0][1])), int(size/2), 255, -1)
                    
        if self.external_mask is not None:
            mask = cv2.bitwise_or(mask, self.external_mask)
            
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask

    def cancel_generation(self):
        self._cancel_event.set()

    def _should_cancel(self):
        return self._cancel_event.is_set()

    def _parse_options(self, model_config: dict):
        """ 모델별 options 필드 파싱
            - options 문자열에서 단일 플래그(set)와 key=value(dict) 형태를 동시에 파싱하여 분리 반환합니다.
        """
        raw = model_config.get("options", [])
        items = raw if isinstance(raw, list) else [x.strip() for x in raw.split(",")]
        flags, kv = set(), {}
        
        for item in items:
            item = item.strip()
            if not item: continue
            
            if "=" in item:
                k, _, v = item.partition("=")
                kv[k.strip().lower()] = v.strip()
            else:
                flags.add(item.lower())
                
        return flags, kv

    def _task_smart_generation(self, model_cfg, mode_ui, is_remote_forced, **kwargs):
        """ 스마트 백그라운드 워커 추론 작업 수행 
            - 마스크 체크 해제 혹은 빈 마스크일 경우 명시적으로 None 처리하여 API 및 후처리로 넘김
        """
        execution_mode = "remote" if is_remote_forced or model_cfg.get("mode") == "remote" else "local"
        
        if execution_mode == "local":
            try:
                return self.diffusion_estimator.predict(**kwargs)
            except Exception as e:
                # 무조건 Remote로 전환하지 않고, 모델이 Remote를 지원할 때만 전환하도록 변경
                if model_cfg.get("mode") in ["remote", "both"]:
                    print(f"Local failed: {e}. Switching to Remote.")
                    execution_mode = "remote"
                else:
                    # 로컬 전용 모델이면 예외를 그대로 발생시켜 워커가 정상적으로 에러 처리하게 함
                    raise RuntimeError(f"Local 추론 실패 (필수 입력값을 확인하세요): {e}")
        
        if execution_mode == "remote":
            gen_pil = self.call_remote_api(model_cfg=model_cfg, **kwargs)
            
            # UI에서 마스크 사용을 해제했거나, 전달된 마스크가 완전히 비어있을 경우 None으로 덮어씀
            post_mask = kwargs.get("mask")
            if not kwargs.get("use_mask") or post_mask is None:
                post_mask = None
            elif isinstance(post_mask, np.ndarray) and post_mask.max() == 0:
                post_mask = None
            
            return self.diffusion_estimator.manual_post_process(
                gen_pil, kwargs.get("image"), post_mask, upscale_opts=kwargs.get("upscale_opts")
            )
            
        return None

    def on_model_combo_changed(self, index=None):
        """ 콤보박스 모델 변경 핸들러
            - 선택된 모델의 옵션을 파싱하여 Multi-Angle 및 Multi-Image UI의 활성화 상태를 제어합니다.
        """
        model_key = self.combo_model.currentData()
        if not model_key: return
        model_config = self.generation_models_dict.get(model_key, {})
        self._active_model_config = model_config
        
        flags, opt_kv = self._parse_options(model_config)
        
        # Multi-Angle 체크
        has_multi_angle = "multi-angle" in flags or "multi_angle" in flags
        self.chk_multi_angle.setEnabled(has_multi_angle)
        self.chk_multi_angle.setChecked(has_multi_angle)
        
        # Multi-Image 체크 (Qwen)
        has_multi_image = "multi-image" in flags
        self.multi_image_group.setVisible(has_multi_image)
        
        # 이미지 필수 여부
        should_check_image = True # Qwen, DreamShaper 모두 이미지 권장
        self.chk_use_image.setChecked(should_check_image)
        self.chk_use_mask.setEnabled(should_check_image)
        
        self.update_word_counts()
        self.lbl_model_status.setText("Load Needed")

    def on_manual_prompt_toggled(self, checked):
        """ 수동 프롬프트 모드 토글 핸들러
            - 활성화 시 번역 관련 제어 UI들을 모두 비활성화 처리
            - 수정사항: 존재하지 않는 버튼 객체 호출 에러 수정
        """
        self.chk_translate.setEnabled(not checked)
        self.btn_view_trans.setEnabled(not checked)

    def on_multi_angle_toggled(self, checked):
        """ 다각도 카메라 제어 체크박스 상태 변경 핸들러 """
        # --- (수정) controller 대신 scroll_area 단위로 탭에 추가/제거 ---
        idx = self.tabs.indexOf(self.camera_scroll_area)
        
        if checked and idx == -1: 
            self.tabs.insertTab(0, self.camera_scroll_area, "Camera")
            self.tabs.setCurrentIndex(0) 
        elif not checked and idx != -1: 
            self.tabs.removeTab(idx)

    def run_generation(self):
        """ 이미지 생성 추론 실행 요청 처리
            - 사용자의 프롬프트 설정 및 UI 옵션에 맞춰 최종 데이터를 가공.
            - 백그라운드 Worker를 통해 로컬 또는 원격 API 추론을 실행.
            - 추론에 전달되는 최종 프롬프트 텍스트를 Log 창에 기록.
            - API Provider 및 라우터 주소에 따라 필요한 인증 키(Token)를 검증.
            - API가 이미지를 필수로 요구할 경우, UI 상 이미지 누락 여부 사전 검증.
        """
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
            elif provider == "fal_ai":
                api_uri = self._active_model_config.get("api_model_uri", "")
                # HF 라우터를 통하는 경우 HF 토큰 검사
                if "router.huggingface.co" in api_uri and not token_key.get_valid_hf_token():
                    QMessageBox.warning(self, "Token 누락", "Hugging Face 라우터를 통한 Fal-AI 모델 접근 시 HF Token이 필요합니다.\n[API Key] 버튼을 눌러 설정해주세요.")
                    self.open_token_settings()
                    return
                # 순수 fal.ai API를 직접 사용하는 경우 (향후 확장성 고려)
                elif "router.huggingface.co" not in api_uri and not token_key.get_valid_api_key():
                    QMessageBox.warning(self, "API Key 누락", "Fal AI API Key가 설정되지 않았습니다.\n[API Key] 버튼을 눌러 키를 설정해주세요.")
                    self.open_token_settings()
                    return

        p_raw = self.txt_prompt.toPlainText().strip()
        n_raw = self.txt_negative.toPlainText().strip()
        
        # 메뉴얼 모드 여부
        is_manual = self.chk_manual_prompt.isChecked()
        use_rag = self.chk_use_rag.isChecked() and not is_manual
        
        # RAG 누락 방지: 민화 최적화 체크 후 view Result를 누르지 않은 경우
        if use_rag and not hasattr(self, '_cached_rag_data'):
            QMessageBox.information(self, "RAG 최적화 안내", "민화 최적화가 켜져 있습니다.\n먼저 [view Result] 버튼을 눌러 프롬프트를 분석 및 최적화해주세요.")
            return

        rag_data = getattr(self, '_cached_rag_data', None) if use_rag else None
        
        # RAG API 통신 에러 감지 및 차단 로직
        if rag_data and "RAG Error:" in str(rag_data.get("negative", "")):
            QMessageBox.critical(
                self, 
                "RAG API 통신 오류", 
                "AI 프롬프트 최적화 중 오류가 발생했습니다.\n서버 트래픽 문제일 수 있으니 [view Result] 버튼을 다시 눌러 RAG 결과를 먼저 확인해 주세요."
            )
            return
            
        # RAG 사용 여부와 상관없이 번역기 옵션이 켜져있으면 True를 전달.
        need_translation = self.chk_translate.isChecked()

        # 프롬프트 통합 처리
        p_txt, n_txt = self.prompt_engine.process(
            p_raw=p_raw,
            n_raw=n_raw,
            manual_mode=is_manual,
            multi_angle=self.chk_multi_angle.isChecked(),
            model_cfg=self._active_model_config,
            use_translator=need_translation,
            rag_data=rag_data
        )

        final_rgb = self.image[:,:,:3] if (self.chk_use_image.isChecked() and self.image is not None) else None
        final_mask = self.generate_mask_from_canvas() if (final_rgb is not None and self.chk_use_mask.isChecked()) else None
        
        # 대상 이미지 수집 로직: None 및 빈 객체 사전 검증 추가 (Multi-line 적용)
        target_imgs = []
        
        if final_rgb is not None:
            target_imgs.append(final_rgb)
            
        for f in self.multi_images:
            if not f:  # None 이거나 빈 문자열일 경우 안전하게 스킵
                continue
            try:
                img = Image.open(f).convert("RGB")
                target_imgs.append(np.array(img))
            except Exception:
                pass

        # --- [예외 처리 추가] 수집된 이미지가 없는 경우의 방어 로직 ---
        # 원격 API 사용 시 모델 URI에 image-to-image 등 이미지가 필수적인 키워드가 포함되어 있고, target_imgs가 비어있다면 422 에러가 발생하므로 사전에 차단 처리.
        if is_remote and provider == "fal_ai":
            api_uri = self._active_model_config.get("api_model_uri", "").lower()
            needs_image = any(kw in api_uri for kw in ["image", "inpainting", "controlnet", "mask"])
            
            if needs_image and not target_imgs:
                QMessageBox.warning(self, "이미지 누락", "선택하신 API 모델은 입력 이미지가 필수입니다.\n'Use Image' 체크박스를 확인하거나 이미지를 로드해주세요.")
                return

        self._stop_worker('worker')
        self._cancel_event.clear()
        self.toggle_loading(True, "Generating", "Processing...")
        
        # 최종 제출 프롬프트를 Log 탭에 기록
        log_msg = f"[Final Prompt] {p_txt}"
        if n_txt:
            log_msg += f" | [Negative] {n_txt}"
            
        self.log(log_msg, switch_tab=True)
        
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
        self.worker.error.connect(self.on_worker_error)
        self.worker.finished.connect(lambda: self._stop_worker('worker'))
        self.worker.start()

    def on_progress_update(self, percent):
        self.loading_overlay.set_message("GENERATING", f"Processing... ({percent}%)")

    def on_gen_finished(self, result):
        """ 생성 완료 후 결과 이미지를 검증하여 UI에 표시 및 저장
            - 결과가 PIL Image인 경우 numpy array로 변환하여 캔버스 호환성 확보
            - 저장 및 갤러리 업데이트 프로세스 수행
        """
        self.toggle_loading(False)
        
        if result is not None:
            # 1. 데이터 타입 변환 (PIL -> Numpy)
            # 캔버스가 PIL 객체를 직접 지원하지 않을 수 있으므로 RGB 배열로 변환합니다.
            if hasattr(result, "convert"):
                result = np.array(result.convert("RGB"))
            
            self.result_image = result
            
            # 2. UI 시각화 업데이트
            if self.result_canvas:
                self.result_canvas.set_image(result)
                self.result_canvas.fit_to_window()
            
            # 3. 파일 저장 및 후속 작업
            save_image_file(result, f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", str(self.output_dir))
            self.update_gallery()
            self.log("Generation result displayed and saved.")
        else:
            # 예외가 발생하지 않았더라도 결과가 None인 경우 알림
            QMessageBox.warning(self, "Generation Failed", "이미지 생성에 실패하였습니다. 모델 설정이나 프롬프트를 확인하세요.")
            
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

    def on_worker_error(self, err_msg):
        """ 백그라운드 워커에서 에러 발생 시 UI 복구 및 안전 필터 예외 처리 """
        self.blink_timer.stop()
        self.toggle_loading(False)
        self.btn_gen.setText("GENERATE")
        self.set_ui_enabled(True)
        
        # --- (추가) 안전 필터(NSFW) 및 특수 에러의 우아한 처리 ---
        err_lower = str(err_msg).lower()
        
        # 1. Gemini 정책 차단 (API 권한/안전 필터 분리)
        if "gemini_policy_violation" in err_lower:
            reason = err_msg.split("사유:")[-1].strip().replace(")", "").strip() if "사유:" in err_msg else "UNKNOWN"
            if "SAFETY" in reason or "IMAGE_SAFETY" in reason:
                QMessageBox.warning(
                    self,
                    "⚠️ 안전 필터 감지",
                    "입력하신 프롬프트나 이미지가 AI 안전 정책에 의해 차단되었습니다.\n내용을 수정하신 후 다시 시도해주세요."
                )
            else:
                QMessageBox.warning(
                    self,
                    "🚫 Gemini 이미지 생성 권한 없음",
                    f"이미지 생성이 차단되었습니다. (사유: {reason})\n\n"
                    "프롬프트 내용과 무관한 API 권한 문제입니다.\n\n"
                    "다음을 확인해주세요:\n"
                    "  • Google AI Studio에서 유료 플랜(Paid Tier) 전환 여부\n"
                    "  • Imagen API 사용 권한 활성화 여부\n"
                    "  • 유효한 API Key 입력 여부 (상단 [API Key] 버튼)"
                )

        # 1. 안전 필터(NSFW) / 비윤리적 콘텐츠 차단 감지
        if any(kw in err_lower for kw in ["safety", "nsfw", "policy", "blocked", "content is not allowed", "inappropriate"]):
            QMessageBox.warning(
                self, 
                "⚠️ 안전 필터 감지", 
                "입력하신 프롬프트나 이미지가 AI 안전 정책(NSFW/비윤리적 콘텐츠)에 의해 차단되었습니다.\n내용을 수정하신 후 다시 시도해주세요."
            )
        # 2. 서버 일시 과부하(503) 감지
        elif "503" in err_lower or "unavailable" in err_lower:
            QMessageBox.warning(
                self,
                "🔄 서버 일시 과부하",
                "Gemini 서버에 요청이 몰리고 있습니다.\n잠시 후 다시 시도해주세요.\n(보통 30초~1분 이내 해소됩니다)"
            )
        # 3. API 호출 횟수 제한(Rate Limit) 감지
        elif "rate limit" in err_lower or "429" in err_lower or "quota" in err_lower:
            QMessageBox.warning(
                self,
                "⏱️ API 호출 제한",
                "API 호출 횟수 제한에 도달했습니다.\n잠시 후 다시 시도하거나, API Key의 사용량을 확인해주세요."
            )
        # 3. 인증(API Key) 오류 감지
        elif "401" in err_lower or "unauthorized" in err_lower or "invalid api key" in err_lower:
            QMessageBox.warning(
                self, 
                "🔑 인증 오류", 
                "API Key가 유효하지 않습니다.\n상단의 [API Key] 버튼을 눌러 정확한 키를 입력해주세요."
            )
        # 4. 그 외 일반적인 에러
        else:
            QMessageBox.critical(self, "❌ 생성 오류", f"이미지 생성 중 오류가 발생했습니다:\n{err_msg}")
        # --------------------------------------------------------

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

    def get_hf_client(self, space_url, hf_token=None):
        if not space_url: return None
        key = (space_url, bool(hf_token))
        if key not in HF_CLIENT_CACHE:
            HF_CLIENT_CACHE[key] = Client(space_url, token=hf_token)
        return HF_CLIENT_CACHE[key]

    def call_remote_api(self, *, model_cfg, **kwargs):
        """ 원격 API 호출 분기 처리: Provider 및 라우터 주소에 따른 적절한 API 호출 및 토큰 할당 수행 """
        provider = model_cfg.get("provider", "hf_space")
        
        if provider == "google_genai": 
            return self.call_gemini_api(model_cfg=model_cfg, **kwargs)
            
        elif provider == "fal_ai":
            # api_model_uri에 허깅페이스 라우터 주소가 포함되어 있는지 확인하여 토큰 분기 처리
            api_uri = model_cfg.get("api_model_uri", "")
            
            if "router.huggingface.co" in api_uri:
                target_token = kwargs.get("hf_token")
            else:
                target_token = kwargs.get("api_key")
                
            return self.call_fal_ai_api(
                model_id=model_cfg.get("repo_id"), 
                token=target_token,
                model_cfg=model_cfg, **kwargs
            )
            
        elif provider == "hf_space": 
            return self.call_hf_space_api(model_cfg=model_cfg, **kwargs)
            
        raise ValueError(f"Unknown provider: {provider}")

    def call_gemini_api(self, *, model_cfg, pil_images, prompt, **kwargs):
        """ 제미나이 API 호출: 응답 객체 유효성 검사 강화 및 인페인팅 처리 지원
            - 전달된 mask 데이터를 활용하여 inpainting-insert 등의 편집 모드를 적용.
            - 기존 PIL 이미지 배열 뒤에 마스크 이미지를 contents로 추가하여 전송하되, 각 파트의 역할을 텍스트로 명시.
            - 프롬프트에 원본 이미지의 조명, 색온도, 분위기를 마스크 영역에 강제로 동기화하도록 보정 처리 추가.
        """
        api_key = kwargs.get("api_key")
        if not api_key: 
            raise ValueError("Google API Key is missing.")
            
        # config.ini에 정의된 gemini-3-pro-image-preview 모델을 우선 사용
        model_name = model_cfg.get("api_model_uri", "gemini-3-pro-image-preview")
        client = genai.Client(api_key=api_key)
        
        # 마스크 사용 시 이질감을 제거하기 위한 동기화 지시어 주입
        final_prompt = prompt
        mask_np = kwargs.get("mask")
        
        # 프롬프트에 카메라 앵글 변경 지시가 있는지 확인
        has_angle_change = "CAMERA DIRECTION" in prompt and "Stable centered frontal view" not in prompt
        
        if mask_np is not None:
            if has_angle_change:
                # 앵글 변경 시: 원본 보존 강박을 줄이고 3D 변형 허용
                sync_instruction = (
                    "CRITICAL: Adapt the masked area to match the specified CAMERA DIRECTION. "
                    "Do NOT strictly preserve the 2D shape of the original if it conflicts with the new perspective."
                )
            else:
                # 앵글 변경 없을 시: 기존처럼 원본 100% 동기화 (조명, 그림자 등)
                sync_instruction = (
                    "CRITICAL: Match the lighting, shadows, color temperature, and texture of the original image "
                    "perfectly in the masked area. The output must be a single, seamless, and natural composite."
                )
            final_prompt = f"{sync_instruction}\n\nTask: {prompt}"

        # 앵글 변경 시 Temperature를 확 올려서 모델이 창의적으로 형태를 부수도록 허용
        default_temp = kwargs.get("temperature", 0.4)
        gen_temp = 0.8 if has_angle_change else default_temp
        
        res_mode = kwargs.get("resolution_mode", "1:1")
        
        gen_config_params = {
            "response_modalities": ["IMAGE"],
            "temperature": gen_temp
        }

        # 1) 해상도 옵션 설정 
        img_cfg_args = {}
        if res_mode != "match_input":
            gemini_allowed_ratios = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
            img_cfg_args["aspect_ratio"] = res_mode if res_mode in gemini_allowed_ratios else "1:1"

        if img_cfg_args:
            gen_config_params["image_config"] = types.ImageConfig(**img_cfg_args)

        # 2) 콘텐츠 구성 (텍스트 레이블 추가로 모델의 오인식 방지)
        contents = [final_prompt]
        if pil_images:
            contents.append("Reference Original Image:")
            for img in pil_images:
                if max(img.size) > 2048: 
                    img = img.copy()
                    img.thumbnail((2048, 2048), Image.LANCZOS)
                contents.append(img)
                
            # 인페인팅일 경우 마스크 이미지를 레이블과 함께 추가
            if mask_np is not None:
                mask_img = Image.fromarray(mask_np).convert("L")
                if max(mask_img.size) > 2048:
                    mask_img.thumbnail((2048, 2048), Image.LANCZOS)
                contents.append("Edit Mask (White indicates the area to change):")
                contents.append(mask_img)
                
        try:
            # ── 503/429 재시도 래퍼 (최대 3회, 지수 백오프) ──────────────
            import time as _time
            response = None
            for _attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(**gen_config_params)
                    )
                    break  # 성공 시 루프 탈출
                except Exception as _retry_err:
                    _e = str(_retry_err)
                    _is_retryable = "503" in _e or "UNAVAILABLE" in _e or "429" in _e or "RESOURCE_EXHAUSTED" in _e
                    if _is_retryable and _attempt < 2:
                        _wait = 5 * (2 ** _attempt)
                        print(f"[Gemini] 서버 과부하, {_wait}초 후 재시도 ({_attempt + 1}/2)...")
                        _time.sleep(_wait)
                    else:
                        raise  # 재시도 불가 에러이거나 3회 모두 실패 시 상위로 전달
            # ── 재시도 래퍼 끝 ────────────────────────────────────────────

            if not response or not response.candidates:
                raise RuntimeError("Gemini API: 안전 정책에 의해 차단되었습니다.")

            candidate = response.candidates[0]
            finish_reason = getattr(candidate, 'finish_reason', 'UNKNOWN')
            
            # 안전 필터 및 파트 검증
            policy_block_reasons = ["IMAGE_OTHER", "IMAGE_SAFETY", "SAFETY", "RECITATION"]
            if any(reason in str(finish_reason) for reason in policy_block_reasons):
                raise RuntimeError(f"GEMINI_POLICY_VIOLATION: (사유: {finish_reason})")

            if not hasattr(candidate, 'content') or not candidate.content or not candidate.content.parts:
                raise RuntimeError("Gemini API: 생성된 결과가 비어있습니다.")

            for part in candidate.content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    return Image.open(BytesIO(part.inline_data.data)).convert("RGB")
            
            raise RuntimeError("Gemini API: 이미지 데이터를 찾을 수 없습니다.")
        
        except Exception as e:
            raise RuntimeError(str(e))

    def call_fal_ai_api(self, *, model_id, pil_images, prompt, num_inference_steps, guidance_scale, seed=None, token, use_queue=True, **kwargs):
        """ Fal-AI API 호출 
            - 큐(Queue) 방식 추론에 맞춘 상태 폴링(Polling) 로직 수행 및 Qwen 전용 파라미터(true_cfg_scale 등) 처리
        """
        if not token:
            raise RuntimeError("Fal-AI API Key가 누락되었습니다. 설정에서 API 키를 확인해주세요.")

        model_cfg = kwargs.get("model_cfg")
        api_url = model_cfg.get("api_model_uri")

        if not api_url:
            base_router = "https://router.huggingface.co/fal-ai"
            clean_id = model_id.split('/')[-1] if '/' in model_id else model_id
            api_url = f"{base_router}/{clean_id}"

        if use_queue and "?" not in api_url:
            api_url += "?_subdomain=queue"

        # 프롬프트 변환은 run_generation 처리 완료 그대로 사용
        final_prompt = prompt

        def encode_to_b64_dataurl(pil_img):
            buf = BytesIO()
            pil_img.save(buf, format="PNG") 
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{b64}"

        # 수정사항: 422 에러 방지를 위해 기본 payload를 먼저 구성
        payload = {
            "prompt": final_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed if seed is not None else random.randint(0, 2**32 - 1),
        }

        # Qwen-Edit 전용 파라미터(true_cfg_scale 등)를 options에서 가져와 API로 전달
        if model_cfg:
            _, opt_kv = self._parse_options(model_cfg)
            if "true_cfg_scale" in opt_kv:
                payload["true_cfg_scale"] = float(opt_kv["true_cfg_scale"])
                
            if "lora" in (api_url or "").lower():
                lora_path  = opt_kv.get("lora_path", "")
                lora_scale = float(opt_kv.get("lora_scale", 0.9))
                if lora_path:
                    payload["loras"] = [{"path": lora_path, "scale": lora_scale}]

        # 수정사항: pil_images에 실제 이미지가 있을 때만 image_url(s) 필드 추가 (빈 배열 전송 방지)
        if pil_images:
            payload["image_url"] = encode_to_b64_dataurl(pil_images[0])
            payload["image_urls"] = [encode_to_b64_dataurl(img) for img in pil_images]

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=300.0) as client:
                # 1. 작업 제출 단계
                resp = client.post(api_url, headers=headers, json=payload)
                
                if resp.status_code >= 400:
                    try:
                        err_json = resp.json()
                        detail = err_json.get('detail', str(err_json))
                    except:
                        detail = resp.text
                    raise RuntimeError(f"Fal-AI API 요청 실패 ({resp.status_code}):\n{detail}")
                
                result = resp.json()
                img_info = None

                # 2. Queue Polling 단계 (수정된 핵심 로직: status_url 폴링 후 response_url 호출)
                if result.get("status") in ["IN_QUEUE", "IN_PROGRESS"] and "status_url" in result:
                    status_url = result["status_url"]
                    response_url = result["response_url"]
                    
                    for i in range(150): # 최대 약 225초 대기
                        if self._should_cancel():
                            raise RuntimeError("USER_CANCEL")
                            
                        time.sleep(1.5)
                        
                        # 결과 URL이 아닌 상태 URL(status_url)을 찔러 400 에러를 방지함
                        poll = client.get(status_url, headers=headers)
                        
                        if poll.status_code >= 400:
                            raise RuntimeError(f"Fal-AI Polling 상태 확인 오류 ({poll.status_code}): {poll.text}")

                        poll_json = poll.json()
                        current_status = poll_json.get("status")

                        if current_status == "COMPLETED":
                            # 작업이 완료되었을 때 비로소 최종 결과 데이터(response_url)를 요청함
                            final_resp = client.get(response_url, headers=headers)
                            if final_resp.status_code >= 400:
                                raise RuntimeError(f"Fal-AI 최종 결과 로드 오류 ({final_resp.status_code}): {final_resp.text}")
                                
                            final_json = final_resp.json()
                            if "images" in final_json and final_json["images"]:
                                img_info = final_json["images"][0]
                                break
                            else:
                                raise RuntimeError("Fal-AI: 작업은 완료되었으나 생성된 이미지가 없습니다.")
                                
                        elif current_status in ["IN_QUEUE", "IN_PROGRESS"]:
                            continue # 작업 중이므로 루프를 계속 돌며 대기
                        else:
                            # 예상치 못한 상태이거나 폴링 중 이미지가 바로 내려왔을 경우 대비
                            if "images" in poll_json and poll_json["images"]:
                                img_info = poll_json["images"][0]
                                break
                    else:
                        raise RuntimeError("Fal-AI: 대기열 시간이 초과되었습니다.")
                
                # 큐를 타지 않고 즉시 반환된 경우 (동기 처리)
                elif "images" in result and result["images"]:
                    img_info = result["images"][0]
                else:
                    raise RuntimeError("Fal-AI: 응답 데이터 형식이 올바르지 않습니다.")

                # 3. 이미지 데이터 복구
                if img_info:
                    if "url" in img_info:
                        img_resp = client.get(img_info["url"], timeout=60.0)
                        img_resp.raise_for_status()
                        return Image.open(BytesIO(img_resp.content)).convert("RGB")
                    elif "base64" in img_info:
                        return Image.open(BytesIO(base64.b64decode(img_info["base64"]))).convert("RGB")

                raise RuntimeError("Fal-AI: 유효한 이미지 데이터를 수신하지 못했습니다.")

        except Exception as e:
            # 예외를 RuntimeError로 통합하여 워커 시그널로 전달
            raise RuntimeError(str(e))

    def call_hf_space_api(self, *, model_cfg, pil_images, mask, prompt, negative_prompt, num_inference_steps=30, hf_token=None, **kwargs):
        """ HF Space API 호출: Gradio Client의 예측 실패 및 연결 오류를 RuntimeError로 포워딩
            - API 호출 전후의 세션 유효성을 검증하고 실패 시 명시적 에러 메시지 생성
            - 임시 파일 생성 및 파일 핸들링 중 발생하는 IO 예외 처리 포함
        """
        try:
            client = self.get_hf_client(model_cfg.get("remote_url"), hf_token=hf_token)
            pipeline_type = model_cfg.get("pipeline_type", "").lower()
            api_name = model_cfg.get("api_model_uri", "/infer")
            
            if "qwen" in pipeline_type:
                # Qwen-Edit 전용 로직 (현재는 Placeholder 상태이나 에러 핸들링 구조 확보)
                if not pil_images: 
                    raise RuntimeError("Qwen 모델 실행을 위해 입력 이미지가 필요합니다.")
                # 실제 구현 시 client.predict 호출 및 결과 검증 로직 추가
                return Image.new("RGB", (1024, 1024))

            elif "inpaint" in pipeline_type:
                predict_kwargs = {
                    "prompt": prompt, "negative_prompt": negative_prompt, 
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": kwargs.get("guidance_scale", 7.5),
                    "api_name": api_name
                }
                
                # 이미지 및 마스크 임시 파일 처리
                temp_files = []
                if pil_images:
                    tf_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    pil_images[0].save(tf_img.name)
                    predict_kwargs["image"] = handle_file(tf_img.name)
                    temp_files.append(tf_img.name)
                
                if mask is not None:
                    tf_mask = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    Image.fromarray(mask).save(tf_mask.name)
                    predict_kwargs["mask_image"] = handle_file(tf_mask.name)
                    temp_files.append(tf_mask.name)
                    
                try:
                    res = client.predict(**predict_kwargs)
                    
                    # 결과 객체 파싱
                    img_path = None
                    if isinstance(res, str): img_path = res
                    elif isinstance(res, dict) and "image" in res: img_path = res["image"]
                    elif isinstance(res, (list, tuple)) and len(res) > 0: img_path = res[0]
                    
                    if img_path and os.path.exists(img_path):
                        return Image.open(img_path).convert("RGB")
                    else:
                        raise RuntimeError("HF Space: 생성이 완료되었으나 결과 파일 경로가 유효하지 않습니다.")
                        
                finally:
                    # 사용된 임시 파일 정리
                    for f in temp_files:
                        try: os.unlink(f)
                        except: pass
            
            raise RuntimeError(f"HF Space: 지원하지 않는 파이프라인 타입입니다. ({pipeline_type})")

        except Exception as e:
            # Gradio AppError 등을 포함한 모든 예외를 상위 워커로 전달
            raise RuntimeError(f"HF Space API 오류: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    qdarktheme.setup_theme("dark")
    app.setFont(QFont("Malgun Gothic", 10))
    win = BgComposerApp()
    win.show()
    sys.exit(app.exec())