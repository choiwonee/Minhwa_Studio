import os, sys
import qdarktheme
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QPainter, QColor, QMovie, QFont

# High DPI 자동 스케일링 활성화
# if hasattr(Qt, 'AA_EnableHighDpiScaling'):
#     from PySide6.QtWidgets import QApplication
#     QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
# if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
#     from PySide6.QtWidgets import QApplication
#     QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

# High DPI Scaling Fix: 윈도우 배율 설정(125% 등)에 따라 UI가 과도하게 확대되는 것을 방지합니다.
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

# CRITICAL: 메모리 최적화 환경변수 (반드시 import torch 이전에 설정)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True, max_split_size_mb:128"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Qwen 전용 최적화 (배경 합성 등에서 사용 가능성 대비)
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("[Launcher] Memory optimization environment variables configured")

# --- 경로 설정 --- 
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- 모듈 임포트 (오류 방지 처리) ---
try:
    from modules.masking import SAMGuiApp as MaskingApp
    from modules.effect_25d import Effect25DApp
    from modules.bg_composer import BgComposerApp  # 배경 합성 모듈
    # from modules.restoration import RestorationApp # 아직 없다면 주석 유지
except ImportError as e:
    print(f"Module Import Error: {e}")
    # 불러오기 실패 시 None 처리하여 프로그램 실행은 되도록 함
    MaskingApp = None
    Effect25DApp = None
    BgComposerApp = None

# 미구현 모듈용 Placeholder
class PlaceholderApp(QMainWindow):
    def __init__(self, title):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(600, 400)
        self.setCentralWidget(QLabel(f"{title}\n\n(Coming Soon - 개발 예정)", self, alignment=Qt.AlignCenter))

class TraditionalArtLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Minhwa Studio - Launcher")
        self.resize(1000, 700)
        
        # [리소스 설정] resources 폴더에 bg_main.gif (없으면 jpg)가 있어야 배경이 나온다.
        self.bg_gif_path = os.path.join(current_dir, "assets", "bg_main.gif") 
        self.bg_img_path = os.path.join(current_dir, "assets", "bg_main.jpg")
        
        # 배경 객체 초기화
        self.movie = None
        self.static_bg = None
        self._init_background()

        self.sub_window = None # 현재 열린 기능창 저장
        self.init_ui()

    def _init_background(self):
        """ 배경 리소스 로드 (GIF 우선 -> JPG -> 없으면 회색) """
        print(f"[Debug] GIF 경로 확인: {self.bg_gif_path}")
        print(f"[Debug] 파일 존재 여부: {os.path.exists(self.bg_gif_path)}")
        
        if os.path.exists(self.bg_gif_path):
            self.movie = QMovie(self.bg_gif_path)
            if self.movie.isValid(): # GIF가 정상적으로 로드되었는지 확인
                print("[Debug] GIF 정상 로드됨! 애니메이션 시작.")
                self.movie.frameChanged.connect(self.repaint)
                self.movie.start()
            else:
                print("[Debug] 에러: 파일은 있지만 GIF를 읽을 수 없습니다. (파일 손상 또는 포맷 문제)")
        elif os.path.exists(self.bg_img_path):
            self.static_bg = QPixmap(self.bg_img_path)

    def init_ui(self):
        cw = QWidget()
        
        # --- 추가된 부분: CentralWidget의 배경을 투명하게 만들어 밑바탕(GIF)이 보이게 함 ---
        cw.setObjectName("TransparentCentralWidget")
        cw.setStyleSheet("QWidget#TransparentCentralWidget { background-color: transparent; }")
        # -------------------------------------------------------------------------
        
        self.setCentralWidget(cw)
        
        # 메인 레이아웃 (여백 제거하여 배경 꽉 채움)
        main_layout = QVBoxLayout(cw)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. 상단 여백
        main_layout.addStretch(1)

        # 2. 타이틀 영역
        title_container = QWidget()
        title_container.setStyleSheet("background: transparent;")
        title_layout = QVBoxLayout(title_container)
        title_layout.setSpacing(5)
        
        lbl_title_en = QLabel("MINHWA STUDIO")
        lbl_title_en.setAlignment(Qt.AlignCenter)
        lbl_title_en.setStyleSheet("background: transparent;color: white; font-size: 56px; font-weight: Bold; letter-spacing: 2px; font-family: 'Arial';")
        
        lbl_title_ko = QLabel("이미지 합성 및 입체 창작 스튜디오")
        lbl_title_ko.setAlignment(Qt.AlignCenter)
        lbl_title_ko.setStyleSheet("color: #cccccc; font-size: 14px; font-weight: 500; font-family: 'Malgun Gothic';")

        title_layout.addWidget(lbl_title_en)
        title_layout.addWidget(lbl_title_ko)
        main_layout.addWidget(title_container)

        # 3. 중간 여백
        main_layout.addSpacing(50)

        # 4. 메뉴 버튼 영역
        menu_container = QWidget()
        menu_container.setStyleSheet("background: transparent;")
        menu_layout = QVBoxLayout(menu_container)
        menu_layout.setSpacing(15)
        menu_layout.setAlignment(Qt.AlignCenter)

        # [메뉴 정의] (라벨, 실행함수)
        menus = [
            ("2.5D Relief Studio", self.run_25d),
            ("Smart Mask Extractor", self.run_masking),
            ("Heritage Composer", self.run_composer),
            # ("Art Restorer", self.run_restorer) # Placeholder
        ]

        for text, func in menus:
            btn = QPushButton(text)
            btn.setFixedSize(420, 55)
            btn.setCursor(Qt.PointingHandCursor)
            # 스타일: 반투명 배경 + 흰색 테두리 + 호버 효과
            btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(255, 255, 255, 0.08);
                    color: #ecf0f1;
                    border: 1px solid rgba(255, 255, 255, 0.3);
                    border-radius: 27px;
                    font-size: 15px;
                    font-weight: 600;
                    font-family: 'Malgun Gothic', sans-serif;
                    padding-left: 0 20px;
                }
                QPushButton:hover {
                    background-color: rgba(255, 255, 255, 0.2);
                    border: 1px solid rgba(255, 255, 255, 0.8);
                    color: #ffffff;
                }
                QPushButton:pressed {
                    background-color: rgba(255, 255, 255, 0.05);
                    padding-top: 2px;
                }
            """)
            btn.clicked.connect(func)
            menu_layout.addWidget(btn)

        main_layout.addWidget(menu_container)

        # 5. 하단 여백 및 종료 버튼
        main_layout.addStretch(2)
        
        bot_layout = QHBoxLayout()
        bot_layout.setContentsMargins(0, 0, 30, 30)
        bot_layout.addStretch(1)
        
        btn_exit = QPushButton("EXIT")
        btn_exit.setFixedSize(100, 40)
        btn_exit.setCursor(Qt.PointingHandCursor)
        btn_exit.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #7f8c8d;
                font-weight: bold;
                border: none;
                font-size: 14px;
            }
            QPushButton:hover { color: #c0392b; }
        """)
        btn_exit.clicked.connect(self.close)
        bot_layout.addWidget(btn_exit)
        
        main_layout.addLayout(bot_layout)

    # --- 배경 그리기 (GIF + 셀로판지 효과) ---
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 1. 배경 이미지/GIF 그리기 (화면 꽉 채우기)
        if self.movie and not self.movie.currentPixmap().isNull(): # isNull() 체크 추가
            pixmap = self.movie.currentPixmap()
            scaled_pix = pixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            
            # 중앙 크롭(Crop) 계산
            x = (self.width() - scaled_pix.width()) // 2
            y = (self.height() - scaled_pix.height()) // 2
            painter.drawPixmap(x, y, scaled_pix)
            
        elif self.static_bg:
            scaled_pix = self.static_bg.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            x = (self.width() - scaled_pix.width()) // 2
            y = (self.height() - scaled_pix.height()) // 2
            painter.drawPixmap(x, y, scaled_pix)
        else:
            # 배경 없으면 어두운 회색
            painter.fillRect(self.rect(), QColor(30, 30, 30))

        # 2. 셀로판지 효과 (반투명 검정 오버레이): Alpha 값(0~255)을 조절하여 배경의 어두운 정도 설정 (180 = 약 70% 어둡게)
        overlay_color = QColor(0, 0, 0, 180) 
        painter.fillRect(self.rect(), overlay_color)

    # --- 단일 창 전환 로직 ---
    def launch_module(self, app_class, error_msg="모듈을 로드할 수 없습니다."):
        """ 1. 런처 숨김 
            2. 기능 창 실행 
            3. 기능 창 종료 시 런처 복구 
        """
        if app_class is None:
            QMessageBox.warning(self, "Module Error", f"{error_msg}\n(Import Failed)")
            return

        # 1. 메인 런처 숨기기
        self.hide()
        
        # 2. 서브 윈도우 생성
        try:
            self.sub_window = app_class()
        except Exception as e:
            self.show()
            QMessageBox.critical(self, "Execution Error", f"모듈 실행 중 오류 발생:\n{e}")
            return
        
        # 3. 종료 이벤트 후킹 (Hook): 닫히면 런처를 다시 띄움
        original_close_event = self.sub_window.closeEvent

        def new_close_event(event):
            # 원래 종료 로직 수행
            original_close_event(event)
            # 메인 런처 복구
            self.show()
            # 참조 제거
            self.sub_window = None

        self.sub_window.closeEvent = new_close_event
        
        # 4. 화면 표시
        self.sub_window.show()

    # --- 각 기능 실행 함수 ---
    def run_25d(self):
        self.launch_module(Effect25DApp, "Effect 2.5D 모듈이 로드되지 않았습니다.")

    def run_masking(self):
        self.launch_module(MaskingApp, "Masking 모듈이 로드되지 않았습니다.")

    def run_composer(self):
        self.launch_module(BgComposerApp, "BgComposer 모듈이 로드되지 않았습니다.")

    def run_restorer(self):
        # 복원은 아직 미구현이므로 PlaceholderApp 실행
        self.launch_module(lambda: PlaceholderApp("Art Restorer"), "Restoration 모듈 오류")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 모던 다크 테마 강제 적용, 윈도우 시스템 설정이 라이트모드여도 무조건 예쁜 다크모드로 실행.
    qdarktheme.setup_theme("dark")
    # 폰트 설정 (선택 사항)
    font = QFont("Malgun Gothic", 10)
    app.setFont(font)

    win = TraditionalArtLauncher()
    win.show()
    sys.exit(app.exec())