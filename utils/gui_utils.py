import os
import cv2
import numpy as np
import math
import traceback
from PIL import Image, ImageOps
from utils.common import qimage_from_ndarray, SuppressStderr

try:
    from utils import token_key
except ImportError:
    token_key = None
    
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFrame, QGraphicsDropShadowEffect, QGridLayout, QGroupBox, QLabel, QLineEdit, QMessageBox, QHBoxLayout, QPushButton, QSizePolicy, 
    QSlider, QVBoxLayout, QWidget
)
from PySide6.QtCore import QPoint, Qt, QThread, QTimer, QRect, QRectF, Signal
from PySide6.QtGui import (
    QBrush, QColor, QFont, QImage, QKeyEvent, QLinearGradient, 
    QPainter, QPainterPath, QPen, QPixmap, QPolygon, QTransform
)

# ==============================================================================
# Generic Worker
# ==============================================================================
class GenericWorker(QThread):
    error = Signal(object)
    signal_finished = Signal(object)       
    signal_progress = Signal(int)
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._result = None
        self._exc = None
        self.finished.connect(self._emit_outcome)
    
    def run(self):
        try:
            self.kwargs['progress_callback'] = self.signal_progress.emit
            self.kwargs.setdefault('abort_check', self.isInterruptionRequested)
            self._result = self.func(*self.args, **self.kwargs)
        except Exception as e:
            e_str = str(e)
            # 사용자 취소 또는 API 쿼터 초과와 같은 '알려진 오류'는 Traceback 출력 제외
            if "USER_CANCEL" in e_str:
                self._exc = e 
            elif "Quota Exceeded" in e_str or "RESOURCE_EXHAUSTED" in e_str:
                # 콘솔 스팸 방지: 단순 에러 객체만 저장 (UI에서 메시지박스로 출력됨)
                self._exc = e
            else:
                # 그 외 실제 코드 오류만 Traceback 출력
                traceback.print_exc()
                self._exc = e
        finally:
            try:
                self.func = None
                self.args = None
                self.kwargs = None
            except Exception:
                pass

    def _emit_outcome(self):
        """ 결과를 안전하게 내보내고 객체 참조 정리 run()의 finally와 중복되지 않도록 여기서 최종 정리 수행 """
        try:
            if self._exc is not None:
                self.error.emit(self._exc)
            else:
                self.signal_finished.emit(self._result)
        finally: # 참조를 None으로 밀어 메모리 해제 유도
            self._result = None
            self._exc = None
            self.func = None

# ==============================================================================
# FloatingToolBar
# ==============================================================================
class FloatingToolBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        self.setFixedWidth(140)
        self.setStyleSheet("""
            QWidget { background-color: rgba(30, 30, 30, 220); border-radius: 10px; border: 1px solid #555; }
            QLabel { background-color: transparent; color: #ecf0f1; font-weight: bold; font-size: 12px; border: none; margin-top: 5px; }
            QPushButton { background-color: #34495e; color: white; border: 1px solid #2c3e50; border-radius: 4px; padding: 6px; font-size: 11px; text-align: left; }
            QPushButton:hover { background-color: #3e5871; }
            QPushButton:checked { background-color: #2980b9; border: 1px solid #1f618d; }
        """)

    def add_section_label(self, text):
        self.layout().addWidget(QLabel(text))
        
    def add_separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: rgba(255, 255, 255, 50); border: none;")
        self.layout().addWidget(line)

# ==============================================================================
# ImageCanvas
# ==============================================================================
class ImageCanvas(QLabel):
    sig_view_changed = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.image = None
        self.qimage = None
        self.pixmap_img = None
        self.img_w = 0
        self.img_h = 0
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.min_scale = 0.1
        self.max_scale = 8.0
        self.mode = "view"
        self.pan_active = False
        self.pan_start = None
        self.last_mouse_pos = QPoint(0,0)
        self.box_img = None
        self.box_drawing = False
        self.box_start = None
        self.lasso_img = []
        self.current_lasso_preview = []
        self.point_list = []
        self.brush_strokes = []       
        self.current_brush_stroke = [] 
        self.brush_size = 20
        self.overlay_mask = None
        self.overlay_pixmap = None
        self.show_crosshair = True
        self.crosshair_pos = None
        self.history = []
        self.max_history = 50
        self.on_selection_done = None
        self.on_selection_cancelled = None
        self.on_undo = None
        self.on_point_added = None
        self.auto_scroll_timer = QTimer(self)
        self.auto_scroll_timer.setInterval(30)
        self.auto_scroll_timer.timeout.connect(self.perform_auto_scroll)
        self.scroll_margin = 40
        self.scroll_step = 20
        self.pen_box = QPen(QColor(0, 255, 50), 2)
        self.pen_lasso_preview = QPen(QColor(0, 255, 255), 2, Qt.DashLine)
        self.pen_lasso_done = QPen(QColor(255, 0, 255), 2)
        self.pen_point_border = QPen(QColor(255, 255, 255), 2)
        self.brush_point_pos = QBrush(QColor(0, 255, 50))
        self.brush_point_neg = QBrush(QColor(255, 0, 0))
        self.pen_brush_preview = QPen(QColor(0, 255, 0, 150), 1, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)

    def set_mode(self, mode):
        self.mode = mode
        self.setCursor(Qt.OpenHandCursor if mode == "pan" else Qt.ArrowCursor)
        self.update()

    def set_image(self, img: np.ndarray, preloaded_qimage=None):
        self.image = img
        if img is None:
            self.img_w = 0
            self.img_h = 0
            self.qimage = None
            self.pixmap_img = None
        else:
            self.img_h, self.img_w = img.shape[:2]
            self.qimage = preloaded_qimage if preloaded_qimage else qimage_from_ndarray(img)
            self.pixmap_img = QPixmap.fromImage(self.qimage)
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.reset_selection()
        self.updateGeometry()
        self.update()
        self.sig_view_changed.emit()

    def reset_selection(self):
        self.box_img = None
        self.box_drawing = False
        self.box_start = None
        self.lasso_img = []
        self.current_lasso_preview = []
        self.point_list = []
        self.brush_strokes = []
        self.current_brush_stroke = []
        self.overlay_mask = None
        self.overlay_pixmap = None
        self.update()

    def set_brush_size(self, size):
        self.brush_size = size
        self.update()

    def load_from_file(self, file_path, log_func=None):
        if not os.path.exists(file_path):
            if log_func: log_func(f"[Error] File not found: {file_path}")
            return False
        try:
            with SuppressStderr():
                pil = Image.open(file_path).convert("RGBA")
                pil = ImageOps.exif_transpose(pil)
                img_np = np.array(pil)
            self.set_image(img_np)
            self.fit_to_window()
            if log_func: log_func(f"View loaded: {os.path.basename(file_path)}")
            return True
        except Exception as e:
            if log_func: log_func(f"Failed to load image: {e}")
            return False

    def fit_to_window(self):
        if self.image is None or self.width() == 0 or self.height() == 0:
            return
        ratio = min(self.width() / self.img_w, self.height() / self.img_h) * 0.98
        self.scale = max(0.01, ratio)
        self.offset_x = (self.width() - self.img_w * self.scale) / 2
        self.offset_y = (self.height() - self.img_h * self.scale) / 2
        self.update()
        self.sig_view_changed.emit()

    def set_actual_size(self):
        if self.image is None:
            return
        self.scale = 1.0
        self.offset_x = (self.width() - self.img_w) / 2
        self.offset_y = (self.height() - self.img_h) / 2
        self.update()
        self.sig_view_changed.emit()

    def set_image_keep_view(self, img: np.ndarray):
        if self.image is None:
            self.set_image(img)
            return
        prev_s, prev_x, prev_y = self.scale, self.offset_x, self.offset_y
        self.set_image(img)
        self.scale, self.offset_x, self.offset_y = prev_s, prev_x, prev_y
        self.update()
        self.sig_view_changed.emit()

    def set_overlay_mask(self, mask_np):
        if mask_np is None:
            self.overlay_mask = None
            self.overlay_pixmap = None
        else:
            if mask_np.shape[:2] != (self.img_h, self.img_w):
                mask_np = cv2.resize((mask_np>0).astype(np.uint8), (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
            self.overlay_mask = (mask_np > 0).astype(np.uint8)
            h, w = self.img_h, self.img_w
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[..., 0] = 255; rgba[..., 1] = 50; rgba[..., 2] = 50
            rgba[..., 3] = (self.overlay_mask * 110).astype(np.uint8)
            try:
                contours, _ = cv2.findContours(self.overlay_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgba, contours, -1, (255, 255, 0, 200), thickness=2) 
            except Exception: pass
            qimg = QImage(rgba.data, w, h, rgba.strides[0], QImage.Format_RGBA8888)
            self.overlay_pixmap = QPixmap.fromImage(qimg)
        self.update()

    def save_state(self):
        state = {'box_img': self.box_img, 'lasso_img': list(self.lasso_img), 'point_list': list(self.point_list), 'brush_strokes': [s.copy() for s in self.brush_strokes]}
        self.history.append(state)
        if len(self.history) > self.max_history: self.history.pop(0)

    def undo_last_action(self):
        if not self.history: return False
        state = self.history.pop()
        self.box_img = state['box_img']
        self.lasso_img = state['lasso_img']
        self.point_list = state['point_list']
        self.brush_strokes = state.get('brush_strokes', [])
        self.update()
        if callable(self.on_undo): self.on_undo(True, "Undo: Restored previous state")
        return True

    def image_to_view(self, x, y):
        return int(round(x * self.scale + self.offset_x)), int(round(y * self.scale + self.offset_y))

    def view_to_image(self, x, y):
        ix = (x - self.offset_x) / self.scale
        iy = (y - self.offset_y) / self.scale
        return int(max(0, min(self.img_w - 1, round(ix)))), int(max(0, min(self.img_h - 1, round(iy))))

    def paintEvent(self, event):
        super().paintEvent(event) 
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self.pixmap_img is None:
            painter.fillRect(self.rect(), QColor(30, 30, 30))
            painter.end()
            return
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        target_rect = QRectF(self.offset_x, self.offset_y, self.img_w * self.scale, self.img_h * self.scale)
        painter.drawPixmap(target_rect, self.pixmap_img, QRectF(self.pixmap_img.rect()))
        if self.overlay_pixmap:
            painter.drawPixmap(target_rect, self.overlay_pixmap, QRectF(self.overlay_pixmap.rect()))
        if self.box_img:
            l, t, r, b = self.box_img
            x0, y0 = self.image_to_view(l, t)
            x1, y1 = self.image_to_view(r, b)
            self.pen_box.setWidth(max(2, int(3 * self.scale)))
            painter.setPen(self.pen_box)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(min(x0,x1), min(y0,y1), abs(x1-x0), abs(y1-y0))
        if len(self.lasso_img) > 1:
            pen = QPen(QColor(255, 0, 255), max(2, int(3 * self.scale)))
            painter.setPen(pen)
            pts = [QPoint(*self.image_to_view(x, y)) for x, y in self.lasso_img]
            painter.drawPolyline(pts)
            if len(self.lasso_img) > 2: painter.drawLine(pts[-1], pts[0])
        if len(self.current_lasso_preview) > 1:
            pen = QPen(QColor(0, 255, 255), max(2, int(3 * self.scale)), Qt.DashLine)
            painter.setPen(pen)
            pts = [QPoint(*self.image_to_view(x, y)) for x, y in self.current_lasso_preview]
            painter.drawPolyline(pts)
        for x, y, lbl in self.point_list:
            vx, vy = self.image_to_view(x, y)
            r = max(4, int(7 * self.scale))
            painter.setPen(self.pen_point_border)
            painter.setBrush(self.brush_point_pos if lbl == 1 else self.brush_point_neg)
            painter.drawEllipse(QPoint(vx, vy), r, r)
        for stroke in self.brush_strokes:
            points = stroke['points']; size = stroke['size']
            view_points = [QPoint(*self.image_to_view(x, y)) for x, y in points]
            pen = QPen(QColor(0, 255, 0, 100), max(1, int(size * self.scale)), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            if len(view_points) > 0: painter.drawPolyline(view_points)
        if self.current_brush_stroke:
            view_points = [QPoint(*self.image_to_view(x, y)) for x, y in self.current_brush_stroke]
            pen = QPen(QColor(0, 255, 0, 150), max(1, int(self.brush_size * self.scale)), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            if len(view_points) > 0: painter.drawPolyline(view_points)
        if self.mode == "brush" and self.last_mouse_pos and not self.pan_active:
            mx, my = self.last_mouse_pos.x(), self.last_mouse_pos.y()
            cursor_size = max(1, int(self.brush_size * self.scale))
            painter.setPen(QPen(QColor(255, 255, 255), 1, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPoint(mx, my), cursor_size // 2, cursor_size // 2)
        if self.show_crosshair and self.crosshair_pos and not self.box_drawing and not self.pan_active and self.mode != "view":
            mx, my = self.crosshair_pos.x(), self.crosshair_pos.y()
            painter.setPen(QPen(QColor(0, 255, 255, 200), 1, Qt.DashLine))
            painter.drawLine(mx, 0, mx, self.height())
            painter.drawLine(0, my, self.width(), my)
        if self.image is not None:
            text = f"{self.img_w} x {self.img_h}"
            painter.setFont(QFont("Arial", 9, QFont.Bold))
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(text); th = fm.height()
            bx, by = self.width() - tw - 22, self.height() - th - 18
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(0, 0, 0, 160))
            painter.drawRoundedRect(bx, by, tw+12, th+8, 4, 4)
            painter.setPen(QColor(220, 220, 220))
            painter.drawText(bx, by, tw+12, th+8, Qt.AlignCenter, text)
        painter.end()

    def mousePressEvent(self, event):
        if self.image is None: return
        vx, vy = event.position().x(), event.position().y()
        self.last_mouse_pos = event.position()
        if (event.modifiers() & Qt.ShiftModifier) or (event.button() == Qt.MiddleButton) or (self.mode == "pan" and event.button() == Qt.LeftButton):
            self.pan_active = True
            self.pan_start = (vx, vy, self.offset_x, self.offset_y)
            if self.mode == "pan": self.setCursor(Qt.ClosedHandCursor)
            return
        ix, iy = self.view_to_image(vx, vy)
        if self.mode == "box" and event.button() == Qt.LeftButton:
            self.save_state(); self.box_drawing = True; self.box_start = (ix, iy); self.box_img = (ix, iy, ix, iy)
        elif self.mode == "lasso" and event.button() == Qt.LeftButton:
            self.save_state(); self.current_lasso_preview = [(ix, iy)]
        elif self.mode == "point":
            self.save_state(); lbl = 1 if event.button() == Qt.LeftButton else 0; self.point_list.append((ix, iy, lbl))
            if callable(self.on_point_added): self.on_point_added((ix, iy, lbl))
        elif self.mode == "brush" and event.button() == Qt.LeftButton:
            self.save_state(); self.current_brush_stroke = [(ix, iy)]
        self.update()

    def mouseMoveEvent(self, event):
        self.crosshair_pos = event.position()
        self.last_mouse_pos = event.position()
        if self.image is None: self.update(); return
        vx, vy = event.position().x(), event.position().y()
        if self.pan_active:
            sx, sy, ox, oy = self.pan_start
            self.offset_x = ox + (vx - sx)
            self.offset_y = oy + (vy - sy)
            self.update(); self.sig_view_changed.emit()
            return
        if self.box_drawing:
            margin = self.scroll_margin; w, h = self.width(), self.height()
            if (vx < margin or vx > w - margin or vy < margin or vy > h - margin):
                if not self.auto_scroll_timer.isActive(): self.auto_scroll_timer.start()
            else:
                if self.auto_scroll_timer.isActive(): self.auto_scroll_timer.stop()
        ix, iy = self.view_to_image(vx, vy)
        if self.box_drawing and self.box_start:
            ix0, iy0 = self.box_start
            self.box_img = (ix0, iy0, ix, iy)
        elif self.mode == "lasso" and self.current_lasso_preview:
            if (ix, iy) != self.current_lasso_preview[-1]: self.current_lasso_preview.append((ix, iy))
        elif self.mode == "brush" and self.current_brush_stroke:
            last_pt = self.current_brush_stroke[-1]
            if (abs(ix - last_pt[0]) + abs(iy - last_pt[1])) > 0: self.current_brush_stroke.append((ix, iy))
        self.update()

    def mouseReleaseEvent(self, event):
        self.auto_scroll_timer.stop()
        if self.image is None: return
        if self.pan_active:
            self.pan_active = False; self.pan_start = None
            if self.mode == "pan": self.setCursor(Qt.OpenHandCursor)
            return
        if self.box_drawing:
            self.box_drawing = False
            if self.box_img:
                l, t, r, b = self.box_img
                self.box_img = (min(l,r), min(t,b), max(l,r), max(t,b))
                if callable(self.on_selection_done): self.on_selection_done("box", self.box_img)
        elif self.mode == "lasso" and len(self.current_lasso_preview) > 2:
            self.lasso_img = list(self.current_lasso_preview); self.current_lasso_preview = []
            if callable(self.on_selection_done): self.on_selection_done("lasso", self.lasso_img)
        elif self.mode == "brush" and self.current_brush_stroke:
            self.brush_strokes.append({'points': self.current_brush_stroke, 'size': self.brush_size})
            self.current_brush_stroke = []
            if callable(self.on_selection_done): self.on_selection_done("brush", None)
        self.update()

    def wheelEvent(self, event):
        if self.pixmap_img is None: return
        factor = 1.0 + (0.0015 * event.angleDelta().y())
        new_scale = max(self.min_scale, min(self.max_scale, self.scale * factor))
        mx, my = event.position().x(), event.position().y()
        ix = (mx - self.offset_x) / self.scale
        iy = (my - self.offset_y) / self.scale
        self.scale = new_scale
        self.offset_x = mx - ix * self.scale
        self.offset_y = my - iy * self.scale
        self.update(); self.sig_view_changed.emit()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Escape: self.cancel_current_selection()
        elif event.key() == Qt.Key_Z and event.modifiers() == Qt.ControlModifier: self.undo_last_action()
        else: super().keyPressEvent(event)

    def cancel_current_selection(self):
        updated = False
        if self.box_drawing: self.box_drawing = False; self.box_start = None; self.box_img = None; updated = True
        elif self.current_lasso_preview: self.current_lasso_preview = []; updated = True
        elif self.box_img: self.save_state(); self.box_img = None; updated = True
        elif self.lasso_img: self.save_state(); self.lasso_img = []; updated = True
        elif self.point_list: self.save_state(); self.point_list = []; updated = True
        elif self.brush_strokes: self.save_state(); self.brush_strokes = []; updated = True
        if updated:
            self.update()
            if callable(self.on_selection_cancelled): self.on_selection_cancelled("Selection cleared")

    def perform_auto_scroll(self):
        if not self.box_drawing or self.image is None: return
        mx, my = self.last_mouse_pos.x(), self.last_mouse_pos.y()
        w, h = self.width(), self.height()
        dx, dy = 0, 0
        if mx < self.scroll_margin: dx = self.scroll_step
        elif mx > w - self.scroll_margin: dx = -self.scroll_step
        if my < self.scroll_margin: dy = self.scroll_step
        elif my > h - self.scroll_margin: dy = -self.scroll_step
        if dx or dy:
            self.offset_x += dx; self.offset_y += dy
            ix, iy = self.view_to_image(mx, my)
            if self.box_start:
                ix0, iy0 = self.box_start
                self.box_img = (ix0, iy0, ix, iy)
            self.update(); self.sig_view_changed.emit()

    def set_show_crosshair(self, enabled: bool):
        self.show_crosshair = enabled
        self.update()
        
    def leaveEvent(self, event):
        self.crosshair_pos = None
        self.update()
        super().leaveEvent(event)

    def clear_all_overlays(self):
        """ 캔버스 위에 그려진 모든 드로잉 및 오버레이 시각 요소 제거 """
        # 1. 오버레이 마스크(외부 파일 등) 데이터 삭제
        self.overlay_mask = None
        self.overlay_pixmap = None
        
        # 2. 선택 도구(Box, Lasso, Brush) 데이터 초기화
        self.box_img = None
        self.box_drawing = False

        self.lasso_img = []
        self.current_lasso_preview = []

        self.point_list = []

        self.brush_strokes = []
        self.current_brush_stroke = []

        # 3. 히스토리 및 상태 초기화
        self.history.clear()
        self.crosshair_pos = None
        self.last_mouse_pos = None

        # 4. 화면 강제 갱신 (update -> repaint), update()는 이벤트 큐에 따라 지연될 수 있으므로 repaint()로 즉시 지움
        self.repaint()
       
# ==============================================================================
# UI Components
# ==============================================================================
class InfiniteLoadingBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(4)
        self.offset = 0
        self.chunk_width = 0.3
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_anim)
        self.timer.start(15)

    def update_anim(self):
        if self.isVisible():
            self.offset += 0.015
            if self.offset > 1.3: self.offset = -0.3
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.setBrush(QColor("#404040"))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(0, 0, w, h, 2, 2)
        start_x = int(self.offset * w)
        bar_w = int(self.chunk_width * w)
        grad = QLinearGradient(start_x, 0, start_x + bar_w, 0)
        grad.setColorAt(0, QColor(230, 126, 34, 0))
        grad.setColorAt(0.5, QColor(243, 156, 18, 255))
        grad.setColorAt(1, QColor(230, 126, 34, 0))
        p.setBrush(QBrush(grad))
        p.drawRoundedRect(QRect(start_x, 0, bar_w, h).intersected(self.rect()), 2, 2)
        p.end()

class ProcessingOverlay(QWidget):
    sig_cancel_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setHidden(True)
        self.setStyleSheet("border: none; background: transparent;")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        self.card = QFrame()
        self.card.setFixedSize(340, 200)
        self.card.setStyleSheet("background: #2b2b2b; border: 1px solid #3d3d3d; border-radius: 12px;")
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20); shadow.setYOffset(10); shadow.setColor(QColor(0,0,0,100))
        self.card.setGraphicsEffect(shadow)
        cl = QVBoxLayout(self.card)
        cl.setContentsMargins(30,25,30,25); cl.setSpacing(15)
        self.lbl_title = QLabel("PROCESSING")
        self.lbl_title.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        self.loading_bar = InfiniteLoadingBar()
        self.lbl_desc = QLabel("Please wait...")
        self.lbl_desc.setStyleSheet("color: #aaa; font-size: 13px;")
        self.lbl_desc.setAlignment(Qt.AlignCenter)
        self.btn_cancel = QPushButton("🛑 STOP")
        self.btn_cancel.setFixedWidth(120)
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.setStyleSheet("QPushButton { background-color: #c0392b; color: white; border-radius: 18px; padding: 6px; font-weight: bold; border: 1px solid #e74c3c; } QPushButton:hover { background-color: #e74c3c; } QPushButton:disabled { background-color: #555; border: 1px solid #444; color: #888; }")
        self.btn_cancel.clicked.connect(self.on_cancel)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch(); btn_layout.addWidget(self.btn_cancel); btn_layout.addStretch()
        cl.addWidget(self.lbl_title); cl.addWidget(self.loading_bar); cl.addWidget(self.lbl_desc); cl.addLayout(btn_layout)
        layout.addWidget(self.card)

    def on_cancel(self):
        self.lbl_desc.setText("Stopping...")
        self.btn_cancel.setEnabled(False)
        self.sig_cancel_requested.emit()

    def set_message(self, title, desc="Please wait..."):
        self.lbl_title.setText(title)
        self.lbl_desc.setText(desc)
        self.btn_cancel.setEnabled(True)
        self.btn_cancel.setText("🛑 STOP")

    def paintEvent(self, e):
        QPainter(self).fillRect(self.rect(), QColor(0,0,0,160))

class KeySettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Authentication Settings")
        self.setFixedWidth(420)
        self.setStyleSheet("""
            QDialog { background-color: #2b2b2b; color: #ecf0f1; }
            QLabel { color: #bdc3c7; font-size: 12px; }
            QLineEdit { 
                background-color: #3b3b3b; color: #fff; 
                border: 1px solid #555; padding: 6px; border-radius: 4px; 
            }
            QLineEdit:focus { border: 1px solid #3498db; }
            QGroupBox { 
                border: 1px solid #555; border-radius: 6px; 
                margin-top: 20px; font-weight: bold; color: #ddd; 
                padding-top: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
            QCheckBox { color: #aaa; spacing: 5px; }
            QCheckBox::indicator { width: 14px; height: 14px; }
            QPushButton { 
                background-color: #3498db; color: white; border: none; 
                padding: 8px 16px; border-radius: 4px; font-weight: bold; 
            }
            QPushButton:hover { background-color: #2980b9; }
            QPushButton#SaveBtn { background-color: #27ae60; }
            QPushButton#SaveBtn:hover { background-color: #2ecc71; }
        """)
        
        self.init_ui()
        self.load_current_keys()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # 안내 문구
        lbl_info = QLabel("Manage your API keys securely (AES-256 Encrypted).")
        lbl_info.setStyleSheet("color: #aaa; font-size: 11px; margin-bottom: 5px;")
        layout.addWidget(lbl_info)

        # 1. Hugging Face Token Section
        self.hf_group = QGroupBox("Hugging Face Token")
        hf_layout = QVBoxLayout()
        
        self.txt_hf_token = QLineEdit()
        self.txt_hf_token.setEchoMode(QLineEdit.Password) # 기본 암호화 표시
        self.txt_hf_token.setPlaceholderText("Paste HF Write Token...")
        
        self.chk_show_hf = QCheckBox("Show Token")
        self.chk_show_hf.toggled.connect(lambda c: self.txt_hf_token.setEchoMode(QLineEdit.Normal if c else QLineEdit.Password))
        
        hf_layout.addWidget(self.txt_hf_token)
        hf_layout.addWidget(self.chk_show_hf)
        self.hf_group.setLayout(hf_layout)
        layout.addWidget(self.hf_group)

        # 2. Remote / Google API Key Section
        self.api_group = QGroupBox("Remote / Google API Key")
        api_layout = QVBoxLayout()
        
        self.txt_api_key = QLineEdit()
        self.txt_api_key.setEchoMode(QLineEdit.Password) # 기본 암호화 표시
        self.txt_api_key.setPlaceholderText("Paste Google GenAI / Remote API Key...")
        
        self.chk_show_api = QCheckBox("Show Key")
        self.chk_show_api.toggled.connect(lambda c: self.txt_api_key.setEchoMode(QLineEdit.Normal if c else QLineEdit.Password))
        
        api_layout.addWidget(self.txt_api_key)
        api_layout.addWidget(self.chk_show_api)
        self.api_group.setLayout(api_layout)
        layout.addWidget(self.api_group)

        # 3. Action Buttons
        btns = QHBoxLayout()
        btns.addStretch()
        
        self.btn_save = QPushButton("Save && Encrypt")
        self.btn_save.setObjectName("SaveBtn")
        self.btn_save.clicked.connect(self.save_keys)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        
        btns.addWidget(self.btn_save)
        btns.addWidget(self.btn_cancel)
        layout.addLayout(btns)

    def load_current_keys(self):
        if token_key:
            hf = token_key.get_valid_hf_token()
            api = token_key.get_valid_api_key()
            if hf: self.txt_hf_token.setText(hf)
            if api: self.txt_api_key.setText(api)

    def save_keys(self):
        hf_val = self.txt_hf_token.text().strip()
        api_val = self.txt_api_key.text().strip()

        if not token_key:
            QMessageBox.critical(self, "Error", "token_key module not loaded.")
            return

        # HF Token 저장
        success_hf, msg_hf = token_key.save_hf_token(hf_val)
        # API Key 저장
        success_api, msg_api = token_key.save_api_key(api_val)

        if success_hf and success_api:
            QMessageBox.information(self, "Success", "Authentication keys saved securely.")
            self.accept()
        else:
            QMessageBox.critical(self, "Error", f"Save Failed:\nHF: {msg_hf}\nAPI: {msg_api}")

class ArrowButton(QPushButton):
    """ 벡터 그래픽으로 삼각형 화살표를 그리는 커스텀 버튼
        - direction: 'left' (감소/하락), 'right' (증가/상승)
        - 이미지 파일 없이 코드로 렌더링하므로 선명함 유지
    """
    def __init__(self, direction="right", parent=None):
        super().__init__(parent)
        self.direction = direction
        self.setFixedSize(24, 24)
        self.setCursor(Qt.PointingHandCursor)
        # 기본 배경 스타일
        self.setStyleSheet("""
            QPushButton {
                background-color: #3e3e3e;
                border: 1px solid #555;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #4e4e4e; border-color: #777; }
            QPushButton:pressed { background-color: #222; border-color: #333; }
        """)

    def paintEvent(self, event):
        # 1. 배경(StyleSheet) 그리기
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 2. 삼각형 좌표 계산
        rect = self.rect()
        w, h = rect.width(), rect.height()
        cx, cy = w / 2, h / 2
        
        # 삼각형 크기 (반지름 느낌)
        size = 5.0
        
        # 삼각형 폴리곤 정의
        points = QPolygon()
        
        if self.direction == "left": # ◀ (Decrease)
            # 오른쪽 위, 오른쪽 아래, 왼쪽 중앙
            points.append(QPoint(int(cx + size), int(cy - size)))
            points.append(QPoint(int(cx + size), int(cy + size)))
            points.append(QPoint(int(cx - size), int(cy)))
            
        elif self.direction == "right": # ▶ (Increase)
            # 왼쪽 위, 왼쪽 아래, 오른쪽 중앙
            points.append(QPoint(int(cx - size), int(cy - size)))
            points.append(QPoint(int(cx - size), int(cy + size)))
            points.append(QPoint(int(cx + size), int(cy)))
            
        # 3. 상태별 색상 설정 (Normal / Hover / Pressed)
        if self.isDown():
            color = QColor(0, 255, 255) # 눌렀을 때 Cyan
        elif self.underMouse():
            color = QColor(255, 255, 255) # 오버 시 White
        else:
            color = QColor(180, 180, 180) # 평소 Grey
            
        # 4. 그리기
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(points)
        painter.end()

# OrbitCameraVisualizer (Updated: 3D Projection & Corrected Arc)
class OrbitCameraVisualizer(QWidget):
    """ Multiangle Camera 스타일의 3D 궤도 시각화 위젯 
        수직/수평 각도에 따라 내부 이미지 평면이 입체적으로 회전하며, 
        수직 이동 호(Arc)가 구(Sphere)의 궤적을 따라 안쪽으로 휘도록 수정됨.
    """
    sig_angle_changed = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #121212; border-radius: 12px; border: 1px solid #333;")
        
        self.h_angle = 0.0
        self.v_angle = 0.0
        self.zoom = 1.0
        self.pixmap = None
        self.default_pixmap = self._create_default_thumbnail()
        self.is_dragging = False
        self.last_pos = QPoint()

        # 컬러 설정
        self.col_bg = QColor(18, 18, 24)
        self.col_orbit = QColor(255, 40, 110)   # Pink
        self.col_vert_arc = QColor(0, 240, 240) # Cyan
        self.col_beam = QColor(255, 180, 0)     # Orange
        self.col_camera = QColor(255, 60, 100)
        self.col_grid = QColor(255, 255, 255, 30)

    def _create_default_thumbnail(self):
        """ 이미지가 없을 때 보여줄 기본 타겟 평면 생성 """
        pm = QPixmap(250, 250)
        pm.fill(QColor(30, 30, 35))
        p = QPainter(pm)
        p.setPen(QPen(QColor(80, 80, 100), 2))
        p.drawRect(5, 5, 240, 240)
        p.setPen(QPen(QColor(80, 80, 100, 100), 1, Qt.DashLine))
        p.drawLine(125, 0, 125, 250)
        p.drawLine(0, 125, 250, 125)
        p.end()
        return pm

    def set_thumbnail(self, img_np):
        if img_np is None:
            self.pixmap = None
        else:
            h, w = img_np.shape[:2]
            scale = min(400/w, 400/h)
            img_resized = cv2.resize(img_np, (0,0), fx=scale, fy=scale) if scale < 1.0 else img_np
            self.pixmap = QPixmap.fromImage(qimage_from_ndarray(img_resized))
        self.update()

    def set_values(self, h, v, z):
        self.h_angle = h
        self.v_angle = v
        self.zoom = z
        self.update()

    def _project_3d(self, x, y, z, cx, cy, orbit_rx):
        """ 3D 좌표를 2D 스크린 좌표로 투영 (시점 동기화 수정)
            - 카메라가 '왼쪽'으로 이동하면, 이미지의 '왼쪽' 면이 더 크게(가깝게) 보여야 함.
            - 이를 위해 회전 각도의 부호를 반전(-self.h_angle)시켜 투영 계산
        """
        # 각도 부호 반전 (-)
        rad_h = math.radians(-self.h_angle) 
        rad_v = math.radians(self.v_angle)
        
        # Y축 회전 (Horizontal)
        x1 = x * math.cos(rad_h) - z * math.sin(rad_h)
        z1 = x * math.sin(rad_h) + z * math.cos(rad_h)
        
        # X축 회전 (Vertical)
        y2 = y * math.cos(rad_v) - z1 * math.sin(rad_v)
        z2 = y1 = y * math.sin(rad_v) + z1 * math.cos(rad_v)
        
        # 원근 투영 및 스케일 조정
        perspective = 400 / (400 + z2 * 50)
        px = cx + x1 * orbit_rx * perspective
        py = cy + y2 * orbit_rx * 0.8 * perspective 
        return px, py

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        painter.fillRect(self.rect(), self.col_bg)
        
        orbit_rx = min(w, h) * 0.35
        orbit_ry = orbit_rx * 0.35

        # 1. 바닥 그리드
        self._draw_floor_grid(painter, cx, cy, w, h)

        # 2. 이미지 투영
        target_pm = self.pixmap if self.pixmap else self.default_pixmap
        self._draw_projected_image(painter, target_pm, cx, cy, orbit_rx)

        # 3. 수평 궤도
        painter.setPen(QPen(self.col_orbit, 2, Qt.SolidLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPoint(int(cx), int(cy)), int(orbit_rx), int(orbit_ry))

        # 4. 수직 호 및 카메라 위치 계산
        # 슬라이더 방향과 화면 이동 방향 일치화, - 기존: h_angle + 90 (슬라이더 오른쪽 -> 화면 왼쪽 이동됨), - 변경: 90 - h_angle (슬라이더 오른쪽 -> 화면 오른쪽 이동됨)
        rad_h = math.radians(90 - self.h_angle)
        rad_v = math.radians(self.v_angle)
        
        # 카메라의 지표 투영점
        gx = cx + orbit_rx * math.cos(rad_h)
        gy = cy + orbit_ry * math.sin(rad_h)
        
        # 카메라의 3D 실제 투영 위치
        cam_x = cx + (orbit_rx * math.cos(rad_v)) * math.cos(rad_h)
        cam_y = cy + (orbit_ry * math.cos(rad_v)) * math.sin(rad_h) - (orbit_rx * 0.7 * math.sin(rad_v))

        # 수직 이동 경로 (Arc)
        if abs(self.v_angle) > 0.1:
            path_arc = QPainterPath()
            path_arc.moveTo(gx, gy)
            ctrl_x = gx + (cx - gx) * 0.2
            ctrl_y = (gy + cam_y) / 2
            path_arc.quadTo(ctrl_x, ctrl_y, cam_x, cam_y)
            
            painter.setPen(QPen(self.col_vert_arc, 3, Qt.DashLine))
            painter.drawPath(path_arc)
            painter.setBrush(self.col_vert_arc)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPoint(int(gx), int(gy)), 4, 4)

        # 5. 연결 빔
        painter.setPen(QPen(self.col_beam, 2))
        painter.drawLine(QPoint(int(cam_x), int(cam_y)), QPoint(int(cx), int(cy)))
        
        # 6. 카메라 아이콘
        self._draw_camera_icon(painter, cam_x, cam_y)
        painter.end()

    def _draw_projected_image(self, p, pixmap, cx, cy, orbit_rx):
        """ 이미지의 4개 모서리 좌표를 3D 투영하여 렌더링 (Zoom 배율 적용 수정)
            - orbit_rx 기준 크기에 self.zoom을 곱하여 시각적 확대/축소 반영
        """
        if not pixmap: return
        
        # 평면 모서리 정의 (중앙 기준)
        size = 1.0
        src_w, src_h = pixmap.width(), pixmap.height()
        aspect = src_h / src_w
        pts_3d = [(-size, -size*aspect, 0), (size, -size*aspect, 0), (size, size*aspect, 0), (-size, size*aspect, 0)]
        
        dest_pts = []
        # [수정] Zoom 값 반영: orbit_rx * 0.6 * self.zoom
        base_scale = orbit_rx * 0.6 * self.zoom
        
        for x, y, z in pts_3d:
            px, py = self._project_3d(x, y, z, cx, cy, base_scale)
            dest_pts.append([px, py])
            
        src_pts = np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]], dtype=np.float32)
        dst_pts = np.array(dest_pts, dtype=np.float32)

        try:
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            transform = QTransform(matrix[0,0], matrix[1,0], matrix[2,0], matrix[0,1], matrix[1,1], matrix[2,1], matrix[0,2], matrix[1,2], matrix[2,2])
            
            p.save()
            p.setTransform(transform, True)
            p.setOpacity(0.9)
            p.drawPixmap(0, 0, pixmap)
            p.restore()
            
            # 테두리
            poly = QPolygon([QPoint(int(pt[0]), int(pt[1])) for pt in dest_pts])
            p.setPen(QPen(QColor(0, 255, 255, 100), 1))
            p.drawPolygon(poly)
        except: pass

    def _draw_floor_grid(self, p, cx, cy, w, h):
        p.setPen(QPen(self.col_grid, 1))
        max_r = min(w, h) * 0.5
        for r in range(1, 4):
            rx = max_r * (r/3)
            ry = rx * 0.35
            p.drawEllipse(QPoint(int(cx), int(cy)), int(rx), int(ry))

    def _draw_camera_icon(self, p, x, y):
        p.save()
        p.translate(x, y)
        p.setBrush(self.col_camera)
        p.setPen(QPen(Qt.white, 1))
        p.drawRoundedRect(-12, -10, 24, 18, 4, 4)
        p.setBrush(QColor(255, 255, 255, 150))
        p.drawEllipse(QPoint(0, 0), 5, 5)
        p.restore()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = True
            self.last_pos = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            delta = event.position().toPoint() - self.last_pos
            self.last_pos = event.position().toPoint()
            self.h_angle += delta.x() * 0.5
            if self.h_angle > 180: self.h_angle -= 360
            elif self.h_angle < -180: self.h_angle += 360
            self.v_angle = max(-90.0, min(90.0, self.v_angle - delta.y() * 0.5))
            self.sig_angle_changed.emit(self.h_angle, self.v_angle)
            self.update()

    def mouseReleaseEvent(self, event):
        self.is_dragging = False
        self.setCursor(Qt.ArrowCursor)

# VisualCameraWidget (Updated: Balanced UI Layout)
class VisualCameraWidget(QWidget):
    """ 메인 카메라 컨트롤러
        - [초기화] [ - ] [컨트롤] [ + ] [값] 순서로 앞쪽 정렬 배치
        - 항목별 수직 위치 정렬 최적화
    """
    sig_prompt_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """ UI 구성 및 시그널 연결
            수평 제어 위젯을 QDial에서 QSlider로 변경하여 일관성을 높임.
        """
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10); layout.setSpacing(10)

        # 1. 헤더 (타이틀 및 전체 리셋)
        top = QHBoxLayout()
        lbl = QLabel("📷 Multiangle Camera")
        lbl.setStyleSheet("color: white; font-weight: bold; font-size: 13px;")
        
        btn_reset_all = QPushButton("Reset All")
        btn_reset_all.setMinimumWidth(70)  # 고정폭 대신 최소폭 사용
        btn_reset_all.setFixedHeight(24)
        btn_reset_all.setStyleSheet("background: #444; color: #eee; border-radius: 4px; font-size: 11px;")
        btn_reset_all.clicked.connect(self.reset_all_values)
        
        top.addWidget(lbl); top.addStretch(); top.addWidget(btn_reset_all)
        layout.addLayout(top)

        # 2. 비주얼라이저 (3D 궤도 뷰어)
        self.visualizer = OrbitCameraVisualizer(self)
        self.visualizer.sig_angle_changed.connect(self.sync_controls_from_visualizer)
        layout.addWidget(self.visualizer, 1)

        # 3. 컨트롤 패널 (그리드 레이아웃)
        ctrl_box = QWidget()
        ctrl_box.setStyleSheet("background: #252525; border-radius: 8px; padding: 10px;")
        g_layout = QGridLayout(ctrl_box)
        g_layout.setSpacing(12); g_layout.setColumnStretch(3, 1) # 슬라이더 영역 확장

        # (A) 수평 슬라이더 (기존 Dial에서 변경)
        self.slider_h = QSlider(Qt.Horizontal)
        self.slider_h.setRange(-180, 180)
        self.slider_h.setValue(0)
        self.slider_h.valueChanged.connect(self.sync_visualizer_from_controls)

        # (B) 수직 슬라이더
        self.slider_v = QSlider(Qt.Horizontal)
        self.slider_v.setRange(-90, 90)
        self.slider_v.setValue(0)
        self.slider_v.valueChanged.connect(self.sync_visualizer_from_controls)

        # (C) 줌 슬라이더
        self.slider_z = QSlider(Qt.Horizontal)
        self.slider_z.setRange(50, 200)
        self.slider_z.setValue(100)
        self.slider_z.valueChanged.connect(self.sync_visualizer_from_controls)

        # 값 표시 라벨 생성
        def create_val_lbl():
            l = QLabel("0")
            l.setStyleSheet("color:#00d2d3; font-weight:bold; min-width:40px;")
            l.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            return l

        self.lbl_h = create_val_lbl(); self.lbl_v = create_val_lbl(); self.lbl_z = create_val_lbl()

        # 그리드 배치 (Label, Reset, Minus, Widget, Plus, Value)
        # 수평(Pan)은 360도 순환을 위해 wrap=True 적용
        self._add_row_to_grid(g_layout, 0, "Horizontal", self.slider_h, self.lbl_h, 0, True)
        self._add_row_to_grid(g_layout, 1, "Vertical", self.slider_v, self.lbl_v, 0, False)
        self._add_row_to_grid(g_layout, 2, "Zoom", self.slider_z, self.lbl_z, 100, False)

        # Trigger Word 입력칸
        g_layout.addWidget(QLabel("Trigger"), 3, 0)
        self.txt_trigger = QLineEdit("img")
        self.txt_trigger.setStyleSheet("background: #333; color: white; border: 1px solid #555; padding: 4px;")
        self.txt_trigger.textChanged.connect(self.update_output_tag)
        g_layout.addWidget(self.txt_trigger, 3, 1, 1, 5)

        layout.addWidget(ctrl_box)

    def sync_controls_from_visualizer(self, h, v):
        """ 비주얼라이저의 드래그 결과값을 슬라이더에 반영
            드래그 시 발생하는 시그널을 수신하여 컨트롤러의 위치를 동기화함.
        """
        self.blockSignals(True)
        self.slider_h.setValue(int(h))
        self.slider_v.setValue(int(v))
        self.blockSignals(False)
        self.update_output_tag()

    def sync_visualizer_from_controls(self):
        """ 슬라이더 조작 값을 비주얼라이저와 출력 태그에 반영
            사용자가 슬라이더를 움직이면 3D 뷰의 카메라 각도를 즉시 업데이트함.
        """
        h = self.slider_h.value()
        v = self.slider_v.value()
        z = self.slider_z.value() / 100.0
        self.visualizer.set_values(h, v, z)
        self.update_output_tag()

    def update_output_tag(self):
        """ 현재 설정된 카메라 파라미터를 <camera> 태그 형식으로 생성
            값 라벨을 갱신하고 최종 프롬프트 파트를 부모 위젯으로 방출함.
        """
        h = self.slider_h.value()
        v = self.slider_v.value()
        z = self.slider_z.value() / 100.0
        self.lbl_h.setText(f"{h}°"); self.lbl_v.setText(f"{v}°"); self.lbl_z.setText(f"{z:.1f}")
        # UI 슬라이더 v값을 반전해서 태그 생성
        #   UI: +위쪽 = 사용자 직관상 "위에서 내려다봄(high-angle)"
        #   LoRA: vertical 양수 = high-angle  →  부호 반전 불필요하게 됨
        #   현재 _V_BINS: 양수 = high-angle (이미 올바름)
        #   문제는 슬라이더 +값이 태그에 그대로 넘어가는 것이므로, 여기서 반전
        tag = f"<camera>horizontal={h} vertical={-v} zoom={z:.2f}</camera>" # -v 로 변경
        self.sig_prompt_changed.emit(tag)

    def reset_all_values(self):
        """ 모든 카메라 파라미터를 초기 상태(정면, 1.0배)로 복구"""
        self.slider_h.setValue(0)
        self.slider_v.setValue(0)
        self.slider_z.setValue(100)

    # bg_composer.py의 _on_preview_translation_done에서 호출됨
    def get_prompt_part(self):
        """ 레거시 모델 또는 일반 텍스트 결합용 프롬프트 파트 반환 
            형식: "trigger horizontal=0 vertical=0 zoom=1.0"
        """
        h = self.slider_h.value()
        v = self.slider_v.value()
        z = self.slider_z.value() / 100.0
        trigger = self.txt_trigger.text().strip()
        
        part = f"horizontal={h} vertical={-v} zoom={z:.2f}" # -v로 변경
        return f"{trigger} {part}" if trigger else part

    # bg_composer.py의 run_generation (Safety Net)에서 호출됨
    def get_camera_tag_only(self):
        """ <camera> 태그 포맷만 반환 (트리거 제외)
            형식: "<camera>horizontal=0 vertical=0 zoom=1.0</camera>"
        """
        h = self.slider_h.value()
        v = self.slider_v.value()
        z = self.slider_z.value() / 100.0
        return f"<camera>horizontal={h} vertical={-v} zoom={z:.2f}</camera>" # -v로 변경

    def _step_val(self, widget, step, wrap):
        """ 슬라이더 값 증감 처리 (버튼 연동)
            - widget: 대상 QSlider
            - step: 증감 값 (+1 또는 -1)
            - wrap: 값 순환 여부 (True일 경우 Min/Max 연결)
        """
        val = widget.value() + step
        mn, mx = widget.minimum(), widget.maximum()

        if wrap:
            if val > mx:
                val = mn
            elif val < mn:
                val = mx
        else:
            val = max(mn, min(mx, val))

        widget.setValue(val)
        
    def set_thumbnail(self, img_np):
        """ 비주얼라이저 내부에 표시될 썸네일 이미지를 설정 """
        self.visualizer.set_thumbnail(img_np)

    def _add_row_to_grid(self, grid, row, label, widget, val_lbl, reset_val, wrap):
        """ 그리드에 각 카메라 컨트롤 행을 추가 (UI 개선)
            - Reset 버튼 너비 확장 (45 -> 52)
            - 폰트 크기 및 색상 변경 (가독성 향상)
        """
        lbl = QLabel(label)
        lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        grid.addWidget(lbl, row, 0)

        # 개별 리셋 버튼 (스타일 개선)
        btn_reset = QPushButton("reset") 
        # 높이를 -> 26으로 약간 늘려 텍스트 공간 확보
        btn_reset.setMinimumWidth(50)  # 글자가 길어지면 버튼도 같이 늘어나게 변경
        btn_reset.setFixedHeight(26)
        
        btn_reset.setCursor(Qt.PointingHandCursor)
        
        # padding: 0px 추가 (내부 여백 초기화로 텍스트 중앙 정렬 유도)
        btn_reset.setStyleSheet("""
            QPushButton { 
                background: #333; 
                color: #bbb; 
                border: 1px solid #444; 
                border-radius: 4px; 
                font-size: 11px; 
                font-weight: bold;
                padding: 0px; 
            }
            QPushButton:hover { background: #444; color: #fff; }
        """)
        
        btn_reset.clicked.connect(lambda: widget.setValue(reset_val))
        grid.addWidget(btn_reset, row, 1)

        # 마이너스 버튼
        btn_min = ArrowButton("left")
        btn_min.clicked.connect(lambda: self._step_val(widget, -1, wrap))
        grid.addWidget(btn_min, row, 2)

        # 메인 위젯 (Slider)
        grid.addWidget(widget, row, 3)

        # 플러스 버튼
        btn_plus = ArrowButton("right")
        btn_plus.clicked.connect(lambda: self._step_val(widget, 1, wrap))
        grid.addWidget(btn_plus, row, 4)

        # 값 라벨
        grid.addWidget(val_lbl, row, 5)