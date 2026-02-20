# utils/common.py
import os, sys

import cv2
import numpy as np

from datetime import datetime
from pathlib import Path
from PIL import Image
from PySide6.QtGui import QImage

# C 레벨의 stderr(콘솔 에러) 출력을 잠시 막는 컨텍스트 매니저
class SuppressStderr:
    def __init__(self):
        self.null_fds = None
        self.save_fds = None

    def __enter__(self):
        try:
            # 1. 표준 에러 출력(stderr)의 파일 디스크립터를 복제해둠
            self.save_fds = os.dup(2)
            # 2. null 장치(윈도우는 NUL, 리눅스는 /dev/null)를 엶
            self.null_fds = os.open(os.devnull, os.O_RDWR)
            # 3. 표준 에러 출력을 null 장치로 돌려버림 (출력 무시)
            os.dup2(self.null_fds, 2)
        except Exception:
            pass # 권한 문제 등으로 실패하면 그냥 둠

    def __exit__(self, *_):
        try:
            # 4. 원래대로 표준 에러 출력 복구
            if self.save_fds is not None:
                os.dup2(self.save_fds, 2)
                os.close(self.save_fds)
            if self.null_fds is not None:
                os.close(self.null_fds)
        except Exception:
            pass

# 모델의 줄임말 정의
MODEL_SHORT_NAMES = {
    "depth-anything/Depth-Anything-V2-Large-hf": "DAv2L",
    "prs-eth/marigold-depth-hr-v1-1": "MGHR",
    "prs-eth/marigold-depth-v1-1": "MGv1",
    "prs-eth/marigold-depth-v1-0": "MGv0",
    "Intel/dpt-hybrid-midas": "DPT_M",
    "Intel/dpt-large": "DPT_L",
    "Intel/zoedepth-nyu": "Zoe_NYU",
    "Intel/zoedepth-nyu-kitti": "Zoe_NK",
    "Intel/zoedepth-kitti": "Zoe_K"
}

# 파일 기본 저장 경로
DEFAULT_SAVE_PATH: Path = Path("./output")

# 디렉토리 생성 함수 (한글/유니코드 경로도 지원)
def ensure_dir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)

def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """ Unicode-compatible image reading function """
    try:
        if not os.path.exists(path):
            print(f"### File not found: {path}")
            return None
        
        # C 레벨 경고 끄기
        with SuppressStderr():
            return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    except Exception as e:
        print(f"### [ERROR] reading image: {e}")
        return None

def imwrite_unicode(path, img, params=None):
    """ 한글/유니코드 파일명을 지원하여 이미지 저장(알파 채널 처리 명확화) """
    ext = os.path.splitext(path)[1].lower()
    
    # 4채널(RGBA) 이미지 처리시 압축 옵션 지정
    if len(img.shape) == 3 and img.shape[2] == 4 and ext == '.png':
        # PIL 사용하여 BGRA 이미지를 PNG로 저장하는 것이 더 안정적임
        try:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
            pil_img.save(path)
            return True
        except Exception as e:
            print(f"### [ERROR] saving BGRA image with PIL: {e}")
            # PIL 처리 실패시, 기존 OpenCV 방법으로 시도
            result, buf = cv2.imencode(ext, img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if result:
                buf.tofile(path)
                return True
    
    # 그 외 타 이미지 타입이면 기본 조건 사용
    result, buf = cv2.imencode(ext, img, params or [])
    if result:
        try:
            buf.tofile(path)
            return True
        except IOError:
            print(f"### [ERROR] Could not save file to {path}")
            return False
    return False

def load_image_numpy(path: str | Path, scale=1.0) -> np.ndarray:
    """
    이미지 파일을 불러와 uint8 NumPy 배열로 반환 (RGB로 변환, 필요 시 리사이즈)
    - scale: 이미지 크기 조절 비율 (기본값 1.0)
    """
    try:
        # 경로 객체 처리 및 파일 존재 여부 확인
        if not os.path.exists(str(path)):
            print(f"### [ERROR] File not found: {path}")
            return None

        # Pillow Image.open 시 발생하는 경고 차단
        with SuppressStderr():
            img = Image.open(path).convert("RGB") # RGB 변환
            
        # 스케일 옵션이 1.0이 아닐 경우 리사이즈 수행
        if scale != 1.0:
            w, h = img.size
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            # LANCZOS 필터를 사용하여 고품질 리사이즈
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        return np.array(img)  # 0~255 범위 uint8 배열

    except Exception as e:
        print(f"### [ERROR] Failed to load image: {e}")
        return None

def save_unit16_png(arr01: np.ndarray,
                    output_dir: Path,
                    input_path: str | Path,
                    model_id: str = "depth-anything/Depth-Anything-V2-Large-hf",
                    suffix: str = "Untitled") -> None:
    """ 0~1 범위 float 배열을 0~65535 uint16로 변환하여 16비트 PNG로 저장(깊이맵, 마스크 등 고정밀 이미지용)"""
    input_stem = Path(input_path).stem
    model_short = MODEL_SHORT_NAMES.get(model_id, model_id.split("/")[-1])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = f"{model_short}_{input_stem}_{suffix}_{timestamp}.png"

    arr = np.clip(arr01, 0, 1)
    arr_u16 = (arr * 65535.0 + 0.5).astype(np.uint16)

    save_path = output_dir / filename
    success = imwrite_unicode(str(save_path), arr_u16)

    print("Saving to :", output_dir / filename)
    if not success:
        print("Failed to save file!")

def save_rgb_png(rgb01: np.ndarray,
                 output_dir: Path,
                 input_path: str | Path,
                 model_id: str = "depth-anything/Depth-Anything-V2-Large-hf",
                 suffix: str = "Untitled") -> None:
    """ 0~1 범위 float RGB 배열을 0~255 uint8로 변환하여 PNG로 저장(시각화용 RGB 이미지) """
    input_stem = Path(input_path).stem
    model_short = MODEL_SHORT_NAMES.get(model_id, model_id.split("/")[-1])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_short}_{input_stem}_{suffix}_{timestamp}.png"
    
    arr = np.clip(rgb01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    
    # RGB → BGR 변환 (OpenCV 저장 시 색상 순서 맞춤)
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    save_path = output_dir / filename
    success = imwrite_unicode(str(save_path), arr_bgr)
    
    print("Saving to :", output_dir / filename)
    if not success:
        print("Failed to save file!")

def normalize_np_img_array(x: np.ndarray, eps: float=1e-9) -> np.ndarray:
    """ NumPy 배열을 0~1 범위로 정규화 (딥러닝 모델 입력이나 이미지 처리 전 표준화용) """
    x = x.astype(np.float32)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    
    x = np.nan_to_num(x, nan=mn) # NaN 값 처리: 정규화 전에 NaN 값을 0으로 대체 (이미지 경계 등)
    return (x - mn) / (mx - mn + eps) # 분모에 eps를 더해 0으로 나누는 오류 방지

def get_project_root():
    """ 프로젝트의 최상위 루트 경로를 반환. EXE로 패키징된 경우 실행 파일의 위치를, 그렇지 않으면 스크립트 위치를 기준 잡는다. """
    if getattr(sys, 'frozen', False):
        # PyInstaller 등으로 패키징된 경우 (.exe 위치)
        return os.path.dirname(sys.executable)
    else:
        # 개발 환경 (utils 폴더의 상위 폴더)
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_output_dir(sub_folder_name=None):
    """ outputs 폴더 경로를 반환하고, 없으면 생성. sub_folder_name이 있으면 그 하위 폴더까지 생성한다."""
    root = get_project_root()
    out_path = os.path.join(root, 'outputs')
    
    if sub_folder_name:
        out_path = os.path.join(out_path, sub_folder_name)
        
    os.makedirs(out_path, exist_ok=True)
    return out_path

def qimage_from_ndarray(img: np.ndarray) -> QImage:
    """ NumPy 배열을 QImage로 변환 """
    if img is None: return QImage()
    h, w = img.shape[:2]
    if img.ndim == 2:
        fmt = QImage.Format_Grayscale8
        return QImage(img.data, w, h, img.strides[0], fmt).copy()
    if img.shape[2] == 3:
        return QImage(img.data, w, h, img.strides[0], QImage.Format_RGB888).copy()
    if img.shape[2] == 4:
        return QImage(img.data, w, h, img.strides[0], QImage.Format_RGBA8888).copy()
    return QImage()

def save_image_file(img_np: np.ndarray, filename: str, sub_folder: str = "common", force_png: bool = True):
    """ [Sanitization Save] 이미지 데이터를 '세탁'하여 저장한다.
            1. 메타데이터/Exif 완전 제거 (새 이미지 인스턴스 생성)
            2. RGBA(투명) 이미지의 경우 강제로 .png 확장자 적용
            3. RGB/BGR 색상 공간 자동 보정
    """
    out_dir = get_output_dir(sub_folder)
    
    # 1. 확장자 강제 보정 (투명도가 있거나 force_png가 True면 무조건 png)
    name_stem = os.path.splitext(filename)[0]
    extension = os.path.splitext(filename)[1].lower()
    
    # 4채널(RGBA)이거나 마스크(Binary)인 경우 PNG 강제
    is_rgba = (img_np.ndim == 3 and img_np.shape[2] == 4)
    if is_rgba or force_png or extension not in ['.jpg', '.jpeg', '.bmp', '.tiff']:
        target_filename = f"{name_stem}.png"
        save_format = "PNG"
    else:
        target_filename = filename
        save_format = None # 확장자에 따름

    full_path = os.path.join(out_dir, target_filename)

    # 2. 데이터 정규화 (0~1 Float -> 0~255 Uint8)
    if img_np.dtype == np.float32 or img_np.dtype == np.float64:
        img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    elif img_np.dtype != np.uint8:
        img_np = img_np.astype(np.uint8)

    try:
        # 3. PIL Image 변환 (데이터 세탁 핵심), 단순히 fromarray만 하지 않고, 새 이미지를 생성하여 데이터를 복사하는 방식이 가장 안전함
        h, w = img_np.shape[:2]
        
        if img_np.ndim == 2: # Grayscale (Mask)
            mode = "L"
            clean_img = Image.fromarray(img_np, mode=mode)
        elif img_np.shape[2] == 3: # RGB
            mode = "RGB"
            # 입력이 BGR(OpenCV)인지 RGB(PIL)인지 확인 필요하지만, masking.py는 기본적으로 RGB로 로드하므로 그대로 진행
            clean_img = Image.fromarray(img_np, mode=mode)
        elif img_np.shape[2] == 4: # RGBA
            mode = "RGBA"
            clean_img = Image.fromarray(img_np, mode=mode)
        else:
            print(f"[Sanitize] Unsupported dimensions: {img_np.shape}")
            return None

        # 4. 메타데이터 없이 순수 픽셀 데이터만 저장, optimize=False 주어 불필요한 압축 과정에서의 헤더 변형 방지
        clean_img.save(full_path, format=save_format)
        
        print(f"[Sanitize] Saved clean image to: {full_path}")
        return full_path

    except Exception as e:
        print(f"[Sanitize] Save Error: {e}")
        return None