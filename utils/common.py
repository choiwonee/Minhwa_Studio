import numpy as np
import cv2
import os

from datetime import datetime
from pathlib import Path
from PIL import Image

# 현재 사용 중인 모델의 줄임말 정의
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

# 디렉토리 생성 함수 (한글/유니코드 경로도 지원)
def ensure_dir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)


def imwrite_unicode(path, img, params=None):
    """한글/유니코드 파일명을 지원하여 이미지 저장"""
    ext = os.path.splitext(path)[1]
    
    # 16비트 이미지이면 PNG 무압축으로 저장
    if img.dtype == np.uint16:
        params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
    
    # 4채널(RGBA) 이미지 처리 시 압축 옵션 지정
    if len(img.shape) == 3 and img.shape[2] == 4:
        params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
    
    result, buf = cv2.imencode(ext, img, params or [])
    if result:
        try:
            buf.tofile(path)
            return True
        except IOError:
            print(f"Error: Could not save file to {path}")
            return False
    return False
    

def load_image_numpy(path: str | Path, scale=1.0) -> np.ndarray:
    """이미지 파일을 불러와 uint8 NumPy 배열로 반환 (RGB로 변환, 필요 시 리사이즈)"""
    img = Image.open(path).convert("RGB")
    if scale != 1.0:
        w, h = img.size
        img  = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    return np.array(img)  # 0~255 범위 uint8 배열


def save_unit16_png(arr01: np.ndarray,
                    output_dir: Path,
                    input_path: str | Path,
                    model_id: str = "depth-anything/Depth-Anything-V2-Large-hf",
                    suffix: str = "Untitled") -> None:
    """
    0~1 범위 float 배열을 0~65535 uint16로 변환하여 16비트 PNG로 저장
    (깊이맵, 마스크 등 고정밀 이미지용)
    """
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
    """
    0~1 범위 float RGB 배열을 0~255 uint8로 변환하여 PNG로 저장
    (시각화용 RGB 이미지)
    """
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


def normalize_np_img_array(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    NumPy 배열을 0~1 범위로 정규화
    (딥러닝 모델 입력이나 이미지 처리 전 표준화용)
    """
    x = x.astype(np.float32)
    mn, mx = np.min(x), np.max(x)
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + eps)  # 분모에 eps를 더해 0으로 나누는 오류 방지
