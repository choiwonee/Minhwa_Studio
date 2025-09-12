import numpy as np

from datetime import datetime
from pathlib import Path
from PIL import Image

# 파일 저장 경로
outDir = None

MODEL_SHORT_NAMES = {
    "depth-anything/Depth-Anything-V2-Large-hf": "DAv2L",
    "depth-anything/Depth-Anything-V2-Small-hf": "DAv2S",
    "Intel/dpt-hybrid-midas": "DPTm",
    "Intel/dpt-large": "DPTL",
    "Intel/zoedepth-nyu": "ZoeNYU",
    "Intel/zoedepth-kitti": "ZoeKITTI"
}

def generateDepthMapFilename(model_id: str, input_path: str | Path) -> str:
    # 모델 약칭 + 입력 파일명 + timestamp를 포함한 깊이맵 PNG 파일명 생성
    input_stem = Path(input_path).stem
    model_short = MODEL_SHORT_NAMES.get(model_id, model_id.split("/")[-1])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_short}_{input_stem}_DepthMap_{timestamp}.png"


def ensureDir(savePath: str | Path) -> None:
    global outDir
    Path(savePath).mkdir(parents=True, exist_ok=True)
    outDir = savePath
    

def loadImageNumpy(path: str | Path, scale=1.0) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if scale != 1.0:
        w, h = img.size
        img  = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return np.array(img)


def saveUint16Png(path: str | Path, arr01: np.ndarray) -> None:
    arr = np.clip(arr01, 0, 1)
    arr_u16 = (arr * 65535.0 + 0.5).astype(np.uint16)
    # cv2.imwrite(str(path), arr_u16)
    img = Image.fromarray(arr_u16, mode="I;16")
    img.save(str(path))


# NumPy 배열을 0~1 범위로 정규화
def normalizeNpImgArray(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = np.min(x), np.max(x)
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + eps)
