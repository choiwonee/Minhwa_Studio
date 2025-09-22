import numpy as np
import torch
import utils.common as COM

from transformers import AutoImageProcessor, AutoModelForDepthEstimation

def perform_depth_anything_modeler(rgb_img_numpy: np.ndarray, 
                                   model_id: str = "depth-anything/Depth-Anything-V2-Large-hf", 
                                   device: str = "cuda") -> np.ndarray:
    """
    'depth-anything' 모델 기반으로 깊이맵 추론 후
    0~1 범위로 정규화된 깊이맵을 반환하는 함수.

    Parameters
    ----------
    rgb_img_numpy : np.ndarray
        입력 RGB 이미지 (H, W, 3), dtype=np.uint8 또는 float32
    model_id : str, default="depth-anything/Depth-Anything-V2-Large-hf"
        Hugging Face 모델 ID
    device : str, default="cuda"
        모델 연산 디바이스 ("cuda" 또는 "cpu")

    Returns
    -------
    np.ndarray
        정규화된 깊이맵, shape=(H, W), dtype=float32, 범위 0~1
    """

    # --- (1) 정보 출력 ---
    print(f"[INFO] model_id: {model_id}, torch.cuda: {device}")
    print(f"[INFO] numpy size: {rgb_img_numpy.shape}, numpy type: {rgb_img_numpy.dtype}")

    # --- (2) 입력 이미지 크기 추출 ---
    h, w = rgb_img_numpy.shape[:2]

    # --- (3) 모델과 프로세서 불러오기 ---
    processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf", use_fast=True)
    model     = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)

    import torch.nn.functional as Func

    # --- (4) 깊이 추론 ---
    with torch.inference_mode():
        # 이미지 전처리 → tensor 변환 및 device 이동
        inputs = processor(images=rgb_img_numpy, return_tensors="pt").to(device)
        pred   = model(**inputs).predicted_depth      # 모델 예측 깊이

        # --- (4-1) 원본 이미지 크기에 맞춰 보간 ---
        pred_resized = Func.interpolate(
            pred.unsqueeze(1),    # (1, 1, H, W) → batch + channel 추가
            size=(h, w),          # 원본 크기로 리사이즈
            mode="bilinear",      # bilinear 보간
            align_corners=False
        ).squeeze()              # (H, W)로 차원 축소

    # --- (5) numpy 변환 및 정규화 ---
    depth = pred_resized.cpu().numpy().astype(np.float32)
    depth = COM.normalize_np_img_array(depth)  # 최종 0~1 범위 정규화

    # --- (6) 결과 반환 ---
    return depth
