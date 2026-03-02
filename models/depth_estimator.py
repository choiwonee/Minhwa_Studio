import os
import numpy as np
import torch
import torch.nn.functional as Func
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import utils.common as COM
from pathlib import Path

# [Config] 설정 로더 가져오기
try:
    from utils.config_loader import config
except ImportError:
    config = None

class DepthEstimator:
    def __init__(self):
        """ DepthEstimator 초기화
            - config.ini의 weights_path가 상대 경로일 경우 프로젝트 루트 기준 절대 경로로 변환 """
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_model_id = None
        
        # 1) 통합 경로 설정 (소스 파일 위치 기준 절대 경로화)
        if config and getattr(config, "weights_path", None):
            raw_path = config.weights_path
            if not os.path.isabs(raw_path):
                # models 폴더의 상위인 프로젝트 루트 확보
                project_root = Path(__file__).parent.parent.absolute()
                self.base_dir = (project_root / raw_path.replace("./", "")).resolve()
            else:
                self.base_dir = Path(raw_path)
        else:
            self.base_dir = Path.home() / ".cache" / "minhwa"
            
        # 2) 폴더 생성 및 로그 출력
        self.base_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DepthEstimator] Integrated Weights Path: {self.base_dir}")

    def load_model(self, model_id: str):
        """ 모델 로딩 메서드 (전체 코드 생략 없이 유지) """
        if self.model is not None and self.current_model_id == model_id:
            return

        print(f"[DepthEstimator] Loading model: {model_id}...")
        
        load_args = {
            "cache_dir": str(self.base_dir)
        }

        try:
            self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True, **load_args)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_id, **load_args).to(self.device)
            self.current_model_id = model_id
            
            if self.device == "cuda":
                self.model.half()
                
            print(f"[DepthEstimator] Model saved/loaded at: {self.base_dir}")
        except Exception as e:
            print(f"[DepthEstimator] Error loading model: {e}")
            raise e
        
    def estimate_depth(self, rgb_img_numpy: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        h, w = rgb_img_numpy.shape[:2]

        with torch.inference_mode():
            inputs = self.processor(images=rgb_img_numpy, return_tensors="pt").to(self.device)
            
            # FP16 사용 시 입력 데이터도 변환
            if self.device == "cuda":
                inputs["pixel_values"] = inputs["pixel_values"].half()

            pred = self.model(**inputs).predicted_depth

            # 원본 크기로 보간
            pred_resized = Func.interpolate(
                pred.unsqueeze(1),
                size=(h, w),
                mode="bilinear",
                align_corners=False
            ).squeeze()

        depth = pred_resized.cpu().numpy().astype(np.float32)
        return COM.normalize_np_img_array(depth)

# 전역 인스턴스 생성 (어디서든 import하여 사용 가능)
depth_estimator = DepthEstimator()