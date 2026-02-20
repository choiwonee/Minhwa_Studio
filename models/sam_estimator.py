import os, sys
import numpy as np
from pathlib import Path

# [Config] 설정 로더 가져오기
try:
    from utils.config_loader import config
except ImportError:
    config = None

try:
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    print(f"[SAM2Estimator] Library missing: {e}")
    torch = None
    SAM2ImagePredictor = None

class SAM2Estimator:
    def __init__(self):
        """ SAM2Estimator 초기화
            - 프로젝트 루트 기준 절대 경로를 강제하여 소스 폴더 내 관리를 보장함 """
        self.predictor = None
        self.is_ready = False
        self.is_image_set = False
        
        # 1) 통합 경로 설정 (절대 경로화 로직 적용)
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
            
        # 2) SAM2 폴더 생성
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.hw_info = self.detect_hardware_capabilities()
        self._prepare_device_and_precision()
        
        print(f"[SAM2Estimator] Integrated Weights Path: {self.base_dir}")
        
    def detect_hardware_capabilities(self):
        """ 시스템의 GPU 종류(NVIDIA vs Intel vs CPU) 감지 """
        info = {"device": "cpu", "vram_gb": 0, "is_pascal": False, "description": "CPU Mode"}
        
        if torch and torch.cuda.is_available():
            info["device"] = "cuda"
            try:
                p = torch.cuda.get_device_properties(0)
                info["vram_gb"] = round(p.total_memory / (1024**3), 2)
                info["is_pascal"] = (p.major < 7)
                arch_type = 'Pascal (Legacy)' if info["is_pascal"] else 'Modern'
                info["description"] = f"NVIDIA {p.name} ({info['vram_gb']}GB) - {arch_type}"
            except: pass
            
        elif torch and hasattr(torch, 'xpu') and torch.xpu.is_available():
            info["device"] = "xpu"
            try:
                total_mem = torch.xpu.get_device_properties(0).total_memory
                info["vram_gb"] = round(total_mem / (1024**3), 2)
                info["description"] = f"Intel Arc/XPU ({info['vram_gb']}GB)"
            except:
                info["description"] = "Intel Arc/XPU Graphics"
                
        return info

    def _prepare_device_and_precision(self):
        """ 장치 및 기본 정밀도 설정 """
        if not torch:
            self.device = "cpu"
            self.dtype = None
            return

        dev_type = self.hw_info["device"]
        
        if dev_type == "cuda":
            self.device = "cuda"
            # Pascal 아키텍처는 FP16 성능 이슈로 FP32 권장
            if self.hw_info["is_pascal"]:
                self.dtype = torch.float32
            else:
                # Ampere 이상 등에서는 BF16/FP16 사용
                if torch.cuda.is_bf16_supported():
                    self.dtype = torch.bfloat16
                else:
                    self.dtype = torch.float16
            # CUDA 최적화 설정
            torch.backends.cudnn.benchmark = True
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
        elif dev_type == "xpu":
            # 🚨 [수정됨] Intel XPU 치명적 메모리 오류 방지를 위해 강제로 CPU 할당
            print("[SAM2Estimator] Intel XPU detected, but forcing CPU mode for stability.")
            self.device = "cpu"
            self.dtype = torch.float32 
            
        else:
            self.device = "cpu"
            self.dtype = torch.float32

    def load_model(self, model_id: str, precision="auto") -> bool:
        """ SAM2 모델 로드 (Precision 인자 및 통합 경로 적용)
            - 모델 로딩 시 정밀도 설정을 반영하고, 지정된 통합 경로에 가중치를 저장함 """
        if not torch or not SAM2ImagePredictor:
            raise ImportError("PyTorch or SAM2 library is not installed.")
        
        # 모델을 새로 로딩하면 predictor 인스턴스가 교체되므로 상태 리셋 필수
        self.is_image_set = False
        
        # 1) 정밀도(Dtype) 결정 로직 (기존 로직 유지)
        target_dtype = self.dtype
        if precision == "float32": 
            target_dtype = torch.float32
        elif precision == "float16": 
            target_dtype = torch.float16
        elif precision == "bfloat16": 
            target_dtype = torch.bfloat16

        try:
            print(f"[SAM2Estimator] Loading model '{model_id}' on {self.device} with {target_dtype}...")
            
            # 2) 통합 경로(self.base_dir (./models)를 cache_dir로 전달) 및 Dtype을 적용하여 로드
            self.predictor = SAM2ImagePredictor.from_pretrained(
                model_id, 
                device=self.device, 
                dtype=target_dtype,
                cache_dir=str(self.base_dir) # 통합 경로 주입
            )
            
            # 3) XPU 전용 IPEX 최적화 로직
            if self.device == "xpu" and "intel_extension_for_pytorch" in sys.modules:
                try:
                    import intel_extension_for_pytorch as ipex
                    # 메모리 생성 오류 방지를 위해 weights_prepack=False 강제 지정
                    self.predictor.model = ipex.optimize(self.predictor.model, dtype=target_dtype, weights_prepack=False)
                    print("[SAM2Estimator] IPEX optimization applied for XPU (weights_prepack=False).")
                except Exception as e:
                    print(f"[SAM2Estimator] IPEX optimize warning: {e}")

            # 4) 상태 업데이트
            self.dtype = target_dtype 
            self.is_ready = True
            self.is_image_set = False # 로드 직후엔 이미지 임베딩이 비어있음
            
            print(f"[SAM2Estimator] Model saved/loaded at: {self.base_dir}")
            return True
            
        except Exception as e:
            print(f"[SAM2Estimator] Load Error: {e}")
            self.is_ready = False
            self.is_image_set = False
            raise e

    def set_image(self, image: np.ndarray):
        """ 이미지를 모델에 입력하여 임베딩을 계산한다. """
        if not self.is_ready or self.predictor is None:
            raise RuntimeError("Model is not loaded.")
        
        # [추가된 부분] 이미지 분석 전 메모리 찌꺼기 싹 비우기
        import gc
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "xpu":
            torch.xpu.empty_cache()

        try:
            with torch.inference_mode():
                # 장치별 Autocast 분기 처리
                if self.device == "cuda":
                    # CUDA는 autocast 사용 (dtype에 맞춰)
                    with torch.autocast("cuda", dtype=self.dtype):
                        self.predictor.set_image(image)
                elif self.device == "xpu":
                    # XPU도 autocast 지원 (Intel PyTorch)
                    with torch.autocast("xpu", dtype=self.dtype):
                        self.predictor.set_image(image)
                else:
                    # CPU는 autocast 없이 진행
                    self.predictor.set_image(image)
                    
            self.is_image_set = True
        except Exception as e:
            self.is_image_set = False
            raise e

    def predict(self, predict_kwargs: dict):
        """ 마스크 추론을 수행한다. """
        if not self.is_ready or not self.is_image_set:
            raise RuntimeError("Model not loaded or image not set.")

        with torch.inference_mode():
            # 장치별 Autocast 분기 처리
            if self.device == "cuda":
                with torch.autocast("cuda", dtype=self.dtype):
                    masks, scores, logits = self.predictor.predict(**predict_kwargs)
            elif self.device == "xpu":
                with torch.autocast("xpu", dtype=self.dtype):
                    masks, scores, logits = self.predictor.predict(**predict_kwargs)
            else:
                masks, scores, logits = self.predictor.predict(**predict_kwargs)
        
        # 결과 가공
        mask_out = None
        multimask = predict_kwargs.get("multimask_output", False)
        
        if multimask:
            return masks, scores # (N, H, W), (N,)
        else:
            if isinstance(masks, np.ndarray) and masks.ndim == 3:
                mask_out = masks[0]
            else:
                mask_out = masks
            return (mask_out > 0).astype(np.uint8) * 255