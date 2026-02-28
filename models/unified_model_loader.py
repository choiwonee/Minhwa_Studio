import os
import threading
from pathlib import Path
from typing import Optional, Dict, Any

# Windows Symlink Permission Fix
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import torch
from huggingface_hub import hf_hub_download

# Diffusers & Transformers (필요한 라이브러리만 남김)
from diffusers import (
    AutoPipelineForInpainting,
    QwenImageEditPipeline,
    QwenImageEditPlusPipeline,
)

try:
    from utils.config_loader import config
except ImportError:
    config = None

# -----------------------------------------------------------------------------
# Unified Model Loader 클래스
# -----------------------------------------------------------------------------
class UnifiedModelLoader:
    def __init__(self, config=None, precision="auto", quantization="auto", device=None, hw_info=None, hf_token=None, **kwargs):
        """ 통합 모델 로더 초기화 """
        self.device = device
        self.hw_info = hw_info or {}
        self.hf_token = hf_token
        
        if config and hasattr(config, "weights_dir"):
            self.base_dir = config.weights_dir
        else:
            project_root = Path(__file__).parent.parent.absolute()
            self.base_dir = project_root / "models" / "weights"
            
        self.base_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Loader] Final Weights Path: {self.base_dir}")
        
        self.lock = threading.Lock()

    def _get_load_kwargs(self) -> Dict[str, Any]:
        """ 공통 로딩 인자 생성 """
        kwargs = {
            "cache_dir": str(self.base_dir),
        }
        if self.hf_token:
            kwargs["token"] = self.hf_token
        return kwargs
    
    def _get_actual_path(self, model_id: str) -> str:
        """ [Path Optimization] 실제 스냅샷 경로 계산 """
        folder_name = f"models--{model_id.replace('/', '--')}"
        model_path = os.path.join(self.base_dir, folder_name)
        
        if os.path.exists(model_path):
            snapshot_dir = os.path.join(model_path, "snapshots")
            if os.path.exists(snapshot_dir):
                subdirs = os.listdir(snapshot_dir)
                if subdirs:
                    actual_path = os.path.join(snapshot_dir, subdirs[0])
                    print(f"[Loader] 📍 Local snapshot found: {actual_path}")
                    return actual_path
        
        return model_id
    
    def _download_model_file(self, repo_id: str, filename: str, abort_check, subfolder: Optional[str] = None) -> Path:
        print(f"[Loader] Checking/Downloading {filename}...")
        return Path(hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder,cache_dir=str(self.base_dir), token=self.hf_token))

    # -------------------------------------------------------------------------
    # Qwen Helper Methods
    # -------------------------------------------------------------------------
    def _install_qwen_image_processor_5d_fix(self, pipeline):
        """ Qwen Image Processor 5D 텐서 문제 해결 """
        if not hasattr(pipeline, "image_processor"):
            return

        processor = pipeline.image_processor
        if hasattr(processor, "config"):
            processor.config.do_resize = False
            processor.config.do_center_crop = False

        if hasattr(pipeline, "vae"):
            orig_vae_encode = pipeline.vae.encode
            def safe_vae_encode(x, *args, **kwargs):
                if isinstance(x, torch.Tensor) and x.dim() == 5:
                    x = x.squeeze(2)
                return orig_vae_encode(x.to(torch.float32), *args, **kwargs)
            pipeline.vae.encode = safe_vae_encode

        print("[Loader] 🛡️ Qwen Processor Resizer Locked & 4D-Unpack Guard Active")

    # -------------------------------------------------------------------------
    # Qwen Native Loaders
    # -------------------------------------------------------------------------
    def _load_qwen_edit_native(self, model_id, model_config, abort_check):
        """ QwenImageEdit 로컬 모델 로딩 """
        if abort_check and abort_check(): return None, None
        
        is_pascal = self.hw_info.get("is_pascal", False)
        vram_gb = self.hw_info.get("vram_gb", 0)
        
        if is_pascal or vram_gb <= 8.5:
            raise RuntimeError("Qwen-Edit Local requires RTX 20 series or higher with 12GB+ VRAM.")

        actual_load_path = self._get_actual_path(model_id)
        try:
            print(f"[Loader] 🏗️ Loading Qwen-Edit (High-Spec Mode)...")
            pipeline = QwenImageEditPipeline.from_pretrained(
                actual_load_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                **self._get_load_kwargs()
            )

            self._install_qwen_image_processor_5d_fix(pipeline)
            pipeline.to(self.device)
            return pipeline, model_config

        except Exception as e:
            print(f"[Loader] Qwen Load Failed: {e}")
            raise e

    def _load_qwen_edit_2511_native(self, model_id, model_config, abort_check):
        """ Qwen-Edit-2511 Plus 로더 """
        if abort_check and abort_check(): return None, None

        if self.hw_info.get("is_pascal", False) or self.hw_info.get("vram_gb", 0) <= 8.5:
            raise RuntimeError("Qwen-Edit 2511 requires RTX 20 series or higher with 12GB+ VRAM.")

        actual_load_path = self._get_actual_path(model_id)
        try:
            print(f"[Loader] 🏗️ Loading Qwen-Edit-2511 Plus (BFloat16)...")
            pipeline = QwenImageEditPlusPipeline.from_pretrained(
                actual_load_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                **self._get_load_kwargs()
            )
            pipeline.to(self.device)
            return pipeline, model_config
        except Exception as e:
            raise e

    # -------------------------------------------------------------------------
    # Standard Pipeline Loader (DreamShaper 등)
    # -------------------------------------------------------------------------
    def _load_standard_pipeline(self, model_id, p_type, precision, quantization, abort_check, model_config=None):
        """ 표준 파이프라인 로더 (SD 1.5 Inpainting 등) """
        print(f"[Loader] Loading Standard Pipeline: {p_type}...")
        
        is_pascal = self.hw_info.get("is_pascal", False)
        # Intel XPU에서 메모리 오류가 계속된다면 강제로 float32 사용
        if self.device == "xpu":
            dtype = torch.float32
        else:
            dtype = torch.float16 if is_pascal else torch.bfloat16
        self._get_actual_path(model_id)
        
        load_args = {
            "torch_dtype": dtype, 
            "use_fast": True, 
            **self._get_load_kwargs() 
        }

        # 최신 .Safetensors 옵션 제어 로직, 구형 모델 호환성 오류 방지(.bin, .ckpt 등)
        if model_config:
            opts = model_config.get("options", [])
            if isinstance(opts, str):
                opts = [x.strip() for x in opts.split(',')]
                
            # 명시적으로 no_safetensors가 있으면 끄기 (구형 모델용)
            if "no_safetensors" in opts:
                load_args["use_safetensors"] = False
            # 명시적으로 safetensors가 있으면 켜기
            elif "safetensors" in opts:
                load_args["use_safetensors"] = True
            
            # 🚨 [추가된 부분] variant 파싱 로직 (DreamShaper 모델 등 fp16 파일 로드용)
            for opt in opts:
                opt_str = str(opt).strip()
                if opt_str.startswith("variant="):
                    load_args["variant"] = opt_str.split("=")[1].strip()
            
        try:
            # 기본값: AutoPipelineForInpainting (DreamShaper 등 호환)
            # actual_path 대신 model_id를 넣어 누락된 파일 자동 다운로드를 유도합니다.
            pipeline = AutoPipelineForInpainting.from_pretrained(model_id, safety_checker=None, **load_args)

            # VAE Tiling (Low VRAM)
            if hasattr(pipeline, "vae") and pipeline.vae is not None:
                 if hasattr(pipeline.vae, "enable_tiling"):
                    pipeline.vae.enable_tiling()

           # VRAM 최적화 전략 (Intel XPU 안정성 이슈로 인한 CPU 강제 할당 버전)
            vram_gb = self.hw_info.get("vram_gb", 0)
            
            # 🚨 [수정] Intel XPU 모듈 오류가 있으므로 CPU로 우회
            if self.device == "xpu" or self.device == "cpu":
                print(f"[Loader] 💻 Stability Mode: Forcing CPU device (Original: {self.device})")
                self.device = "cpu" # 로더 세션의 디바이스를 CPU로 변경
                pipeline.to("cpu")
            elif vram_gb <= 8:
                print(f"[Loader] 🚀 Low VRAM ({vram_gb}GB): Enabling Model CPU Offload.")
                pipeline.enable_model_cpu_offload(device=self.device)
            else:
                print(f"[Loader] ⚡ High VRAM ({vram_gb}GB): Moving model to {self.device}.")
                pipeline.to(self.device)
                
            return pipeline, {"pipeline_type": p_type, "precision": str(dtype)}
              
        except Exception as e:
            print(f"[Loader] Standard Load Error: {e}")
            raise
        
    # -------------------------------------------------------------------------
    # Entry Point
    # -------------------------------------------------------------------------
    def load_model(self, model_id, pipeline_type, model_config, precision="auto", quantization="auto", abort_check=None):
        """ 통합 로더 엔트리 포인트 """
        p_type_lower = str(pipeline_type).lower()
        m_id_lower = str(model_id).lower()
        
        print(f"[Loader] Starting load: {model_config.get('short_name')} (Type: {p_type_lower})")
        
        # 1. Qwen Edit Native
        if p_type_lower == "qwenimageeditnative":
            if "2511" in m_id_lower:
                return self._load_qwen_edit_2511_native(model_id, model_config, abort_check)
            return self._load_qwen_edit_native(model_id, model_config, abort_check)

        # 2. Standard Pipeline (그 외 모든 모델 -> DreamShaper 등)
        return self._load_standard_pipeline(model_id, pipeline_type, precision, quantization, abort_check, model_config)