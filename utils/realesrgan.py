import torch
import numpy as np
from PIL import Image
import os
import sys
import gc
from pathlib import Path

# [Config] 설정 로더 가져오기
try:
    from utils.config_loader import config
except ImportError:
    config = None

# [Patch] torchvision 호환성
try:
    from torchvision.transforms import functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F
except ImportError:
    pass

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    HAS_REALESRGAN = True
except ImportError:
    HAS_REALESRGAN = False

class RealESRGANUpscaler:
    # 기본 공식 다운로드 URL
    DEFAULT_URL = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'

    def __init__(self, model_scale=4, model_path=None, device=None):
        """ 초기화 메서드
            - 설정 파일의 weights_path를 프로젝트 루트 기준 절대 경로로 변환 처리 """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.half = False 
        self.model_scale = model_scale
        self.upsampler = None
        self.is_ready = False

        if not HAS_REALESRGAN:
            print("[RealESRGAN] Library not found. Install 'realesrgan' & 'basicsr'.")
            return

        if model_path is None:
            # 1. 소스 기준 프로젝트 루트 계산 (utils 폴더의 상위)
            script_dir = Path(__file__).parent.parent.absolute() 
            
            # 2. Config의 weights_path 적용
            if config and getattr(config, "weights_path", None):
                raw_path = config.weights_path # "./models/weights"
                if not os.path.isabs(raw_path):
                    # 상대 경로를 프로젝트 루트 절대 경로와 결합
                    base_dir = (script_dir / raw_path.replace("./", "")).resolve()
                else:
                    base_dir = Path(raw_path)
            else:
                base_dir = script_dir / "models" / "weights" # 기본 fallback도 소스 폴더 내로 지정
            
            self.base_dir = base_dir
            self.save_dir = self.base_dir / "realesrgan"
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = str(self.save_dir / 'RealESRGAN_x4plus.pth')
            print(f"[Loader] Integrated Model Root: {self.base_dir}")

        # 파일 존재 여부 확인 및 다운로드
        if not os.path.exists(model_path):
            print(f"[RealESRGAN] Weights not found at {model_path}. Starting download...")
            self._download_weights(model_path)

        try:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            
            self.upsampler = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=model,
                tile=0, 
                tile_pad=10,
                pre_pad=0,
                device=self.device,
                half=self.half
            )
            self.is_ready = True
            print(f"[RealESRGAN] Initialized on {self.device}")

        except Exception as e:
            print(f"[RealESRGAN] Init Error: {e}")

    def _download_weights(self, path):
        """ 가중치 다운로드 (Config URL 우선 적용) """
        target_url = self.DEFAULT_URL
        if config:
            cfg_url = config.get_config_value("Settings", "realesrgan_url", None)
            if cfg_url and cfg_url.strip():
                target_url = cfg_url.strip()
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(f"[RealESRGAN] Downloading weights to {path}...")
            torch.hub.download_url_to_file(target_url, path)
        except Exception as e:
            print(f"[RealESRGAN] ❌ Download failed: {e}")

    def upscale(self, image: Image.Image, scale_factor: float = 4.0, tile_size: int = 512) -> Image.Image:
        """ 업스케일 추론 실행 """
        if not self.is_ready or self.upsampler is None:
            return image

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        img_np = np.array(image.convert("RGB"))[:, :, ::-1] 

        tile_attempts = [tile_size]
        if tile_size > 256: tile_attempts.append(256)
        if tile_size > 128: tile_attempts.append(128)
        
        output_np = None
        for t_size in tile_attempts:
            try:
                self.upsampler.tile_size = t_size
                output, _ = self.upsampler.enhance(img_np, outscale=4)
                output_np = output
                break 
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e

        if output_np is None: return image

        output_pil = Image.fromarray(output_np[:, :, ::-1].astype(np.uint8), 'RGB')
        target_w, target_h = int(image.width * scale_factor), int(image.height * scale_factor)
        
        if output_pil.size != (target_w, target_h):
            output_pil = output_pil.resize((target_w, target_h), Image.LANCZOS)

        return output_pil