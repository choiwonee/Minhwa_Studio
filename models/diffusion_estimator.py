import gc
import inspect
import torch
import numpy as np
import traceback
import warnings
from PIL import Image

# 경고 억제
warnings.filterwarnings("ignore")
from diffusers.utils import logging as diffusers_logging
diffusers_logging.set_verbosity_error()

try:
    from utils.config_loader import config
except ImportError:
    config = None

try:
    from utils import token_key
except ImportError:
    token_key = None

from utils.realesrgan import RealESRGANUpscaler
from models.unified_model_loader import UnifiedModelLoader
from utils.vram_monitor import VRAMMonitor

class DiffusionEstimator:
    def __init__(self, progress_callback=None):
        self.is_ready = False
        self.pipeline = None
        self.model_info = {}
        self.current_model_key = None
        self.progress_callback = progress_callback or (lambda *args, **kwargs: None)

        self.vram_monitor = VRAMMonitor()
        self.hw_info = self.detect_hardware_capabilities()

        self.device, self.hf_token, self.api_key = self._prepare_device_and_keys()

        self.loader = UnifiedModelLoader(device=self.device, hw_info=self.hw_info, hf_token=self.hf_token)
        self.upscaler = RealESRGANUpscaler(model_scale=4)

        print(f"[Init] Device: {self.device}, Pascal: {self.hw_info.get('is_pascal')}")

    def _infer_pipeline_device(self) -> str:
        return getattr(self, "device", "cpu")

    def set_progress_callback(self, callback):
        self.progress_callback = callback if callback else (lambda *args, **kwargs: None)

    def detect_hardware_capabilities(self):
        info = {"device": "cpu", "vram_gb": 0, "is_pascal": False, "description": "CPU Mode"}
        if torch.cuda.is_available():
            info["device"] = "cuda"
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024**3)
            info["vram_gb"] = round(vram_gb, 1)
            info["description"] = f"NVIDIA {props.name} ({vram_gb:.1f}GB)"
            if props.major == 6:
                info["is_pascal"] = True
                info["description"] += " - Pascal (Legacy)"
        # 👇 [추가됨] Intel 그래픽카드(XPU) 인식 로직
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            info["device"] = "xpu"
            info["vram_gb"] = 8.0 # 기본값
            info["description"] = "Intel XPU"
            try:
                props = torch.xpu.get_device_properties(0)
                vram_gb = props.total_memory / (1024**3)
                info["vram_gb"] = round(vram_gb, 1)
                info["description"] = f"Intel {props.name} ({vram_gb:.1f}GB)"
            except:
                pass
        return info

    def _prepare_device_and_keys(self):
        device = self.hw_info.get("device", "cpu")
        hf_token = None
        api_key = None
        if token_key is not None:
            hf_token = token_key.get_valid_hf_token()
            api_key = token_key.get_valid_api_key()
        if hf_token is None and config is not None:
            hf_token = getattr(config, "hf_token", None)
        return device, hf_token, api_key

    # -------------------------------------------------------------------------------------------------------------------------------------
    # 모델 로드 및 관리
    # -------------------------------------------------------------------------------------------------------------------------------------
    def load_model(self, model_key: str, precision="auto", quantization="auto", abort_check=None, **kwargs):
        def check_abort(stage=""):
            if abort_check and abort_check():
                self.unload_model()
                raise RuntimeError("USER_CANCEL")

        check_abort("start")
        
        if self.current_model_key == model_key and self.is_ready and self.pipeline is not None:
            return True

        self.unload_model()

        try:
            model_config = kwargs.get("model_config", {})
            p_type = model_config.get("pipeline_type", "SDInpaint")
            
            self.pipeline, actual_config = self.loader.load_model(
                model_id=model_config.get("repo_id", model_key), 
                pipeline_type=p_type, 
                model_config=model_config, 
                precision=precision, 
                quantization=quantization, 
                abort_check=abort_check
            )
            
            self.model_info = {**actual_config, "key": model_key}
            self.current_model_key = model_key
            self.is_ready = True
            
            return True

        except Exception as e:
            if "USER_CANCEL" not in str(e):
                traceback.print_exc()
            self.unload_model()
            raise e

    def _safe_truncate_prompt(self, prompt: str, pipeline_type: str, max_tokens: int = None) -> str:
        """ 프롬프트 길이 안전하게 자르기 """
        if not prompt or not prompt.strip():
            return ""
        
        p_type_lower = str(pipeline_type).lower()
        is_qwen = "qwen" in p_type_lower
        
        if max_tokens is None:
            # Qwen은 긴 프롬프트 지원, 나머지는 75 토큰 제한
            max_tokens = 2000 if is_qwen else 75

        tokenizer = getattr(self.pipeline, "tokenizer", None)
        if tokenizer is None:
            return " ".join(prompt.split()[:max_tokens])
        
        try:
            encoded = tokenizer(prompt, truncation=True, max_length=max_tokens, return_tensors="pt", add_special_tokens=False)
            return tokenizer.decode(encoded.input_ids[0], skip_special_tokens=True)
        except Exception:
            return prompt
        
    def _build_prompt_payload(self, prompt: str, negative_prompt: str, p_type: str) -> dict:
        """ 프롬프트 구성 """
        mode = self.model_info.get("prompt_mode", "two_way")
        
        if mode == "mix_way":
            combined = prompt
            if negative_prompt and negative_prompt.strip():
                combined = f"{prompt}. Avoid: {negative_prompt}"
            return {"prompt": self._safe_truncate_prompt(combined, p_type)}
        
        else: # two_way
            return {
                "prompt": self._safe_truncate_prompt(prompt, p_type),
                "negative_prompt": self._safe_truncate_prompt(negative_prompt, p_type) if negative_prompt else ""
            }

    def _prepare_pipeline_args(self, pil_img, pil_mask, prompt: str, negative_prompt: str, num_inference_steps: int, guidance_scale: float, p_type: str, use_input_image: bool, abort_check, **kwargs) -> dict:
        """ 파이프라인 인자 구성 """
        args = self._build_prompt_payload(prompt, negative_prompt, p_type)
        args.update({
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale
        })
        
        if "width" in kwargs and "height" in kwargs:
             args["width"], args["height"] = kwargs["width"], kwargs["height"]
        
        sig = inspect.signature(self.pipeline.__call__)
        p_type_lower = p_type.lower()
        
        # Qwen 전용 파라미터
        if "qwen" in p_type_lower:
            args.update({"true_cfg_scale": 4.0, "num_inference_steps": min(num_inference_steps, 20)})
            
        # 이미지/마스크 매핑
        if pil_img is not None:
            if "image" in sig.parameters: args["image"] = pil_img
            elif "input_image" in sig.parameters: args["input_image"] = pil_img
            
            if pil_mask is not None:
                if "mask_image" in sig.parameters: args["mask_image"] = pil_mask
                elif "mask" in sig.parameters: args["mask"] = pil_mask

        # 콜백 등록
        if "callback_on_step_end" in sig.parameters:
            def callback_wrapper(pipe, step_index, timestep, callback_kwargs):
                if abort_check and abort_check(): raise RuntimeError("USER_CANCEL")
                if self.progress_callback:
                    try:
                        pct = int((step_index / max(1, num_inference_steps - 1)) * 100)
                        self.progress_callback(pct)
                    except: pass
                return callback_kwargs
            args["callback_on_step_end"] = callback_wrapper

        # 시그니처 필터링
        return {k: v for k, v in args.items() if k in sig.parameters}
    
    def _calculate_target_dimensions(self, res_mode: str, pil_img: Image.Image, align_unit: int = 8) -> tuple:
        """ 해상도 계산 (배수 정렬) """
        target_w, target_h = 1024, 1024
        
        if res_mode == "match_input" and pil_img is not None:
            target_w, target_h = pil_img.size
        elif "x" in res_mode:
            try:
                target_w, target_h = map(int, res_mode.split("x"))
            except ValueError:
                pass

        target_w = (target_w // align_unit) * align_unit
        target_h = (target_h // align_unit) * align_unit
        
        return target_w, target_h

    def _smart_resize_image(self, pil_img: Image.Image, target_w: int, target_h: int, method=Image.LANCZOS) -> Image.Image:
        """ 비율 유지 리사이즈 후 Center Crop """
        if pil_img is None: return None
        src_w, src_h = pil_img.size
        src_ratio = src_w / src_h
        target_ratio = target_w / target_h
        
        if src_ratio > target_ratio:
            new_h = target_h
            new_w = int(new_h * src_ratio)
        else:
            new_w = target_w
            new_h = int(new_w / src_ratio)
            
        img_resized = pil_img.resize((new_w, new_h), method)
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        return img_resized.crop((left, top, left + target_w, top + target_h))
    
    def predict(self, image: np.ndarray, mask: np.ndarray, prompt: str, negative_prompt: str = "", num_inference_steps: int = 30, guidance_scale: float = 7.5, upscale_opts: dict = None, abort_check=None, use_input_image: bool = True, **kwargs) -> np.ndarray:
        """ 추론 실행 메인 메서드 """
        if not self.is_ready or self.pipeline is None:
            raise RuntimeError("Model not loaded.")
        
        self._post_inference_cleanup()
        
        p_type = self.model_info.get('pipeline_type', '').lower()
        # [수정 2번 완료] 의미 없이 소문자로 바꾸기만 하던 좀비 코드(str(self.model_info...)) 삭제됨
        
        is_qwen = "qwen" in p_type
        res_mode = kwargs.get("resolution_mode", "match_input")
        align_unit = 8 # SD 1.5 표준
        
        # 🚨 [수정 1번 완료] 메모리 누수 방지를 위해 try 블록을 제일 위로 끌어올림
        try:
            # 1. Qwen 격리 로직 (로컬 실행 시)
            if is_qwen and not kwargs.get("is_remote", False):
                # Qwen은 격리된 메서드에서 실행 (실행 후 맨 아래 finally에서 메모리 자동 청소됨)
                return self._predict_qwen_isolated(
                    self.pipeline,
                    image=image,
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    abort_check=abort_check,
                    **kwargs
                )
            
            # 2. 일반 모델 (DreamShaper 등) 실행 로직
            pil_img = Image.fromarray(image).convert("RGB") if image is not None else None
            target_w, target_h = self._calculate_target_dimensions(res_mode, pil_img, align_unit=align_unit)

            # 이미지 리사이징
            if pil_img is not None:
                if res_mode == "match_input":
                    pil_img = pil_img.resize((target_w, target_h), Image.LANCZOS)
                else:
                    pil_img = self._smart_resize_image(pil_img, target_w, target_h)
            
            # 마스크 처리 (Inpainting 모델인 경우 필수)
            pil_mask = None
            if mask is not None:
                mask_pil_raw = Image.fromarray(mask).convert("L")
                if res_mode == "match_input":
                    pil_mask = mask_pil_raw.resize((target_w, target_h), Image.NEAREST)
                else:
                    pil_mask = self._smart_resize_image(mask_pil_raw, target_w, target_h, method=Image.NEAREST)
            elif "inpaint" in p_type:
                # 마스크가 없는데 인페인팅 모델인 경우 -> 전체 마스크 생성
                pil_mask = Image.new("L", (target_w, target_h), 255)

            kwargs["width"], kwargs["height"] = target_w, target_h
            
            with torch.inference_mode():
                # 1. 파이프라인 인자 준비
                pipeline_args = self._prepare_pipeline_args(pil_img, pil_mask, prompt, negative_prompt, num_inference_steps, guidance_scale, p_type, use_input_image, abort_check, **kwargs)
                
                # 🚨 [중요: 형식 오류 해결] 리스트로 감싸진 이미지를 단일 이미지로 꺼내기
                for k in ["image", "mask_image"]:
                    if k in pipeline_args:
                        val = pipeline_args[k]
                        if isinstance(val, list) and len(val) > 0:
                            pipeline_args[k] = val[0]
                
                print(f"[Predict] Running {p_type} | Steps: {num_inference_steps}")
                
                # 2. 장치(Device)에 따른 Autocast 설정
                effective_device = "cuda" if "cuda" in str(self.device) else "cpu"
                
                if effective_device == "cuda" and not self.hw_info.get("is_pascal", False):
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        res = self.pipeline(**pipeline_args)
                else:
                    # CPU 또는 Pascal GPU는 autocast 없이 실행
                    res = self.pipeline(**pipeline_args)
                
                gen_img = res.images[0]
                
                # 임시 변수 해제
                del res

                # 후처리 (합성)
                return self.manual_post_process(gen_img, image, mask, upscale_opts)
        
        finally:
            # ✅ [가장 중요한 부분] Qwen이든 일반 모델이든, 에러가 나든 안 나든 
            # 함수가 끝날 때 무조건 VRAM 메모리를 싹 비워줍니다.
            self._post_inference_cleanup()
    
    def _predict_qwen_isolated(self, pipeline, prompt, image, num_inference_steps, guidance_scale, abort_check=None, **kwargs) -> np.ndarray:
        """ QwenImageEdit 전용 추론 """
        try:
            if abort_check and abort_check(): return None

            # 프로세서 가드
            if hasattr(pipeline, "processor"):
                proc = pipeline.processor
                if hasattr(proc, "do_resize"): proc.do_resize = False

            # Qwen은 32배수 정렬 권장
            pil_img = Image.fromarray(image[0] if isinstance(image, list) else image).convert("RGB")
            w, h = (pil_img.width // 32) * 32, (pil_img.height // 32) * 32
            pil_img = pil_img.resize((max(32, w), max(32, h)), Image.BICUBIC)

            with torch.inference_mode():
                out = pipeline(prompt=prompt, image=pil_img, num_inference_steps=int(num_inference_steps), output_type="pil", true_cfg_scale=float(guidance_scale)).images[0]

            return np.array(out.convert("RGB"))

        except Exception as e:
            print(f"[Estimator] Qwen Execution Failed: {e}")
            return None

    def manual_post_process(self, gen_img: Image.Image, original: np.ndarray, mask: np.ndarray, upscale_opts: dict = None) -> np.ndarray:
        """ 공통 후처리: Upscale & Composite """
        if gen_img is None: return None
        
        org_h, org_w = None, None
        if original is not None:
            org_h, org_w = original.shape[:2]

        final_img = gen_img 
        target_size = gen_img.size
        
        # Upscale
        if upscale_opts:
            scale = upscale_opts.get("scale", 4.0)
            if self.upscaler:
                try:
                    final_img = self.upscaler.upscale(final_img, scale_factor=scale)
                except: pass
            
            if upscale_opts.get("resize_back", True):
                final_img = final_img.resize(target_size, Image.LANCZOS)
        
        # Composite (마스크 영역만 합성)
        if original is not None and mask is not None and mask.mean() < 250:
            try:
                w, h = final_img.size
                orig_pil = self._smart_resize_image(Image.fromarray(original).convert("RGB"), w, h)
                mask_pil = self._smart_resize_image(Image.fromarray(mask).convert("L"), w, h, Image.NEAREST)
                final_img = Image.composite(final_img, orig_pil, mask_pil)
            except: pass
                
        # 원본 크기 복구
        if org_w and org_h and final_img.size != (org_w, org_h):
             final_img = final_img.resize((org_w, org_h), Image.LANCZOS)
        
        return np.array(final_img)
    
    def unload_model(self):
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        self._post_inference_cleanup()
        
    def _post_inference_cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()