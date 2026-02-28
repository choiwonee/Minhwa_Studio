import sys
import platform
import importlib.util
import importlib.metadata

# --- [중요] 최신 PyTorch 호환성 우회 패치 ---
# 최신 torchvision에서 사라진 모듈을 수동으로 연결하여 basicsr/realesrgan 오류를 방지합니다.
try:
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F
except:
    pass
# ------------------------------------------

def get_pkg_version(import_name, pkg_name=None):
    """패키지의 버전을 안전하게 가져오는 헬퍼 함수"""
    try:
        mod = importlib.import_module(import_name)
        if hasattr(mod, '__version__'):
            return mod.__version__
    except Exception:
        pass
    
    if pkg_name:
        try:
            return importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return "Ready"

def check_project_environment():
    print("=" * 85)
    print(" [ Minhwa Studio ] Comprehensive Environment Checker")
    print("=" * 85)

    # 1. System Info
    print(f"\n[1/6] System Information")
    print(f"  - OS      : {platform.system()} {platform.release()}")
    print(f"  - Python  : {sys.version.split()[0]}")
    print(f"  - Venv    : {sys.prefix}")

    # 패키지 검사 목록: (표시이름, 임포트 모듈명, 실제 설치명(pip))
    check_groups = {
        "[2/6] Core & Image Processing": [
            ("numpy", "numpy", "numpy"),
            ("configobj", "configobj", "configobj"),
            ("requests", "requests", "requests"),
            ("pillow", "PIL", "pillow"),
            ("opencv", "cv2", "opencv-python"),
        ],
        "[3/6] GUI & UI Theme": [
            ("PySide6", "PySide6", "PySide6"),
            ("pyqtdarktheme", "qdarktheme", "pyqtdarktheme"),
        ],
        "[4/6] AI Framework Support": [
            ("accelerate", "accelerate", "accelerate"),
            ("transformers", "transformers", "transformers"),
            ("safetensors", "safetensors", "safetensors"),
            ("sentencepiece", "sentencepiece", "sentencepiece"),
            ("protobuf", "google.protobuf", "protobuf"),
            ("huggingface_hub", "huggingface_hub", "huggingface-hub"),
        ],
        "[5/6] APIs, RAG & Utilities": [
            ("deep-translator", "deep_translator", "deep-translator"),
            ("gradio_client", "gradio_client", "gradio-client"),
            ("pycryptodome", "Crypto", "pycryptodome"),
            ("google-genai", "google.genai", "google-genai"),
            ("sentence-trans", "sentence_transformers", "sentence-transformers"),
            ("faiss", "faiss", "faiss-cpu"),
        ],
        "[6/6] Special Components (Generative & Processing)": [
            ("diffusers", "diffusers", "diffusers"),
            ("sam2", "sam2", "sam2"),
            ("basicsr", "basicsr", "basicsr"),
            ("realesrgan", "realesrgan", "realesrgan"),
            ("nunchaku", "nunchaku", "nunchaku"),
        ]
    }

    for group_title, pkgs in check_groups.items():
        print(f"\n{group_title}")
        for display_name, imp_name, pip_name in pkgs:
            try:
                importlib.import_module(imp_name)
                ver = get_pkg_version(imp_name, pip_name)
                print(f"  [OK] {display_name:<15} : {ver}")
            except ImportError:
                print(f"  [FAIL] {display_name:<15} : Not Found (Need: pip install {pip_name})")
            except Exception as e:
                print(f"  [CHECK] {display_name:<14} : {type(e).__name__} (설치됨, 로드 시 경고)")

    # PyTorch Hardware Check
    print(f"\n[ Hardware & PyTorch Check ]")
    try:
        import torch
        print(f"  - PyTorch : {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  - GPU Mode: NVIDIA CUDA ({torch.cuda.get_device_name(0)})")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            print(f"  - GPU Mode: Intel XPU (Accelerated)")
        else:
            print(f"  - GPU Mode: CPU Only (Slow)")
    except ImportError:
        print(f"  [FAIL] PyTorch : Not Installed")

    print("\n" + "=" * 85)
    print(" Check Completed.")
    print("=" * 85)

if __name__ == "__main__":
    check_project_environment()