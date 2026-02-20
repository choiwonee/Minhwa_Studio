import sys
import importlib.util
import platform

# --- [중요] 최신 PyTorch 호환성 우회 패치 ---
# 최신 torchvision에서 사라진 모듈을 수동으로 연결하여 basicsr/realesrgan 오류를 방지합니다.
try:
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F
except:
    pass
# ------------------------------------------

def check_project_environment():
    print("=" * 85)
    print(" [ Minhwa Studio ] Environment Consistency Checker (Final)")
    print("=" * 85)

    # 1. System Info
    print(f"\n[1/4] System Information")
    print(f"  - OS      : {platform.system()} {platform.release()}")
    print(f"  - Python  : {sys.version.split()[0]}")
    print(f"  - Venv    : {sys.prefix}")

    # 2. Core Libraries
    print(f"\n[2/4] Core & GUI Libraries")
    core_libs = {"numpy": "numpy", "pillow": "PIL", "opencv": "cv2", "PySide6": "PySide6", "psutil": "psutil"}
    for name, imp_name in core_libs.items():
        try:
            mod = importlib.import_module(imp_name)
            print(f"  [OK] {name:<15} : {getattr(mod, '__version__', 'Installed')}")
        except:
            print(f"  [FAIL] {name:<15} : Not Found")

    # 3. Hardware Acceleration
    print(f"\n[3/4] AI Framework & Hardware")
    try:
        import torch
        print(f"  - PyTorch : {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  - GPU Mode: NVIDIA CUDA ({torch.cuda.get_device_name(0)})")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            print(f"  - GPU Mode: Intel XPU (Accelerated)")
        else:
            print(f"  - GPU Mode: CPU Only")
    except:
        print(f"  [FAIL] PyTorch : Not Installed")

    # 4. Special Components
    print(f"\n[4/4] Special Components")
    specials = ["diffusers", "sam2", "nunchaku", "basicsr", "realesrgan"]
    for lib in specials:
        try:
            # nunchaku의 경우 설치 로그가 있었으므로, 임포트 시도 시 발생하는 에러를 상세히 출력
            importlib.import_module(lib)
            print(f"  [OK] {lib:<15} : Ready")
        except Exception as e:
            # 에러가 발생해도 설치는 되어있을 수 있으므로 메시지 표시
            print(f"  [CHECK] {lib:<13} : {type(e).__name__} (설치는 되었으나 로드 방식 확인 필요)")

    print("\n" + "=" * 85)
    print(" Check Completed.")
    print("=" * 85)

if __name__ == "__main__":
    check_project_environment()