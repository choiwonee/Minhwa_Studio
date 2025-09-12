import utils.common as COM
import models.depth_estimator as DE
import argparse

from pathlib import Path
from transformers import logging

import warnings

# transformers 경고 레벨 수준을 'ERROR'로 설정, warning 메시지 표시안함.
logging.set_verbosity_error()

# 일반 warning 필터링 처리
warnings.filterwarnings("ignore") 

def main():
    ap = argparse.ArgumentParser(description="민화/고서화 2.5D 변환 (Depth Anything v2)")
    ap.add_argument("--input",               required=True, type=str,     help="입력 이미지 경로")
    ap.add_argument("--outdir",              default="outputs", type=str, help="출력 폴더")
    ap.add_argument("--model-id",            default=None,   help="깊이 추정 모델")
    ap.add_argument("--device",              default=("cuda" if DE.torch.cuda.is_available() else "cpu"), choices=["cuda","cpu"], help="추론 디바이스")

    args = ap.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    COM.ensureDir(outdir)

    # 1) 대상 이미지 입력
    rgbImgNumpy = COM.loadImageNumpy(in_path) # scale option available
    
    # 2) 깊이맵 추출
    print("[1] Running depth modeler ...") # model_id: depth-anything/Depth-Anything-V2-Large-hf
    if args.model_id is None:
        model = "depth-anything/Depth-Anything-V2-Large-hf"
    else:
        model = args.model_id

    depthImg = DE.performDepthModeler(rgbImgNumpy, model_id=model, device=args.device)

    # 3) 16bit 깊이맵 저장 (테스트 출력용)
    print("[2] Saving 16-bit displacement map ...")
    
    filename = COM.generateDepthMapFilename(model, in_path)
    COM.saveUint16Png(outdir / filename, depthImg)

if __name__ == "__main__":
    main()