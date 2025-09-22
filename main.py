import utils.common as COM
import utils.depth_postprocess as DP
import models.depth_estimator as DE
import argparse
import warnings

from pathlib import Path
from transformers import logging

# transformers 경고 레벨 수준을 'ERROR'로 설정, warning 메시지 표시안함.
logging.set_verbosity_error()

# 일반 warning 필터링 처리
warnings.filterwarnings("ignore") 

def main():
    ap = argparse.ArgumentParser(description="민화/고서화 2.5D 변환 (Depth Anything v2)")
    ap.add_argument("--input",               required=True, type=str,     help="입력 이미지 파일 경로")
    ap.add_argument("--output-dir",          default="outputs", type=str, help="출력 폴더")
    ap.add_argument("--model-id",            default="depth-anything/Depth-Anything-V2-Large-hf",   help="깊이 추정 모델")
    ap.add_argument("--device",              default=("cuda" if DE.torch.cuda.is_available() else "cpu"), choices=["cuda","cpu"], help="추론 디바이스")

    #ap.add_argument("--edge-boost",          type=float, default=0.35, help="깊이 경계 선명화 강도, 0..1")
    ap.add_argument("--normal-scale",        type=float, default=50.0, help="노멀맵 입체 강도 조절")
    
    args = ap.parse_args()

    in_path = Path(args.input)
    output_dir = Path(args.output_dir)
    COM.ensure_dir(output_dir)

    # 1) 대상 이미지 입력 로드
    input_image_numpy = COM.load_image_numpy(in_path) # scale option available
    # input_image_height, input_image_width = input_image_numpy.shape[:2] # RGB image 2D shape
    # input_image = (input_image_numpy.astype(np.float32) / 255.0).copy() # RGB image
    
    # 2) 깊이 추출 (Depth modeler)
    print("[1/5] Running depth modeler ...")
    depth_image = DE.perform_depth_anything_modeler(input_image_numpy, model_id=args.model_id, device=args.device) # 필요시 사용: combined_edges_image, edge_mask_image

    # 3) 경량 엣지 부스팅 (선택)
    # if args.edge_boost > 1e-6:
    #     print("[2/5] Enhancing depth edges ...")
    #     depth_image = COM.enhance_depth_edges(depth_image, strength=args.edge_boost)

    # 4) 깊이맵 저장 (확인용)
    print("[3/5] Saving depth map ...")
    COM.save_unit16_png(depth_image, output_dir, in_path, args.model_id, "DepthMap")

    # 5) 노멀 맵 생성 (with bilateral 활성화)
    print("[4/5] Generating normal map ...")
    normal_image = DP.depth_to_normal(depth_image, normal_scale=args.normal_scale)
    COM.save_rgb_png(normal_image, output_dir, in_path, args.model_id, "NormalMap")

if __name__ == "__main__":
    main()