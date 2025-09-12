import numpy as np
import torch
import utils.common as COM

from transformers import AutoImageProcessor, AutoModelForDepthEstimation

def performDepthModeler(rgb_img_numpy: np.ndarray, model_id: str = "depth-anything/Depth-Anything-V2-Large-hf", device: str = "cuda") -> np.ndarray:
    global outDir

    print(f"[INFO] model_id: {model_id}, torch.cuda: {device}")
    print(f"[NUMPY] {rgb_img_numpy.shape, rgb_img_numpy.dtype}")

    # depth-anything 모델, model_id: depth-anything/Depth-Anything-V2-Large-hf
    if "depth-anything" in model_id.lower():
        # processor가 입력 이미지를 모델이 먹을 수 있는 텐서로 바꿈
        processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)

        # AutoModelForDepthEstimation이 깊이를 예측
        model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)

        # 결과를 NumPy 배열로 변환하고, 0~1 사이로 정규화
        # inference_mode 학습이 아니라 추론만 함
        with torch.inference_mode():
        # with torch.no_grad():
            inputs = processor(images=rgb_img_numpy, return_tensors="pt").to(device)
            pred = model(**inputs).predicted_depth
        # GPU → CPU로 옮기고 NumPy 배열로 변환, squeeze()는 차원을 줄여서 (H,W) 형태로 만듦
        depth = pred.squeeze().detach().cpu().numpy().astype(np.float32)        
    
    # elif "marigold" in model_id.lower(): # Marigold 모델, model_id: prs-eth/marigold-depth-hr-v1-1        
    #     pipe = MarigoldDepthPipeline.from_pretrained(
    #         model_id,
    #         torch_dtype=torch.float16 if device == "cuda" else torch.float32
    #     ).to(device)

    #     pil_image = Image.fromarray(rgb_img_numpy) # Numpy 이미지 배열 -> PIL 이미지 변환.
    #     with torch.inference_mode():
    #         depth_output = pipe(
    #             image=pil_image,
    #             num_inference_steps=100, # 과정 단계 수 조절 (고품질)
    #             ensemble_size=1,         # 앙상블 횟수(성능 향상 시 5~10까지), ensemble_size 1 보다 크면 pip install scipy 선행 설치 필요함. 
    #         ).prediction
    #     # [후처리 1] Colormap(RGB) 시각화 이미지로 저장
    #     vis = pipe.image_processor.visualize_depth(depth_output, color_map="Spectral") # list
    #     vis[0].save(outDir / "depth_vis.png")

    #     # [후처리 2] 16비트 PNG 포맷 저장 (정밀 깊이값)
    #     # depth_16bit = pipe.image_processor.export_depth_to_16bit_png(depth_output) # list
    #     # depth_16bit[0].save(outDir / "depth_16bit-01.png")
    #     # visualize_depth()는 [0, 1] 깊이맵을 선택한 컬러맵(Spectral 기본) 기반 RGB 이미지로 변환합니다.
    #     # export_depth_to_16bit_png()는 1채널 16비트 PNG로 변환합니다.  
        
    #     depth = np.array(depth_output) # Marigold 에서 가져온 깊이 맵
    #     depth = np.squeeze(depth) # 배치 및 채널 차원을 포함하는 4D 구조(1,H,W,1) 텐서이므로, numpy 변환시 이 차원들을 **제거(squeeze) -> shape: (H, W)
        
    # elif "zoedepth" in model_id.lower(): # ZoeDepth 계열, AutoModel 사용 + post_process_depth_estimation 필요, model_id: Intel/zoedepth-kitti or Intel/zoedepth-nyu-kitti or Intel/zoedepth-nyu
    #     processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
    #     model     = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    #     with torch.inference_mode():
    #         inputs  = processor(images=rgb_img_numpy, return_tensors="pt").to(device)
    #         outputs = model(**inputs)
    #     # ZoeDepth 공식 가이드에 따른 패딩/리사이즈 보정을 post_process로 처리
    #     post  = processor.post_process_depth_estimation(outputs, target_sizes=[(rgb_img_numpy.shape[0], rgb_img_numpy.shape[1])], do_remove_padding=False)[0]["predicted_depth"]
    #     depth = post.detach().cpu().numpy()
        
    else: # DPT/MiDaS 계열 모델 처리, DPT 전용 로더 사용(post_process_depth_estimation 지원 X), model_id: Intel/dpt-large or Intel/dpt-hybrid-midas
        processor = AutoImageProcessor.from_pretrained(model_id,  use_fast=True)
        model     = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
        with torch.inference_mode():
            inputs  = processor(images=rgb_img_numpy, return_tensors="pt").to(device)
            outputs = model(**inputs)
            # depth   = outputs.predicted_depth.squeeze().cpu().numpy()
            # 예측된 깊이 맵을 원본 이미지의 크기로 리사이즈하고 노이즈 제거 등을 수행, 딕셔너리이며, 그 안에 실제 디스패스 맵 텐서가 "predicted_depth"라는 키로 들어있음
            resized_outputs = processor.post_process_depth_estimation(
                outputs,
                target_sizes=[(rgb_img_numpy.shape[0], rgb_img_numpy.shape[1])]
            )
            # 즉 resized_outputs[0]이 딕셔너리이고, 그 안에 "predicted_depth" 키가 들어있음
            depth_tensor = resized_outputs[0]["predicted_depth"]
            depth = depth_tensor.squeeze().cpu().numpy() # 텐서에 numpy() 메서드를 적용하여 변환
            
    depth = COM.normalizeNpImgArray(depth) # 최종 깊이 맵 정규화 (0..1 범위)
    return depth