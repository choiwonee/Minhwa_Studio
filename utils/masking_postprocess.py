import cv2
import numpy as np

def adjust_target_pixel_size(image: np.ndarray, pixel_limit=True, min_area_ratio: float=0.0, max_area_ratio: float=0.0005, debug: bool=False) -> tuple[int, int, int, float, float]:
    """ 이미지 해상도 티어 기반 지정 픽셀 수와 해당 최종 비율(Float)을 배열 구조로 정의하여 반환.
        - 반환 순서: total_pixels(int), min_pixels(int), max_pixels(int), min_ratio_final(float), max_ratio_final(float)
    """
    if image is None or image.size == 0:
        return 0, 0, 0, 0.0, 0.0
        
    H, W = image.shape[:2]
    total_pixels = H * W 
    
    # 해상도 티어별 limit_pixels 정의 (배열 구조), (Threshold, Limit_Pixels) 순서이며, Threshold는 해당 limit이 적용되는 최대 픽셀 수를 의미.
    PIXEL_LIMIT_TIERS = [
        (500_000, 200, 2),
        (1_000_000, 500, 5),
        (2_000_000, 1000, 10),
        (4_000_000, 2000, 20),
        (5_000_000, 2500, 25),
        (8_000_000, 4000, 40),
        (10_000_000, 5000, 50),
        (20_000_000, 10000, 100),
        (30_000_000, 17000, 170),
        (50_000_000, 30000, 300)
    ]
    
    # total_pixels에 맞는 limit_pixels 찾기
    max_limit_pixels = 40000
    min_limit_pixels = 400
    for threshold, max_limit, min_limit in PIXEL_LIMIT_TIERS:
        if total_pixels < threshold:
            max_limit_pixels = max_limit
            min_limit_pixels = min_limit
            break
    
    # 최종 픽셀 계산
    if pixel_limit:
        min_pixels = min(min_limit_pixels, total_pixels * min_area_ratio)
        max_pixels = min(max_limit_pixels, total_pixels * max_area_ratio)
    else:
        min_pixels = total_pixels * min_area_ratio
        max_pixels = total_pixels * max_area_ratio 
        
    # 최종 비율 계산
    total_pixels_f = float(total_pixels)
    if total_pixels_f == 0:
        return 0, 0, 0, 0.0, 0.0
        
    min_ratio_final = min_pixels / total_pixels_f
    max_ratio_final = max_pixels / total_pixels_f
    if debug:
        print(f"### [DEBUG] <adjust_target_pixel_size> - total_pixels: {int(total_pixels):,}, min_pixels: {int(min_pixels):,}, max_pixels: {int(max_pixels):,}, min_ratio_final: {min_ratio_final:.4f}, max_ratio_final: {max_ratio_final:.4f}")
        
    # 반환 구조: 픽셀 수(int) 3개 + 최종 비율(float) 2개
    return int(total_pixels), int(min_pixels), int(max_pixels), min_ratio_final, max_ratio_final


def remove_white_noise_component(image: np.ndarray, invert: bool=False, pixel_limit: bool=True, min_area_ratio: float=0.0005, debug: bool=False) -> np.ndarray:
    """ 연결 요소 분석을 통해 작은 흰색 노이즈 제거 (min_area_ratio 기준). 벡터화 최적화 버전 """
    if image is None or image.size == 0:
        return image
    
    image_u8 = np.ascontiguousarray(image.copy() if not invert else 255 - image) # 메모리 레이아웃 최적화
    if len(image_u8.shape) == 3: # 3채널 -> 1채널(H, W)
        image_u8 = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY)
    # H, W = image_u8.shape # 높이, 넓이
    
    _, min_pixels, _, _, _ = adjust_target_pixel_size(image_u8, pixel_limit=pixel_limit, min_area_ratio=min_area_ratio, debug=debug) # 최소/최대 픽셀 기준
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image_u8) # 객체 레이블 탐색 및 통계정보 추출
    
    # 벡터화 최적화: 배열 연산으로 한 번에 처리
    areas = stats[1:, cv2.CC_STAT_AREA] # 배경(0) 제외
    valid_labels = np.where(areas > min_pixels)[0] + 1 # +1은 배경 제외 보정
    
    # 마스크 생성을 isin으로 한 번에 처리
    cleaned_mask = np.isin(labels, valid_labels).astype(np.uint8) * 255
    
    if debug:
        removed_count = num_labels - 1 - len(valid_labels)
        print(f"### [DEBUG] <remove_white_noise_component>")
        print(f"    - Total components: {num_labels - 1}")
        print(f"    - Removed (small): {removed_count}")
        print(f"    - Kept (large): {len(valid_labels)}")
        print(f"    - Min area threshold: {min_pixels:,} px")
    
    if invert:
        cleaned_mask = 255 - cleaned_mask
    
    return cleaned_mask
