import cv2
import numpy as np

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