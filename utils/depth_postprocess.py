import numpy as np
import cv2
import utils.common as COM

def enhance_depth_edges(depth01: np.ndarray, strength: float = 0.4) -> np.ndarray:
    """
    깊이맵(0~1 범위 float32)의 경계(edge)만 살짝 선명화하는 함수.
    
    Parameters
    ----------
    depth01 : np.ndarray / 깊이맵
    strength : float, default=0.4 / 엣지 강화 정도, 0은 원본 유지, 1에 가까울수록 강화
    
    Returns
    -------
    np.ndarray / 강화된 깊이맵 (0~1 범위 float32)
    """
    # --- (1) 입력을 float32로 변환 ---
    d = depth01.astype(np.float32)
    
    # --- (2) 3채널 이상이면 그레이스케일로 변환 ---
    # 입력이 컬러 이미지(BGR 또는 BGRA)일 수 있으므로 단일 채널로 변환
    if d.ndim == 3 and d.shape[2] in [3, 4]: 
        d_gray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    else:
        d_gray = d
        
    # --- (3) LoG 기반 언샵 마스크로 엣지 검출 ---
    blur = cv2.GaussianBlur(d_gray, (0, 0), 1.2)  # 노이즈 감소용 블러
    edges = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)  # 엣지 검출
    
    # --- (4) 엣지 강화 ---
    # 깊이맵에서 검출한 엣지를 일부 반영하여 선명화
    sharpen = d - 0.75 * edges
    
    # --- (5) 원본과 합성하여 강도 조절 ---
    out = (1.0 - strength) * d + strength * sharpen  # strength: 0~1 비율로 조정
    
    # --- (6) 결과 정규화 및 반환 ---
    return COM.normalize_np_img_array(out)


<<<<<<< HEAD
def depth_to_normal(depth01: np.ndarray,
                    normal_scale: float = 50.0,
                    alpha: float = 0.7,    # Gradient 가중치
                    beta: float = 0.3      # Sobel 가중치
                    ) -> np.ndarray:
    """
    깊이맵 → 노멀맵 변환

    Parameters
    ----------
    depth01 : np.ndarray / 깊이맵 (float32)
    normal_scale : float / 깊이 기울기 → 노멀 강도 스케일링
    alpha : float / Gradient 기반 노멀 가중치
    beta : float / Sobel 기반 노멀 가중치 (alpha + beta ≈ 1.0 권장)

    Returns
    -------
    np.ndarray / RGB 노멀맵, 0~1 범위 float32
    """

    # --- (1) Gradient 기반 노멀 계산 ---
    gy, gx = np.gradient(depth01.astype(np.float32))
    nz = np.ones_like(depth01)                       # z방향 성분 (항상 1)
    nx = -gx * normal_scale                          # x방향 성분 (음수: 기울기 반전)
    ny = -gy * normal_scale                          # y방향 성분 (음수: 기울기 반전)
    norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-9     # 벡터 길이 계산
    nx /= norm; ny /= norm; nz /= norm
    normal_grad = np.stack([(nx + 1) * 0.5,          # [-1,1] → [0,1] 매핑
                            (ny + 1) * 0.5,
                            (nz + 1) * 0.5], axis=-1)

    # --- (2) Sobel 기반 노멀 계산 ---
    sobelx = cv2.Sobel(depth01.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(depth01.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    nz_s = np.ones_like(depth01)                      # z방향 성분
    nx_s = -sobelx * normal_scale                     # x방향 성분
    ny_s = -sobely * normal_scale                     # y방향 성분
    norm_s = np.sqrt(nx_s**2 + ny_s**2 + nz_s**2) + 1e-9
    nx_s /= norm_s; ny_s /= norm_s; nz_s /= norm_s
    normal_sobel = np.stack([(nx_s + 1) * 0.5,
                             (ny_s + 1) * 0.5,
                             (nz_s + 1) * 0.5], axis=-1)

    # --- (3) Gradient + Sobel 혼합 ---
    normal_final = alpha * normal_grad + beta * normal_sobel
    normal_final = np.clip(normal_final, 0.0, 1.0).astype(np.float32)

    # --- (4) 결과 반환 ---
    return normal_final
=======
# def depth_to_normal(depth01: np.ndarray,
#                     normal_scale: float = 50.0,
#                     sigma: float = 1.0,
#                     bilateral: bool = True,
#                     d: int = 9,
#                     sigmaColor: float = 0.1,
#                     sigmaSpace: float = 5.0) -> np.ndarray:
#     """
#     깊이맵(0~1 범위 float32)을 노멀맵(RGB 이미지)으로 변환하는 함수
#     """

#     # --- (1) 깊이맵 필터링 (노이즈 제거) ---
#     if bilateral:
#         # Bilateral Filter: 경계(Edge)는 유지하면서 노이즈 제거
#         depth_filtered = cv2.bilateralFilter(
#             depth01.astype(np.float32),
#             d=d,
#             sigmaColor=sigmaColor * 255.0,
#             sigmaSpace=sigmaSpace
#         )
#     else:
#         # Gaussian Blur: 전체적으로 부드럽게 흐림
#         depth_filtered = cv2.GaussianBlur(depth01, (0, 0), sigma)

#     # --- (2) 깊이맵 기울기 계산 ---
#     # gx: 가로(x) 방향 변화율, gy: 세로(y) 방향 변화율
#     gy, gx = np.gradient(depth_filtered)

#     # --- (3) 법선 벡터(nx, ny, nz) 계산 ---
#     nz = np.ones_like(depth_filtered)        # z방향 성분 (항상 1로 둠 → 카메라 방향)
#     nx = -gx * normal_scale                  # x방향 성분 (음수: 기울기 반전)
#     ny = -gy * normal_scale                  # y방향 성분 (음수: 기울기 반전)

#     # --- (4) 벡터 정규화 (길이=1로 맞추기) ---
#     norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
#     nx /= norm
#     ny /= norm
#     nz /= norm

#     # --- (5) [-1, 1] → [0, 1] 범위로 변환 ---
#     # RGB로 매핑: (nx, ny, nz) → (R, G, B)
#     normal = np.stack([
#         (nx + 1) * 0.5,
#         (ny + 1) * 0.5,
#         (nz + 1) * 0.5
#     ], axis=-1)

#     # --- (6) 결과 반환 ---
#     # 0~1 범위 클리핑, float32 타입 유지
#     return np.clip(normal.astype(np.float32), 0.0, 1.0)


def depth_to_normal(depth01: np.ndarray,
                    normal_scale: float = 50.0) -> np.ndarray:
    """
    깊이맵(0~1 범위 float32)을 노멀맵(RGB 이미지)으로 변환하는 함수.
    
    Parameters
    ----------
    depth01 : np.ndarray / 정규화된 깊이맵 (0..1 float32)
    normal_scale : float, default=50.0 / 깊이 기울기 -> 노멀 강도 스케일링
    
    Returns
    -------
    np.ndarray / 노멀맵 (0~1 범위 float32)
    """
    # --- (1) 깊이맵 기울기 계산 ---
    # 깊이맵의 가로/세로 방향 기울기(gradient)를 계산
    gy, gx = np.gradient(depth01.astype(np.float32) * normal_scale)
    
    # --- (2) 법선 벡터(nx, ny, nz) 계산 ---
    # nx, ny는 깊이 변화율, nz는 카메라 방향(Z축)
    nx = -gx
    ny = gy # Y축 방향을 반전하지 않아도 일반적으로 노멀맵이 올바르게 생성됨
    nz = np.ones_like(depth01)
    
    # --- (3) 벡터 정규화 (길이=1로 맞추기) ---
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    nx /= norm
    ny /= norm
    nz /= norm
    
    # --- (4) [-1, 1] → [0, 1] 범위로 변환 ---
    # RGB로 매핑: (nx, ny, nz) → (R, G, B)
    # 노멀맵의 값은 -1~1 범위에 있으므로 0~1로 매핑해야 이미지로 저장할 수 있음
    normal = np.stack([
        (nx + 1) * 0.5,
        (ny + 1) * 0.5,
        (nz + 1) * 0.5
    ], axis=-1)
    
    # --- (5) 결과 클리핑 및 반환 ---
    return np.clip(normal, 0.0, 1.0).astype(np.float32)


>>>>>>> 4bdc5bb89478571251b4dbd598614d3d0a91036c

