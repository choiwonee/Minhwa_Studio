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


def compute_normals(depth: np.ndarray) -> np.ndarray:
    """
    깊이맵 → 노멀 벡터 계산 (Sobel 기반)

    Parameters
    ----------
    depth : np.ndarray / 깊이맵 (float32, 0~1 권장)

    Returns
    -------
    np.ndarray / (H, W, 3) 단위 노멀 벡터 맵
                 각 픽셀마다 [nx, ny, nz] 방향 정보를 가짐
    """

    # --- (1) Sobel 필터로 깊이 기울기 계산 ---
    dzdx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)  # x 방향 기울기 (∂z/∂x)
    dzdy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)  # y 방향 기울기 (∂z/∂y)

    # --- (2) 법선 벡터 초기화 ---
    nx, ny = -dzdx, -dzdy              # x,y 성분 (기울기 음수: 관례상 반대 방향)
    nz     = np.ones_like(depth)       # z 성분 (항상 +1, 화면 정면 가정)

    # --- (3) 단위 벡터 정규화 ---
    norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-6  # 벡터 크기(norm), +ε: 0 나눗셈 방지
    nx /= norm; ny /= norm; nz /= norm            # [nx, ny, nz] → 단위 벡터

    # --- (4) 결과 반환 ---
    return np.stack((nx, ny, nz), axis=-1)  # (H, W, 3) 법선 맵 반환


def estimate_light_vector(depth: np.ndarray, img_gray: np.ndarray) -> tuple[float, float, float]:
    """
    깊이맵 + 밝기값 → 광원 벡터 추정 (Least Squares 기반)

    Parameters
    ----------
    depth : np.ndarray / 깊이맵 (float32, 0~1 권장)
    img_gray : np.ndarray / 입력 이미지의 그레이스케일 밝기값 (0~1 float32)

    Returns
    -------
    tuple[float, float, float] / 정규화된 광원 방향 벡터 (lx, ly, lz)
                                절대 세기(intensity)는 포함되지 않고 방향만 표현
    """

    # --- (1) 깊이맵 → 법선 벡터 변환 ---
    normals = compute_normals(depth)  # 각 픽셀의 단위 법선 [nx, ny, nz]

    # --- (2) 벡터/밝기 배열 변환 ---
    h, w, _ = normals.shape
    N = normals.reshape(-1, 3)        # (H*W, 3) 법선 벡터 집합
    I = img_gray.reshape(-1, 1)       # (H*W, 1) 픽셀 밝기(광량)

    # --- (3) 마스크 처리 (어두운 영역 제외) ---
    mask = I[:, 0] > 0.05             # 밝기 ≤ 0.05 → 무시 (노이즈 억제)
    N, I = N[mask], I[mask]           # 유효 픽셀만 사용

    # --- (4) 최소자승법으로 광원 추정 ---
    # I ≈ N · L  → N, I로부터 L(광원 벡터) 근사 추정
    L, _, _, _ = np.linalg.lstsq(N, I, rcond=None)

    # --- (5) 결과 벡터 정규화 ---
    L = L.flatten()                   # (3,1) → (3,)
    L /= np.linalg.norm(L) + 1e-6     # 크기 1로 정규화 (+ε 안전장치)

    # --- (6) 결과 반환 ---
    return tuple(L)  # (lx, ly, lz)


def classify_light_direction(light_dir: np.ndarray) -> str:
    """
    광원 벡터 → 방향 레이블 분류

    Parameters
    ----------
    light_dir : np.ndarray / 광원 방향 벡터 (lx, ly, lz)

    Returns
    -------
    str / 광원 방향 문자열
          예: 'front', 'back', 'upright', 'downleft', 'center' 등
    """

    # --- (1) 벡터 정규화 ---
    light_dir = light_dir / np.linalg.norm(light_dir)  # 크기를 1로 맞춤
    x, y, z = light_dir[0], light_dir[1], light_dir[2]

    # --- (2) Z축이 지배적인 경우 (정면/후면) ---
    if abs(z) > max(abs(x), abs(y)):
        base_direction = "front" if z > 0 else "back"  # z>0 → front, z<0 → back

        # 소량의 X, Y 방향 보정 판단 (임계값 0.05)
        threshold_minor = 0.05

        # 수평 보정
        if x > threshold_minor:
            return f"{base_direction}right"
        elif x < -threshold_minor:
            return f"{base_direction}left"

        # 수직 보정
        if y > threshold_minor:
            return f"{base_direction}up"
        elif y < -threshold_minor:
            return f"{base_direction}down"

        return base_direction

    # --- (3) 측면 지배적인 경우 (X, Y 방향 중심) ---
    else:
        horizontal, vertical = "", ""
        threshold_diag = 0.25  # 대각선 방향 판단 임계값

        # 수직 방향
        if y > threshold_diag:
            vertical = "up"
        elif y < -threshold_diag:
            vertical = "down"

        # 수평 방향
        if x > threshold_diag:
            horizontal = "right"
        elif x < -threshold_diag:
            horizontal = "left"

        # --- (4) 결과 조합 ---
        if vertical and horizontal:       # 대각선 (예: "upright")
            return f"{vertical}{horizontal}"
        elif vertical:                    # 상/하만
            return vertical
        elif horizontal:                  # 좌/우만
            return horizontal
        else:
            return "center"               # 거의 정중앙
        

def phong_shading(normal_rgb01: np.ndarray,
                  light_dir: np.ndarray,
                  ambient: float = 0.5,
                  diffuse_k: float = 0.7,
                  specular_k: float = 0.2,
                  shininess: int = 16) -> np.ndarray:
    """
    노멀맵 + 광원 → Phong 셰이딩 조명 효과

    Parameters
    ----------
    normal_rgb01 : np.ndarray / RGB 노멀맵 (0~1 범위, float32)
    light_dir : np.ndarray / 광원 방향 벡터 (lx, ly, lz)
    ambient : float / 주변광 세기 (기본 0.5)
    diffuse_k : float / 확산광 계수 (기본 0.7)
    specular_k : float / 반사광 계수 (기본 0.2)
    shininess : int / 반사광(하이라이트) 날카로움 정도

    Returns
    -------
    np.ndarray / (H, W, 1) 범위 [0,1]의 셰이딩 결과
    """

    # --- (1) 노멀맵 → [-1,1] 벡터 변환 ---
    nx = normal_rgb01[..., 0] * 2.0 - 1.0
    ny = normal_rgb01[..., 1] * 2.0 - 1.0
    nz = normal_rgb01[..., 2] * 2.0 - 1.0
    n = np.stack([nx, ny, nz], axis=-1)

    # --- (2) 노멀 벡터 정규화 ---
    n_norm = np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9  # 안전장치
    n = n / n_norm

    # --- (3) 광원 벡터 정규화 ---
    L = light_dir.astype(np.float32)
    L = L / (np.linalg.norm(L) + 1e-9)

    # --- (4) 시점(View) 벡터 (정면 고정) ---
    V = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # --- (5) 확산광 계산 (Lambertian) ---
    diff = np.maximum(np.sum(n * L, axis=-1, keepdims=True), 0.0)

    # --- (6) 반사광 계산 (Specular) ---
    R = 2.0 * diff * n - L                                # 반사 벡터
    spec = np.power(np.maximum(np.sum(R * V, axis=-1, keepdims=True), 0.0),
                    shininess)                            # 반사광 세기

    # --- (7) 최종 셰이딩 조합 ---
    shade = ambient + diffuse_k * diff + specular_k * spec

    # --- (8) 결과 반환 ---
    return np.clip(shade.astype(np.float32), 0.0, 1.0)


def depth_shadow_diffusion(rgb01,
                           depth01,
                           strength=0.3,
                           grad_sigma=1.0,
                           grad_spread=5.0,
                           depth_spread=3.0,
                           offset=10,
                           direction='down') -> np.ndarray:
    """
    깊이 기반 그림자 확산 (Depth Shadow Diffusion)

    Parameters
    ----------
    rgb01 : np.ndarray / 원본 RGB 이미지 (0.0~1.0 정규화)
    depth01 : np.ndarray / 깊이맵 (0.0~1.0 정규화)
               값이 작을수록 카메라에 가까움, 클수록 멀어짐
    strength : float / 그림자 강도 (기본 0.3, 범위 0.0~1.0)
                값이 높을수록 그림자가 짙고 어두움
    grad_sigma : float / 경계 부드러움 조절 (기본 1.0)
                 값이 높을수록 경계선 그림자가 더 흐릿함
    grad_spread : float / 경계 확산 정도 (기본 5.0)
                  값이 높을수록 주름이나 경계선을 따라 그림자가 넓게 퍼짐
    depth_spread : float / 깊이 기반 그라데이션 확산 정도 (기본 3.0)
                   값이 높을수록 전체적인 그림자 변화가 부드럽고 완만함
    offset : int / 그림자 이동 거리(픽셀 단위, 기본 10)
              그림자를 특정 방향으로 이동시켜 방향성 부여
    direction : str / 그림자 방향 ('up', 'down', 'left', 'right')
               offset과 함께 사용, 그림자가 드리워지는 방향 결정

    Returns
    -------
    np.ndarray / 그림자 적용된 RGB 이미지 (0.0~1.0 범위)
    """

    # --- (1) 깊이맵 크기 보정 ---
    if depth01.shape[:2] != rgb01.shape[:2]:
        depth01 = cv2.resize(depth01, (rgb01.shape[1], rgb01.shape[0]), interpolation=cv2.INTER_LINEAR)

    # --- (2) 깊이맵을 단일 채널로 변환 ---
    if depth01.ndim == 3:
        depth01 = cv2.cvtColor(depth01, cv2.COLOR_BGR2GRAY) / 255.0

    # --- (3) 경계 기반 그림자 계산 ---
    gx, gy = np.gradient(depth01)
    grad_mag = np.sqrt(gx * gx + gy * gy)                            # 경계 세기
    grad_shadow = cv2.GaussianBlur(grad_mag, (0, 0), grad_sigma)     # 1차 부드러움
    grad_shadow = cv2.GaussianBlur(grad_shadow, (0, 0), grad_spread) # 확산 처리

    # --- (4) 깊이 기반 그라데이션 그림자 ---
    depth_grad = np.clip(1.0 - depth01, 0, 1)
    depth_shadow = cv2.GaussianBlur(depth_grad, (0, 0), depth_spread)

    # --- (5) 기본값: 방향성 없는 그림자 ---
    directional_shadow = depth_shadow

    # --- (6) 그림자 방향 설정 ---
    shift_y, shift_x = 0, 0
    if 'down' in direction:
        shift_y = offset
    elif 'up' in direction:
        shift_y = -offset
    if 'right' in direction:
        shift_x = offset
    elif 'left' in direction:
        shift_x = -offset

    # --- (7) 방향성 그림자 적용 ---
    if shift_y != 0 or shift_x != 0:
        directional_shadow = np.roll(depth_shadow, shift=(shift_y, shift_x), axis=(0, 1))

    # --- (8) 최종 그림자 합성 ---
    shadow = np.maximum(grad_shadow, directional_shadow)
    shadow = np.expand_dims(shadow, axis=-1)
    shaded = rgb01 - strength * shadow

    # --- (9) 결과 반환 ---
    return np.clip(shaded, 0.0, 1.0)


def blend_image(base_rgb01: np.ndarray,
                shade01: np.ndarray,
                mode: str = "softlight",
                alpha: float = 0.5) -> np.ndarray:
    """
    셰이딩 이미지를 원본 RGB 이미지와 합성 (Blend)

    Parameters
    ----------
    base_rgb01 : np.ndarray / 원본 RGB 이미지 (0.0~1.0 또는 0~255)
    shade01 : np.ndarray / 셰이딩 이미지 (단일채널 or 3채널)
    mode : str / 합성 모드
            'multiply' | 'screen' | 'softlight' | 'overlay'
            기본값: 'softlight' (포토샵 유사 소프트라이트 효과)
    alpha : float / 블렌딩 비율 (0.0~1.0)
             0.0 → 원본 유지, 1.0 → 셰이딩 100%

    Returns
    -------
    np.ndarray / 블렌딩된 최종 RGB 이미지 (0.0~1.0 범위)
    """

    # --- (1) 기본 이미지 정규화 ---
    base = base_rgb01.astype(np.float32)
    if base.max() > 1.0:  # 255 스케일 → 0~1로 변환
        base /= 255.0

    # --- (2) 셰이딩 이미지 채널 정규화 ---
    shade3 = np.repeat(shade01, 3, axis=-1) if shade01.ndim == 2 or shade01.shape[-1] == 1 else shade01
    if shade3.max() > 1.0:
        shade3 /= 255.0

    # --- (3) 단일 채널 → 3채널 변환 ---
    if shade01.ndim == 2:
        shade3 = np.repeat(shade01[:, :, np.newaxis], 3, axis=-1)
    else:
        shade3 = shade01.astype(np.float32)

    # --- (4) 이미지 크기 일치화 ---
    if base.shape[:2] != shade3.shape[:2]:
        shade3 = cv2.resize(shade3, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)

    # --- (5) 블렌딩 모드별 계산 ---
    if mode.lower() == "multiply":
        blended = base * shade3                                     # 어둡게 합성
    elif mode.lower() == "screen":
        blended = 1.0 - (1.0 - base) * (1.0 - shade3)               # 밝게 합성
    elif mode.lower() == "softlight":
        blended = (1 - 2 * shade3) * base * base + 2 * shade3 * base # 포토샵 유사 Softlight 근사식
    elif mode == "overlay":
        blended = np.where(
            base < 0.5,
            2 * base * shade3,                                      # 어두운 부분 강조
            1 - 2 * (1 - base) * (1 - shade3)                       # 밝은 부분 강조
        )
    else:
        raise ValueError(f"Unsupported blend mode: {mode}")

    # --- (6) 알파 블렌딩 (최종 합성 비율 적용) ---
    out = (1 - alpha) * base + alpha * blended

    # --- (7) 결과 반환 ---
    return np.clip(out, 0.0, 1.0)


def depth_contrast_boost(rgb01,
                         depth01,
                         strength=0.1,
                         in_place: bool = False) -> np.ndarray:
    """
    깊이 기반 대비(Contrast) 강화

    Parameters
    ----------
    rgb01 : np.ndarray / 입력 RGB 이미지 (0.0~1.0 정규화)
    depth01 : np.ndarray / 깊이맵 (0.0~1.0 정규화)
               값이 작을수록 가까움, 클수록 멀어짐
    strength : float / 대비 강화 강도 (기본 0.1, 범위 0.0~1.0)
                값이 높을수록 가까운 영역의 대비가 더 강해짐

    Returns
    -------
    np.ndarray / 대비 강화가 적용된 RGB 이미지 (0.0~1.0 범위)
    """

    # --- (1) 강도 유효성 검사 ---
    if strength is None or strength <= 0:
        return rgb01  # strength=0이면 원본 그대로 반환

    # --- (2) 깊이맵 크기 보정 ---
    if depth01.shape[:2] != rgb01.shape[:2]:
        depth01 = cv2.resize(depth01, (rgb01.shape[1], rgb01.shape[0]), interpolation=cv2.INTER_LINEAR)

    target = rgb01 if in_place else rgb01.copy()

    # --- (3) 깊이 기반 가중치 생성 ---
    depth_weight = cv2.GaussianBlur(depth01, (0, 0), 3)  # 부드럽게 블러링
    depth_weight = (1.0 - depth_weight)                  # 가까운 영역(깊이 작음)을 강조
    depth_weight = np.expand_dims(depth_weight, axis=-1) # 채널 차원 확장

    mean = target.mean(axis=(0, 1), keepdims=True)

    # --- (4) 대비 강화 연산 ---
    diff = target - mean
    diff *= depth_weight
    diff *= strength
    
    target += diff

    np.clip(target, 0.0, 1.0, out=target)

    # --- (5) 결과 반환 ---
    return target


def depth_shadow_boost(rgb01,
                       depth01,
                       strength=0.3,
                       in_place: bool = False) -> np.ndarray:
    """
    깊이 기반 그림자(Shadow) 강화

    Parameters
    ----------
    rgb01 : np.ndarray / 입력 RGB 이미지 (0.0~1.0 정규화)
    depth01 : np.ndarray / 깊이맵 (0.0~1.0 정규화)
               값이 작을수록 가까움, 클수록 멀어짐
    strength : float / 그림자 강도 (기본 0.3)
               값이 너무 크면 이미지가 지나치게 어두워질 수 있음

    Returns
    -------
    np.ndarray / 깊이 경사 기반의 그림자가 적용된 RGB 이미지 (0.0~1.0 범위)
    """

    # --- (1) 강도 유효성 검사 ---
    if strength is None or strength <= 0:
        return rgb01  # strength=0이면 원본 그대로 반환

    # --- (2) 깊이맵 크기 보정 ---
    if depth01.shape[:2] != rgb01.shape[:2]:
        depth01 = cv2.resize(depth01, (rgb01.shape[1], rgb01.shape[0]), interpolation=cv2.INTER_LINEAR)

    target = rgb01 if in_place else rgb01.copy()

    # --- (3) 깊이 변화(Gradient) 계산 ---
    gx, gy = np.gradient(depth01)
    grad_mag = np.sqrt(gx * gx + gy * gy)          # 깊이 변화량 (경사 크기)
    shadow = cv2.GaussianBlur(grad_mag, (0, 0), 1.0)  # 부드럽게 확산
    shadow = np.expand_dims(shadow, axis=-1)       # 채널 차원 확장

    # --- (4) 그림자 적용 ---
    target -= (shadow * strength)
    np.clip(target, 0.0, 1.0, out=target)          # 경사 강한 곳에 어두운 음영 추가

    # --- (5) 결과 반환 ---
    return target


def highlight_boost(rgb01,
                    strength=0.1,
                    in_place: bool = False) -> np.ndarray:
    """
    하이라이트(밝은 영역) 강화

    Parameters
    ----------
    rgb01 : np.ndarray / 입력 RGB 이미지 (0.0~1.0 정규화, float32)
    strength : float / 하이라이트 강화 비율 (기본 0.1)
                값이 너무 크면 이미지가 과도하게 밝아질 수 있음

    Returns
    -------
    np.ndarray / 밝은 영역이 강조된 RGB 이미지 (0.0~1.0 범위)
    """

    # --- (1) 강도 유효성 검사 ---
    if strength is None or strength <= 0:
        return rgb01  # strength=0이면 원본 그대로 반환

    # --- (2) 밝기 계산 ---
    brightness = rgb01.mean(axis=2)  # 픽셀 평균값으로 밝기(0~1) 계산

    # --- (3) 하이라이트 마스크 생성 ---
    # 0.9 이상인 밝은 영역만 추출 → (밝기 - 0.9)*3 비율로 강조
    highlight_mask = np.clip((brightness - 0.9) * 3, 0.0, 1.0)

    # --- (4) 하이라이트 부스트 적용 ---
    target = rgb01 if in_place else rgb01.copy()
    target += (highlight_mask[..., None] * float(strength))

    np.clip(target, 0.0, 1.0, out=target)

    # --- (5) 결과 반환 ---
    return target