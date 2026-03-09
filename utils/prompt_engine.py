""" utils/prompt_engine.py
──────────────────────────────────────────────────────
    : Traditional_Art_Project 의 프롬프트 처리 통합 모듈.
    1) 담당 기능
        1. 한국어 → 영어 번역 (translator 연동)
        2. <camera> 태그 파싱 및 96 가지 앵글 처리
        3. 모델(Provider)별 최적화된 프롬프트 빌드
        - Gemini  : INSTRUCTION 지시문 구조
        - Qwen : <sks> 트리거 + LoRA vocabulary 공백 구분 캡션 구조
        - 범용     : 단순 묘사형
        4. prompt_mode(mix_way / one_way / two_way) 처리
        5. bg_composer.py 의 기존 메서드 drop-in 교체 지원

    2) 설계 원칙
        - 번역 전에 <camera> 태그를 분리·보존한다. (번역기가 태그 내부 수치를 변환·제거하는 버그 방지)
        - 수평 앵글을 8 분할로 처리하여 대각선 구도 누락을 막는다.
        - 모든 앵글 판단은 _classify() 하나로 집중시켜 일관성을 보장한다.

    3) 통합 방법 (bg_composer.py)
        # __init__() 끝 부분에 추가:
            from utils.prompt_engine import PromptEngine
            self._prompt_engine = PromptEngine()

        # run_generation() 의 기존 프롬프트 처리 블록 전체를 교체:
            p_txt, n_txt = self._prompt_engine.process(
                p_raw       = self.txt_prompt.toPlainText().strip(),
                n_raw       = self.txt_negative.toPlainText().strip(),
                manual_mode = self.chk_manual_prompt.isChecked(),
                multi_angle = self.chk_multi_angle.isChecked(),
                model_cfg   = self._active_model_config,
            )
"""

# ---------------------------------------------------------------------------
# Python 기본 문법에 따라 __future__ import는 일반 import보다 최상단에 위치해야 한다.
# ---------------------------------------------------------------------------
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# 0. 번역기 연동 (없으면 passthrough)
# ---------------------------------------------------------------------------
try:
    from utils.translator import translator as _translator

    def _translate(text: str) -> str:
        return _translator.translate(text) if text else text

except ImportError:
    def _translate(text: str) -> str:
        return text

# ---------------------------------------------------------------------------
# 1. 카메라 파라미터 데이터 클래스
# ---------------------------------------------------------------------------
@dataclass
class CameraParams:
    """<camera> 태그에서 파싱된 카메라 파라미터."""
    horizontal: float = 0.0    # -180 ~ +180  (양수 = 카메라가 피사체 우측, 음수 = 카메라가 피사체 좌측, ±180 = 정후방)
    vertical:   float = 0.0    # -90 ~ +90  (양수 = 카메라가 피사체 위에 위치(high-angle), 음수 = 카메라가 피사체 아래에 위치(low-angle))
    zoom:       float = 1.0    # 0.5  ~ 2.0   (1.0=표준)
    roll:       float = 0.0    # Dutch angle (선택)

# ---------------------------------------------------------------------------
# 2. 앵글 분류 테이블
#    형식: (범위_하한, 범위_상한, 자세한_묘사, 짧은_태그, 각도_레이블)
# ---------------------------------------------------------------------------

# 수평(Azimuth) 8분할 ───────────────────────────────────────────────────
#   기존 코드의 4단계(front/left side/right side/back)에서 대각선 4방향 추가.
#   ※ abs(h)>=120 이 h>=30 조건에 이미 걸려 back view 가 생성되지 않던 버그 수정.
# (카메라 위치 기준: 슬라이더 양수(+) = 카메라가 피사체의 우측에 위치, LoRA vocabulary 일치)
_H_BINS = [
    # 경계값 정책: 각 bin은 (하한 포함, 상한 미포함) 반개방 구간 [lo, hi) 으로 처리한다.
    # 단, 절댓값 최댓값인 ±180은 예외적으로 back view 에 포함시킨다.
    # _classify() 의 조건식을 lo <= value < hi 로 변경하여 중복 매칭을 원천 차단한다.
    (-22.5,   22.5, "directly in front of the subject",    "front view",               "0°"   ),
    ( 22.5,   67.5, "from the front-right of the subject", "front-right quarter view", "45°R" ), # 양수 = RIGHT
    ( 67.5,  112.5, "from the right side of the subject",  "right side view",          "90°R" ),  
    (112.5,  157.5, "from the back-right of the subject",  "back-right quarter view",  "135°R"),  
    (157.5,  180.0, "from directly behind the subject",    "back view",                "180°" ),  
    (-180.0,-157.5, "from directly behind the subject",    "back view",                "180°" ),  
    (-157.5,-112.5, "from the back-left of the subject",   "back-left quarter view",   "135°L"),  
    (-112.5, -67.5, "from the left side of the subject",   "left side view",           "90°L" ),  
    ( -67.5, -22.5, "from the front-left of the subject",  "front-left quarter view",  "45°L" ), # 음수 = LEFT
]

# 수직(Elevation) 4분할 (LoRA vocabulary: low-angle / eye-level / elevated / high-angle), 음수 구간에 high-angle 배치 (태그가 -v이므로)
_V_BINS = [
    (-90.0, -45.0, "high-angle shot looking downward",     "high-angle shot", "high"),
    (-45.0, -15.0, "slightly elevated shot from above",    "elevated shot",   "elevated"),
    (-15.0,  15.0, "straight eye-level shot",              "eye-level shot",  "eye-level"),
    ( 15.0,  90.0, "low-angle shot looking upward",        "low-angle shot",  "low"),
]

# 줌(Distance) 3분할 ────────────────────────────────────────────────────
_Z_BINS = [
    (0.0,  0.75, "a wide establishing shot",   "wide shot",  "wide"    ),
    (0.75, 1.25, "a medium shot",              "medium shot","medium"  ),
    (1.25, 9.9,  "a tight close-up shot",      "close-up",   "close-up"),
]

def _classify(value: float, bins: list) -> dict:
    """ value 가 속하는 bin 을 찾아 dict 로 반환한다.

        [구간 처리 규칙]
        - 기본: 반개방 구간 [lo, hi) — 하한 포함, 상한 미포함.
          → 경계값 중복 매칭을 원천 차단한다.
        - 예외: lo 또는 hi 가 절댓값 최댓값인 경우 닫힌 구간 [lo, hi] 로 처리한다.
          · 상한 최댓값: hi in (180.0, 90.0, 9.9)  → 상한 포함 (ex. back view 180°, high-angle shot 90°)
          · 하한 최솟값: lo in (-180.0, -90.0)     → 하한 포함 (ex. back view -180°, low-angle shot -90°)

        [폴백]
        어느 구간에도 해당하지 않으면 bin 의 하한·상한 경계값까지의 거리가
        가장 짧은 bin 을 반환한다.
    """
    for lo, hi, verbose, short, tag in bins:
        _is_max_boundary = (hi in (180.0, 90.0, 9.9)) or (lo in (-180.0, -90.0))
        if _is_max_boundary:
            if lo <= value <= hi:
                return {"verbose": verbose, "short": short, "tag": tag}
        else:
            if lo <= value < hi:
                return {"verbose": verbose, "short": short, "tag": tag}
    # 폴백: 거리 기준으로 가장 가까운 bin 선택
    best = min(bins, key=lambda b: min(abs(value - b[0]), abs(value - b[1])))
    return {"verbose": best[2], "short": best[3], "tag": best[4]}

def classify_camera(p: CameraParams) -> dict:
    """ CameraParams → 분류 결과 dict(h, v, z, params) 반환. """
    return {
        "h": _classify(p.horizontal, _H_BINS),
        "v": _classify(p.vertical,   _V_BINS),
        "z": _classify(p.zoom,       _Z_BINS),
        "params": p,
    }

# ---------------------------------------------------------------------------
# 3. <camera> 태그 파싱
# ---------------------------------------------------------------------------
_CAM_TAG_RE = re.compile(r"<camera>(.*?)</camera>", re.DOTALL | re.IGNORECASE)
# 패턴: <sks> 뒤에 최대 3개 공백·쉼표 구분 토큰만 캡처 (탐욕 매칭 방지), LoRA 포맷 예: "<sks> front view eye-level shot medium shot"
_SKS_ANGLE_RE = re.compile(r'(<sks>\s*[^,\n<]+(?:,\s*[^,\n<]+){0,2})', re.I)

def _extract_camera_block(prompt: str) -> Tuple[str, Optional[str]]:
    """ 프롬프트에서 <camera>...</camera> 블록을 분리한다.
        1) Returns:
            (태그를_제거한_프롬프트, 원본_camera_블록_문자열 또는 None)
    """
    m = _CAM_TAG_RE.search(prompt)
    if not m:
        return prompt, None
    clean = _CAM_TAG_RE.sub("", prompt).strip()
    return clean, m.group(0)

def parse_camera_tag(prompt: str) -> Tuple[Optional[re.Match], CameraParams]:
    """기존 _parse_camera_tag() 와 동일한 인터페이스 (하위호환용)."""
    m = _CAM_TAG_RE.search(prompt)
    if not m:
        return None, CameraParams()

    body = m.group(1)

    def _get(key: str, default: float) -> float:
        found = re.search(rf"{key}\s*[=:]\s*([-\d\.]+)", body, re.I)
        return float(found.group(1)) if found else default

    return m, CameraParams(
        horizontal=_get("horizontal", 0.0),
        vertical  =_get("vertical",   0.0),
        zoom      =_get("zoom",       1.0),
        roll      =_get("roll",       0.0),
    )

def _parse_params_from_block(camera_block: str) -> CameraParams:
    """camera_block 문자열에서 CameraParams 파싱 (내부 전용)."""
    _, p = parse_camera_tag(camera_block)
    return p

# ---------------------------------------------------------------------------
# 4. 모델별 프롬프트 빌더 (모델 특성에 맞는 최적화 프롬프트 생성 + 실사화 강력 억제)
# ---------------------------------------------------------------------------
class _GeminiBuilder:
    def build(self, translated_prompt: str, cam: dict, rag_data: dict = None) -> str:
        """ Gemini 모델용 프롬프트 템플릿 빌드
            - RAG 미사용 시에도 앵글 조작이 있으면 능동적인 포즈 재계산 지시문을 자동 주입함.
        """
        h, v, z = cam["h"], cam["v"], cam["z"]
        p: CameraParams = cam["params"]
        
        has_angle = abs(p.horizontal) > 0.1 or abs(p.vertical) > 0.1 or abs(p.zoom - 1.0) > 0.05
        cam_text = f"Azimuth: {h['short']}, Elevation: {v['short']}, Distance: {z['short']}" if has_angle else "" # Distance(Zoom), 결과 예: "Azimuth: front view, Elevation: eye-level shot, Distance: medium shot"

        # ── I2I RAG 모드: 이미지 분석 기반 경량 instruction ────────────────
        # style_anchors/negative 를 주입하지 않고 change + keep 만 사용.
        # 모델이 입력 이미지를 직접 보고 있으므로 스타일 재지정은 오히려 충돌을 유발함.
        if rag_data and rag_data.get("mode") == "i2i":
            # [수정] LLM 응답 텍스트 끝에 포함된 마침표(.) 제거하여 중복 마침표(..) 방지
            change = rag_data.get("change", translated_prompt).strip().rstrip('.')
            keep   = rag_data.get("keep", "").strip().rstrip('.')
            keep_clause = f" Preserve {keep}." if keep else ""
            if has_angle:
                return (
                    f"Redraw from this camera position to {cam_text}. "
                    f"{change}.{keep_clause}"
                ).strip()
            return f"{change}.{keep_clause}".strip()
        # ── I2I RAG 모드 끝 ──────────────────────────────────────────────

        # 💡 보완된 부분: RAG 미사용 시 앵글 단독 지시 처리 강화
        if not rag_data:
            if has_angle:
                # 명확한 카메라 위치 기준 명시과 포즈 재계산 강제 지시문 추가로 앵글 변경 시 실사화 억제 및 자연스러운 포즈 조정 유도
                action_instruction = (
                    f"Completely redraw the subject's pose as seen from this camera position to "
                    f"{cam_text}. "
                    f"The camera is positioned {h['verbose']} — "
                    f"redraw the subject's body accordingly so they appear natural from this viewpoint."
                )
                if translated_prompt:
                    return f"{action_instruction} Additional edits: {translated_prompt}".strip()
                return action_instruction
            else:
                return translated_prompt.strip()

        # 일반 RAG 모드에서도 중복 마침표 방지를 위해 rstrip('.') 적용
        keep = rag_data.get("keep", "Maintain original composition and empty spaces.").strip().rstrip('.')
        change = rag_data.get("change", translated_prompt).strip().rstrip('.')
        add = rag_data.get("add", "none").strip().rstrip('.')
        style = rag_data.get("style_anchors", "traditional Korean minhwa painting style").strip().rstrip('.')

        # 앵글 변경 시 구도 보존(compositional weight) 제약을 완전히 삭제하고, 포즈 재계산을 강제함
        if has_angle:
            preservation_target = "the brushwork texture, ink wash gradients, silk or hanji paper grain, and mineral pigment palette"
            l0_rules = "You MUST entirely recalculate the composition and character pose to match the new camera direction. Treat the original image ONLY as a character/style reference, NOT a strict pose guide."
            l1_header = "[L1] ADAPT THESE ELEMENTS TO THE NEW CAMERA ANGLE (Redraw pose and silhouette naturally):"
            l5_style = f"{style}, decorative symmetry"
            change = f"{change}, AND completely redraw the subject's pose to match the {cam_text} perspective."
        else:
            preservation_target = "the brushwork texture, ink wash gradients, silk or hanji paper grain, mineral pigment palette, and the overall compositional weight and balance"
            l0_rules = "Do NOT apply any photorealistic rendering, digital smoothing, or Western perspective correction."
            l1_header = "[L1] PRESERVE THESE ELEMENTS EXACTLY:"
            l5_style = f"{style}, flat perspective without vanishing point, decorative symmetry"

        template = f"""[L0] OVERALL STYLE & PRESERVATION:
                    You are editing a traditional East Asian painting. 
                    The following baseline elements must remain COMPLETELY UNCHANGED: {preservation_target}. {l0_rules}

                    {l1_header}
                    {keep}

                    [L2] CAMERA DIRECTION:
                    {cam_text if cam_text else "Stable centered frontal view"}

                    [L3] MAKE THESE SPECIFIC CHANGES ONLY:
                    {change}

                    [L4] ADD THE FOLLOWING NEW ELEMENTS:
                    {add}

                    [L5] APPLY THESE STYLE ANCHORS TO THE FINAL IMAGE:
                    Synthesize the entire image using these specific domain references: {l5_style}."""
        return template

class _QwenBuilder:
    def build(self, translated_prompt: str, cam: dict, rag_data: dict = None, is_lora: bool = False) -> str:
        """ Qwen 모델용 프롬프트 템플릿 빌드 (LoRA 여부에 따른 동적 분기)
            - is_lora == True : <sks> 트리거 + 공백 구분 LoRA vocabulary 포맷 캡션 생성. 결과 예: <sks> front view eye-level shot medium shot.
            - is_lora == False: <sks> 태그 없이 자연어 동사형(Redraw 등) 지시문 생성.
        """
        h, v, z = cam["h"], cam["v"], cam["z"]
        p: CameraParams = cam["params"]
        
        has_angle = abs(p.horizontal) > 0.1 or abs(p.vertical) > 0.1 or abs(p.zoom - 1.0) > 0.05
        
        # LoRA 여부에 따른 앵글 문구 및 트리거 태그 분리, (공백 구분, v short tag에 shot 이미 포함), 결과(예): "front view eye-level shot medium shot"
        angle_phrase = f"{h['short']} {v['short']} {z['short']}" if has_angle else "" 
        sks = "<sks> " if is_lora else ""

        # ── I2I RAG 모드 ───────────────────────────────────────────────────
        if rag_data and rag_data.get("mode") == "i2i":
            # 마침표 제거 및 첫 글자 대문자화로 자연스러운 문장 구조 형성
            change = rag_data.get("change", translated_prompt).strip().rstrip('.')
            keep   = rag_data.get("keep", "").strip().rstrip('.')
            keep_clause = f", strictly keeping {keep} unchanged" if keep else ""
            
            if change:
                change = change[0].upper() + change[1:]
            
            if has_angle:
                if is_lora:
                    # LoRA 규격 강제 적용: 앵글태그 나열 후 마침표(.) 추가, 그 뒤 지시문 연결
                    return f"{sks}{angle_phrase}. {change}{keep_clause}."
                else:
                    return f"Redraw to {angle_phrase}. {change}{keep_clause}."
            return f"{sks}{change}{keep_clause}."
        # ── I2I RAG 모드 끝 ───────────────────────────────────────────────        

        # 1. RAG 미사용 시
        if not rag_data:
            # [수정] 번역된 프롬프트의 기존 마침표를 미리 제거하여 중복(..) 방지
            translated_prompt = translated_prompt.strip().rstrip('.') if translated_prompt else ""
            
            if has_angle:
                if is_lora:
                    if translated_prompt:
                        translated_prompt = translated_prompt[0].upper() + translated_prompt[1:]
                        # angle_phrase 뒤에 마침표(.)를 명시적으로 추가하여 분리
                        return f"{sks}{angle_phrase}. {translated_prompt}."
                    return f"{sks}{angle_phrase}.".strip()
                else:
                    action_instruction = f"Redraw the subject completely to a {angle_phrase}."
                    if translated_prompt:
                        translated_prompt = translated_prompt[0].upper() + translated_prompt[1:]
                        return f"{action_instruction} Change the image to {translated_prompt}."
                    return action_instruction
            else:
                if translated_prompt:
                    translated_prompt = translated_prompt[0].upper() + translated_prompt[1:]
                    return f"{sks}{translated_prompt}."
                return sks.strip()

        # 2. RAG 사용 시 (T2I)
        keep = rag_data.get("keep", "original elements").strip().rstrip('.')
        change = rag_data.get("change", translated_prompt).strip().rstrip('.')
        add = rag_data.get("add", "").strip().rstrip('.')
        style = rag_data.get("style_anchors", "traditional Korean minhwa, joseon genre painting").strip().rstrip('.')

        add_str = f" Also add {add}." if add and add.lower() != "none" else ""
        
        if change:
            change = change[0].upper() + change[1:]

        # LoRA 특성 및 일반 Qwen 특성에 맞춘 코어 프롬프트 생성
        if is_lora:
            if has_angle:
                # RAG change 필드에서 불필요한 카메라/앵글 관련 문장 제거 후 앵글 태그를 최전방 배치
                change_clean = re.sub(
                    r'(adjust|change|represent|from above|high-angle|low-angle|vertical|horizontal)[^.]*\.?\s*',
                    '', change, flags=re.IGNORECASE
                ).strip().strip(',').strip()
                
                if change_clean:
                    change_clean = change_clean[0].upper() + change_clean[1:]
                    
                # angle_phrase 뒤에 마침표(.) 추가하여 모든 모드에서 통일성 유지
                prompt_core = f"{angle_phrase}. Change the image to {change_clean}, strictly maintaining {keep}.{add_str}"
            else:
                prompt_core = f"Change the image to {change}, strictly keeping the 2D shape of {keep} unchanged.{add_str}"
        else:
            if has_angle:
                prompt_core = f"Redraw the subject completely to a {angle_phrase}. Change the image to {change}, while strictly maintaining {keep}.{add_str}"
            else:
                prompt_core = f"Change the image to {change}, while strictly keeping the 2D shape of {keep} unchanged.{add_str}"

        # 앵글 모순 필터링용 룰
        no_rules = "photorealistic shading, depth of field blur, digital smoothing, anachronistic elements" if has_angle else "photorealistic shading, depth of field blur, Western linear perspective, digital smoothing, anachronistic elements"

        template = f"""{sks}{prompt_core} Render in {style}, decorative symmetry. 
                    NO: {no_rules}"""
        return template.strip()
    
class _GenericBuilder:
    def build(self, translated_prompt: str, cam: dict, rag_data: dict = None) -> str:
        """ 범용 모델 프롬프트 템플릿 빌드 """
        h, v, z = cam["h"], cam["v"], cam["z"]
        p: CameraParams = cam["params"]
        
        has_angle = abs(p.horizontal) > 0.1 or abs(p.vertical) > 0.1 or abs(p.zoom - 1.0) > 0.05
        angle_str = f"{h['short']} {v['short']} {z['short']}" if has_angle else "" # 결과 예: "front view eye-level shot medium shot"
        
        if rag_data and rag_data.get("mode") == "i2i":
            # I2I 모드: style_anchors 없이 change + keep 만 사용
            change = rag_data.get("change", translated_prompt)
            keep   = rag_data.get("keep", "")
            parts  = [change, f"keep {keep}" if keep else "", angle_str]
        elif rag_data:
            parts = [rag_data.get('change', translated_prompt), rag_data.get('style_anchors', ''), angle_str]
        else:
            parts = [translated_prompt, angle_str]

        return ", ".join(p for p in parts if p).strip(", ")
    
# ---------------------------------------------------------------------------
# 5. 프롬프트 모드 처리
# ---------------------------------------------------------------------------
def apply_prompt_mode(positive: str, negative: str, mode: str) -> dict:
    """ config.ini 의 prompt_mode 에 따라 최종 프롬프트 딕셔너리를 생성한다.
        1) Args:
            positive: 긍정 프롬프트 (영어, 번역 완료)
            negative: 부정 프롬프트 (영어, 번역 완료)
            mode:     "mix_way" | "one_way" | "two_way"
        2) Returns:
            {"prompt": ..., "negative_prompt": ...} 형식의 dict.
            mix_way / one_way 는 negative_prompt 키 없음.
    """
    mode = (mode or "two_way").lower().strip()

    if mode == "mix_way":
        combined = positive
        if negative and negative.strip():
            combined = f"{positive}. Avoid: {negative}"
        return {"prompt": combined}

    if mode == "one_way":
        return {"prompt": positive}

    # two_way (기본)
    return {"prompt": positive, "negative_prompt": negative}

# ---------------------------------------------------------------------------
# 6. 통합 PromptEngine (공개 API)
# ---------------------------------------------------------------------------
class PromptEngine:
    """ bg_composer.py 에서 직접 사용하는 프롬프트 통합 처리 엔진.
        1) Quick Start
        ──────────────────────────────────────────────────────
            # __init__() 에서:
                self._prompt_engine = PromptEngine()

            # run_generation() 에서:
                p_txt, n_txt = self._prompt_engine.process(
                    p_raw       = self.txt_prompt.toPlainText().strip(),
                    n_raw       = self.txt_negative.toPlainText().strip(),
                    manual_mode = self.chk_manual_prompt.isChecked(),
                    multi_angle = self.chk_multi_angle.isChecked(),
                    model_cfg   = self._active_model_config,
                )
    """

    def __init__(self):
        self._gemini  = _GeminiBuilder()
        self._qwen    = _QwenBuilder()
        self._generic = _GenericBuilder()

    # 메인 진입점 ──────────────────────────────────────────────────────
    def process(
        self,
        p_raw:          str,
        n_raw:          str  = "",
        manual_mode:    bool = False,
        multi_angle:    bool = False,
        model_cfg:      dict = None,
        use_translator: bool = True,
        rag_data:       dict = None,
    ) -> Tuple[str, str]:
        """ 프롬프트 통합 처리 메인 진입점
            Args:
                p_raw          : 원시 긍정 프롬프트 (한국어 가능, <camera>/<sks> 포함 가능)
                n_raw          : 원시 부정 프롬프트
                manual_mode    : True 이면 use_translator 와 무관하게 번역·앵글 변환을 모두 건너뛴다
                multi_angle    : True 일 때만 <camera> 태그를 앵글 지시문으로 변환한다
                model_cfg      : config.ini 에서 로드된 모델 설정 dict
                use_translator : False 이면 번역을 건너뛴다. manual_mode=True 이면 이 값은 무시된다.
                rag_data       : RAGPrompter 가 반환한 분석 결과 dict (없으면 None)
        """
        model_cfg = model_cfg or {}
        sks_match  = _SKS_ANGLE_RE.search(p_raw)
        sks_prefix = sks_match.group(1).strip() if sks_match else ""
        
        if sks_prefix:
            p_raw_clean = _SKS_ANGLE_RE.sub("", p_raw).strip().lstrip(", \t")
        else:
            p_raw_clean = p_raw
        
        # 1. <camera> 태그 분리 (어떤 모드에서든 앵글 처리를 위해 선행)
        p_no_tag, cam_block = _extract_camera_block(p_raw_clean)

        # 2. 번역 및 수동 모드 처리
        if manual_mode: # manual_mode=True 이면 use_translator 설정과 무관하게 번역·변환을 모두 건너뛴다.
            p_txt = p_no_tag # ← use_translator 무시
            n_txt = n_raw    # ← n_raw도 번역 안 함
        else:
            # rag_data 가 있을 때는 RAGPrompter 가 이미 영어로 분석을 완료했으므로 번역을 건너뛴다.
            p_txt = p_no_tag if rag_data else (_translate(p_no_tag) if use_translator else p_no_tag)
            n_txt = _translate(n_raw) if use_translator else n_raw

        # 3. 템플릿 및 앵글 적용
        # multi_angle=False 이면 cam_block 을 무시하여 앵글 지시문을 생성하지 않는다.
        effective_cam_block = cam_block if multi_angle else None
        params = _parse_params_from_block(effective_cam_block) if effective_cam_block else CameraParams()
        cam = classify_camera(params)
        
        provider = model_cfg.get("provider", "").lower()
        pipe_type = model_cfg.get("pipeline_type", "").lower()
        
        # config.ini에 정의된 모델 URI나 Repo ID에 'lora' 키워드가 있는지 동적 확인
        repo_id = model_cfg.get("repo_id", "").lower()
        api_uri = model_cfg.get("api_model_uri", "").lower()
        is_lora = "lora" in repo_id or "lora" in api_uri

        if "google_genai" in provider:
            p_txt = self._gemini.build(p_txt, cam, rag_data)
        elif "fal_ai" in provider or "qwen" in pipe_type:
            # 빌더에 is_lora 플래그를 넘겨 동적 분기 처리
            p_txt = self._qwen.build(p_txt, cam, rag_data, is_lora=is_lora)
        else:
            p_txt = self._generic.build(p_txt, cam, rag_data)

        # Negative 병합 (RAG에서 분석된 negative 요소 추가)
        if rag_data and rag_data.get("negative"):
            n_txt = f"{n_txt}, {rag_data['negative']}".strip(", ")

        # 4. 앵글 모순 필터링 (입체 앵글 적용 시 평면성 강제 키워드 삭제)
        has_angle = abs(params.horizontal) > 0.1 or abs(params.vertical) > 0.1 or abs(params.zoom - 1.0) > 0.05
        if has_angle and n_txt:
            n_txt = re.sub(r"(?i)\bWestern perspective\b", "", n_txt) 
            n_txt = re.sub(r",\s*,", ",", n_txt).strip(" ,") 

        # 5. sks_prefix 복원 (직접 입력된 <sks> 앵글 명령 보존)
        # cam_block이 없을 때만 복원한다 (cam_block이 있으면 QwenBuilder가 이미 <sks>를 생성함)
        # 이중 <sks> 방지: p_txt에 이미 <sks>가 있으면 sks_prefix의 <sks>를 제거하고 앵글 부분만 앞에 붙인다
        if sks_prefix and not effective_cam_block:
            if re.match(r'<sks>', p_txt.lstrip(), re.I): # p_txt에 이미 <sks>가 있는지 대소문자 무관하게 확인
                # QwenBuilder가 이미 <sks>를 추가한 경우: sks_prefix에서 앵글 텍스트만 추출하여 교체
                angle_only = re.sub(r'^<sks>\s*', '', sks_prefix, flags=re.I).strip()
                # 기존 <sks> 뒤에 앵글 텍스트를 삽입
                p_txt = re.sub(r'^(<sks>\s*)', rf'\1{angle_only} ', p_txt.lstrip(), flags=re.I)
            else:
                p_txt = f"{sks_prefix} {p_txt}".strip()
        
        return p_txt, n_txt
    
    def translate_only(self, text: str) -> str:
        """ Preview UI 출력을 위한 단순 번역 헬퍼 """
        p_no_tag, _ = _extract_camera_block(text)
        return _translate(p_no_tag)

    def build_payload(
        self,
        p_txt:     str,
        n_txt:     str,
        model_cfg: dict = None,
    ) -> dict:
        """ process() 결과를 받아 모델이 요구하는 prompt_mode 형식으로 변환한다.
            Returns:
                {"prompt": ..., "negative_prompt": ...} 형식 dict
        """
        model_cfg = model_cfg or {}
        mode = model_cfg.get("prompt_mode", "two_way")
        return apply_prompt_mode(p_txt, n_txt, mode)

    # 앵글 변환 내부 메서드 ────────────────────────────────────────────
    def _apply_angle(self, translated_prompt: str, cam_block: str, model_cfg: dict) -> str:
        """ 번역 완료된 프롬프트 + 보존된 <camera> 블록으로 앵글 지시문을 생성한다.
            1) 기존 코드 버그 수정 포인트
            ────────────────────────────────────────────
            - 기존: <camera> 태그가 있는지를 p_raw 로 확인하고 변환은 이미 번역된 p_txt 에 적용
            → 번역기가 태그를 손상시키면 match 가 None 을 반환하여 변환이 아무것도 적용되지 않는 문제 발생.
            - 수정: 번역 전 cam_block 을 별도 보존하고, 번역 후 p_txt 에 cam_block 을 붙여서 변환.
        """
        params = _parse_params_from_block(cam_block)
        cam    = classify_camera(params)

        provider  = model_cfg.get("provider", "").lower()
        pipe_type = model_cfg.get("pipeline_type", "").lower()
        repo_id   = model_cfg.get("repo_id", "").lower()
        api_uri   = model_cfg.get("api_model_uri", "").lower()
        is_lora   = "lora" in repo_id or "lora" in api_uri

        if "google_genai" in provider:
            return self._gemini.build(translated_prompt, cam)

        elif "fal_ai" in provider or "qwen" in pipe_type:
            return self._qwen.build(translated_prompt, cam, is_lora=is_lora)

        else:
            return self._generic.build(translated_prompt, cam)

    # 하위 호환 메서드 (bg_composer.py 메서드 교체 없이도 사용 가능)
    def convert_for_gemini(self, prompt: str) -> str:
        """ 기존 _convert_camera_tag_for_gemini() 의 drop-in 대체 메서드.
            prompt 안에 <camera> 태그가 포함되어 있어야 한다.
        """
        _, cam_block = _extract_camera_block(prompt)
        if not cam_block:
            return prompt
        params = _parse_params_from_block(cam_block)
        cam    = classify_camera(params)
        clean  = _CAM_TAG_RE.sub("", prompt).strip()
        return self._gemini.build(clean, cam)

    def convert_for_qwen(self, prompt: str, is_lora: bool = True) -> str:
        """ 기존 _convert_camera_tag_to_sks() 의 drop-in 대체 메서드.
            prompt 안에 <camera> 태그가 포함되어 있어야 한다.
            is_lora: True 이면 <sks> 트리거 포함, False 이면 자연어 지시문 생성.
        """
        _, cam_block = _extract_camera_block(prompt)
        if not cam_block:
            return prompt
        params = _parse_params_from_block(cam_block)
        cam    = classify_camera(params)
        clean  = _CAM_TAG_RE.sub("", prompt).strip()
        return self._qwen.build(clean, cam, is_lora=is_lora)

    # 디버그 헬퍼 ──────────────────────────────────────────────────────
    def explain(self, prompt: str) -> str:
        """ 프롬프트의 카메라 파라미터를 사람이 읽기 쉬운 형태로 설명한다. """
        _, params = parse_camera_tag(prompt)
        cam = classify_camera(params)
        h, v, z = cam["h"], cam["v"], cam["z"]
        return (
            f"Horizontal : {params.horizontal:+.1f}°  →  {h['verbose']}\n"
            f"Vertical   : {params.vertical:+.1f}°  →  {v['verbose']}\n"
            f"Zoom       : {params.zoom:.2f}x       →  {z['verbose']}\n"
            f"Roll       : {params.roll:+.1f}°\n"
        )

# ---------------------------------------------------------------------------
# 7. bg_composer.py 메서드 교체 패치 함수
# ---------------------------------------------------------------------------
def patch_bg_composer(app_instance, engine: PromptEngine = None):
    """ BgComposerApp 인스턴스에 PromptEngine 을 주입하고
        기존 _convert_camera_tag_for_gemini / _convert_camera_tag_to_sks 를 새 버전으로 monkey-patch 한다.

        호출 예시 (bg_composer.py BgComposerApp.__init__() 끝 부분):
            from utils.prompt_engine import patch_bg_composer
            patch_bg_composer(self)

        run_generation() 내 번역 블록 코드 적용. 아래 IMPROVED_RUN_GENERATION_SNIPPET 주석 참고할 것.
    """
    import types

    eng = engine or PromptEngine()
    app_instance._prompt_engine = eng

    def _convert_camera_tag_for_gemini(self, prompt: str) -> str:
        return self._prompt_engine.convert_for_gemini(prompt)

    def _convert_camera_tag_to_sks(self, prompt: str) -> str:
        return self._prompt_engine.convert_for_qwen(prompt)

    app_instance._convert_camera_tag_for_gemini = types.MethodType(
        _convert_camera_tag_for_gemini, app_instance)
    app_instance._convert_camera_tag_to_sks = types.MethodType(
        _convert_camera_tag_to_sks, app_instance)

# ---------------------------------------------------------------------------
# 8. run_generation() 교체 스니펫 (아래 코드를 bg_composer.py 의 기존 번역 블록과 교체한다)
# ---------------------------------------------------------------------------
IMPROVED_RUN_GENERATION_SNIPPET = '''
# ============================================================
# run_generation() 내 프롬프트 처리 블록, bg_composer.py 에서 아래 코드:
    p_txt, n_txt = self._prompt_engine.process(
        p_raw       = p_raw,
        n_raw       = n_raw,
        manual_mode = self.chk_manual_prompt.isChecked(),
        multi_angle = self.chk_multi_angle.isChecked(),
        model_cfg   = self._active_model_config,
    )
# ============================================================
'''

# ---------------------------------------------------------------------------
# 9. 자체 테스트 (python -m utils.prompt_engine)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    engine = PromptEngine()

    print("=" * 72)
    print("PromptEngine — 96 앵글 변환 자체 테스트")
    print("=" * 72)

    BASE = "조선시대 궁중화 스타일의 배경. 섬세한 색채와 붓터치."

    cases = [
        ("정면 아이레벨 미디엄",     0.0,   0.0,  1.0),
        ("우측 45° 하이앵글 와이드", 45.0,  25.0, 0.7),
        ("좌측 90° 로우앵글 클로즈", -90.0,-30.0, 1.6),
        ("정후방 하이앵글 와이드",  170.0,  55.0, 0.5),
        ("후좌 135° 아이레벨",     -135.0,  5.0,  1.0),
        ("정면 극저각 로우앵글",      0.0, -80.0, 1.2),
        ("전방우 더치앵글",          30.0,   0.0,  1.0),
    ]

    gemini_cfg = {"provider": "google_genai", "prompt_mode": "mix_way"}
    qwen_cfg   = {"provider": "fal_ai",       "pipeline_type": "Qwen-Image-Edit-2511",
                  "prompt_mode": "mix_way"}

    for name, h, v, z in cases:
        prompt = f"{BASE} <camera>horizontal={h} vertical={v} zoom={z}</camera>"

        p_g, _ = engine.process(prompt, multi_angle=True, model_cfg=gemini_cfg)
        p_q, _ = engine.process(prompt, multi_angle=True, model_cfg=qwen_cfg)

        print(f"\n▶ {name}  (H={h:+.0f}° V={v:+.0f}° Z={z}x)")
        print(f"  [Gemini]\n    {p_g[:200]}...")
        print(f"  [Qwen  ]\n    {p_q[:200]}...")
        print(f"  [설명  ]\n{engine.explain(prompt)}")