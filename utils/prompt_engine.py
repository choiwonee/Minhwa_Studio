""" utils/prompt_engine.py
──────────────────────────────────────────────────────
    : Traditional_Art_Project 의 프롬프트 처리 통합 모듈.
    1) 담당 기능
        1. 한국어 → 영어 번역 (translator 연동)
        2. <camera> 태그 파싱 및 96 가지 앵글 처리
        3. 모델(Provider)별 최적화된 프롬프트 빌드
        - Gemini  : INSTRUCTION 지시문 구조
        - Qwen    : <sks> 트리거 + 각도 수치 인라인 구조
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
    horizontal: float = 0.0    # -180 ~ +180  (양수=우, 음수=좌, ±180=정후방)
    vertical:   float = 0.0    # -90  ~ +90   (양수=상향, 음수=하향)
    zoom:       float = 1.0    # 0.5  ~ 2.0   (1.0=표준)
    roll:       float = 0.0    # Dutch angle (선택)

# ---------------------------------------------------------------------------
# 2. 앵글 분류 테이블
#    형식: (범위_하한, 범위_상한, 자세한_묘사, 짧은_태그, 각도_레이블)
# ---------------------------------------------------------------------------

# 수평(Azimuth) 8분할 ───────────────────────────────────────────────────
#   기존 코드의 4단계(front/left side/right side/back)에서 대각선 4방향 추가.
#   ※ abs(h)>=120 이 h>=30 조건에 이미 걸려 back view 가 생성되지 않던 버그 수정.
_H_BINS = [
    (-22.5,   22.5,  "directly in front of the subject",       "front view",            "0°"   ),
    ( 22.5,   67.5,  "from the front-right of the subject",    "front-right diagonal",  "45°R" ),
    ( 67.5,  112.5,  "from the right side of the subject",     "right side profile",    "90°R" ),
    (112.5,  157.5,  "from the rear-right of the subject",     "rear-right diagonal",   "135°R"),
    (157.5,  180.0,  "from directly behind the subject",       "rear view",             "180°" ),
    (-180.0, -157.5, "from directly behind the subject",       "rear view",             "180°" ),
    (-157.5, -112.5, "from the rear-left of the subject",      "rear-left diagonal",    "135°L"),
    (-112.5,  -67.5, "from the left side of the subject",      "left side profile",     "90°L" ),
    ( -67.5,  -22.5, "from the front-left of the subject",     "front-left diagonal",   "45°L" ),
]

# 수직(Elevation) 5분할 ────────────────────────────────────────────────
_V_BINS = [
    (-90.0, -40.0, "extreme low-angle worm's-eye view",      "worm's-eye",   "extreme low" ),
    (-40.0, -10.0, "low-angle shot looking upward",          "low-angle",    "low"         ),
    (-10.0,  10.0, "straight eye-level shot",                "eye-level",    "eye-level"   ),
    ( 10.0,  40.0, "high-angle shot looking downward",       "high-angle",   "high"        ),
    ( 40.0,  90.0, "extreme high-angle bird's-eye view",     "bird's-eye",   "extreme high"),
]

# 줌(Distance) 3분할 ────────────────────────────────────────────────────
_Z_BINS = [
    (0.0,  0.75, "a wide establishing shot",   "wide shot",  "wide"    ),
    (0.75, 1.25, "a medium shot",              "medium shot","medium"  ),
    (1.25, 9.9,  "a tight close-up shot",      "close-up",   "close-up"),
]

def _classify(value: float, bins: list) -> dict:
    """ value 가 속하는 bin 을 찾아 dict 로 반환한다.
        어느 구간에도 해당하지 않으면 가장 가까운 bin 을 반환(폴백).
    """
    for lo, hi, verbose, short, tag in bins:
        if lo <= value <= hi:
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
# 4. 모델별 프롬프트 빌더
# ---------------------------------------------------------------------------
class _GeminiBuilder:
    """ Gemini 2.0 / 3.0 이미지 편집 모델 전용 빌더.
        1) 핵심 설계 근거
        ──────────────────────────────────────────────────────
        - Gemini 는 문장 앞쪽 **지시(INSTRUCTION) 동사 구조**에 강하게 반응한다.
        - 단순 형용사 나열("left side view") 보다 "The camera is physically positioned ... and tilted to ..." 식의 물리적 위치 서술이 구도 반영률을 크게 높인다.
        - focal length 힌트를 추가하면 Gemini 내부 렌즈 개념과 매핑되어 원근감·피사계 심도가 더 정확해진다.
        - 앵글 메타 태그([angle: ...])를 별도 줄에 추가하면 모델이 이를 보조 단서(supplementary cue)로 활용한다.
    """

    def build(self, translated_prompt: str, cam: dict) -> str:
        h, v, z = cam["h"], cam["v"], cam["z"]
        p: CameraParams = cam["params"]

        angle_block = self._build_angle_block(h, v, z, p)
        return f"{angle_block}\n\n{translated_prompt}"

    # 내부 메서드 ──────────────────────────────────────────────────────
    def _build_angle_block(self, h: dict, v: dict, z: dict, p: CameraParams) -> str:
        """ 3 레이어 앵글 지시 블록 생성. """

        # L1: 핵심 지시문 (INSTRUCTION 스타일)
        instruction = (
            f"CAMERA DIRECTION: Render this scene as {z['verbose']}, "
            f"captured {h['verbose']}, "
            f"with a {v['verbose']}. "
            f"The camera is physically positioned {h['short']} of the subject "
            f"and tilted to a {v['short']} angle."
        )

        # Dutch angle
        roll_note = ""
        if abs(p.roll) > 5.0:
            dir_word = "clockwise" if p.roll > 0 else "counter-clockwise"
            roll_note = f" Apply a {abs(p.roll):.0f}° Dutch angle ({dir_word})."

        # L2: 렌즈/화각 힌트
        focal    = self._focal_hint(p.zoom)
        lens_note = f"Use a {focal} lens perspective. Subject framing: {z['short']}."

        # L3: 앵글 메타 태그
        meta = f"[angle: {h['tag']}, {v['tag']}, {z['tag']}]"

        return f"{instruction}{roll_note}\n{lens_note}\n{meta}"

    @staticmethod
    def _focal_hint(zoom: float) -> str:
        """ 줌 값을 35mm 환산 초점거리 힌트로 변환. """
        if zoom <= 0.6:   return "14–20mm ultra-wide"
        if zoom <= 0.85:  return "24–28mm wide-angle"
        if zoom <= 1.15:  return "35–50mm standard"
        if zoom <= 1.5:   return "85–105mm portrait telephoto"
        return "200mm+ telephoto compression"

class _QwenBuilder:
    """ Qwen-Image-Edit-2511 / Multiple-Angles-LoRA 전용 빌더.
        1) 핵심 설계 근거
        ──────────────────────────────────────────────────────
        - LoRA 트리거 토큰 "<sks>" 를 프롬프트 맨 앞에 배치해야 LoRA 가중치가 올바르게 활성화된다.
        - Gemini 와 달리 Qwen 은 **묘사(description)** 형식이 더 잘 반영된다.
        → 구도 단어를 복합 형용사로 인라인 배치.
        - 수평·수직 각도 수치를 명시(e.g. "45° rotated to the right")하면 LoRA 학습 데이터 패턴과 매핑되어 정확도가 높아진다.
        - 4단계 표현 → 8분할 표현 전환으로 대각선 구도 누락 해소.
    """

    def build(self, translated_prompt: str, cam: dict, use_sks: bool = True) -> str:
        h, v, z = cam["h"], cam["v"], cam["z"]
        p: CameraParams = cam["params"]

        sks   = "<sks> " if use_sks else ""
        phrase = self._angle_phrase(h, v, z, p)
        return f"{sks}{phrase}, {translated_prompt}"

    # 내부 메서드 ──────────────────────────────────────────────────────
    def _angle_phrase(self, h: dict, v: dict, z: dict, p: CameraParams) -> str:
        """ 짧고 명확한 구도 묘사 구문 생성. """
        h_deg = abs(p.horizontal)
        h_dir = "right" if p.horizontal >= 0 else "left"
        v_deg = abs(p.vertical)
        v_dir = "up"    if p.vertical   >= 0 else "down"

        phrase = f"{z['short']} {h['short']} {v['short']} view"

        angle_hint = ""
        if h_deg > 10.0:
            angle_hint += f", {h_deg:.0f}° rotated to the {h_dir}"
        if v_deg > 5.0:
            angle_hint += f", {v_deg:.0f}° tilted {v_dir}"
        if abs(p.roll) > 5.0:
            roll_dir = "clockwise" if p.roll > 0 else "counter-clockwise"
            angle_hint += f", {abs(p.roll):.0f}° Dutch angle {roll_dir}"

        return phrase + angle_hint

class _GenericBuilder:
    """ 범용 폴백 빌더 (기존 코드와 유사한 단순 묘사형). """

    def build(self, translated_prompt: str, cam: dict) -> str:
        h, v, z = cam["h"], cam["v"], cam["z"]
        return f"{z['short']} {h['short']} {v['short']}, {translated_prompt}"

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
    def process(self, p_raw: str, n_raw: str = "", manual_mode: bool = False, multi_angle: bool = False, model_cfg: dict = None, use_translator: bool = True) -> Tuple[str, str]:
        """ 사용자 입력 프롬프트를 최종 추론용 텍스트로 변환한다. """
        model_cfg = model_cfg or {}

        # 수동 모드: 원본 그대로 반환 ──────────────────────────────────
        if manual_mode:
            return p_raw, n_raw

        # Step 1: <camera> 태그 분리 보존 (번역기 손상 방지) ────────────
        p_no_tag, cam_block = _extract_camera_block(p_raw)

        # Step 2: 번역 여부에 따라 텍스트 처리 (수정된 부분: 이중 번역 방지)
        if use_translator:
            p_txt = _translate(p_no_tag)
            n_txt = _translate(n_raw)
        else:
            p_txt = p_no_tag
            n_txt = n_raw

        # Step 3: 멀티 앵글 변환 및 유실 방지 처리 ───────────────────────
        if cam_block:
            if multi_angle:
                p_txt = self._apply_angle(p_txt, cam_block, model_cfg)
            else:
                p_txt = f"{p_txt} {cam_block}"

        return p_txt, n_txt

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
        # 번역된 텍스트 + 원본 <camera> 블록 결합
        f"{translated_prompt} {cam_block}"

        params = _parse_params_from_block(cam_block)
        cam    = classify_camera(params)

        provider = model_cfg.get("provider", "").lower()

        if "google_genai" in provider:
            return self._gemini.build(translated_prompt, cam)

        elif "fal_ai" in provider or "qwen" in model_cfg.get("pipeline_type", "").lower():
            return self._qwen.build(translated_prompt, cam)

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

    def convert_for_qwen(self, prompt: str, use_sks: bool = True) -> str:
        """ 기존 _convert_camera_tag_to_sks() 의 drop-in 대체 메서드.
            prompt 안에 <camera> 태그가 포함되어 있어야 한다.
        """
        _, cam_block = _extract_camera_block(prompt)
        if not cam_block:
            return prompt
        params = _parse_params_from_block(cam_block)
        cam    = classify_camera(params)
        clean  = _CAM_TAG_RE.sub("", prompt).strip()
        return self._qwen.build(clean, cam, use_sks=use_sks)

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
# 7. bg_composer.py 메서드 2개를 PromptEngine 으로 교체하는 패치 함수
#    (기존 클래스 내부를 수정하기 어려울 때 __init__() 끝에서 호출)
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
        ("정후방 버즈아이 와이드",  170.0,  55.0, 0.5),
        ("후좌 135° 아이레벨",     -135.0,  5.0,  1.0),
        ("정면 극저각 워름즈아이",    0.0, -80.0, 1.2),
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
