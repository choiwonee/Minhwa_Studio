""" utils/config_loader.py
───────────────────────────────────────────────────────────
: configs/config.ini 를 파싱하여 프로젝트 전체에 공급하는 설정 로더.
  configobj 기반으로 계층형 INI 를 지원한다.

    1) 경로 기준:
    - common.py 의 get_project_root() 를 우선 사용한다.
        (PyInstaller EXE 패키징 환경 자동 대응)
    - import 실패 시 __file__ 기반으로 폴백한다.

    2) 하위 모듈 참조 속성:
    config.weights_dir    -> Path   모델 가중치 루트 경로
    config.weights_path   -> str    weights_dir 의 str 버전 (레거시 호환)
    config.hf_token       -> str    HuggingFace 토큰 (평문)
    config.api_key        -> str    Remote/Google API 키 (평문)
    config.get_config_value(section, key, default) -> Any
"""
from __future__ import annotations

from pathlib import Path

try:
    from configobj import ConfigObj
    _HAS_CONFIGOBJ = True
except ImportError:
    import configparser
    _HAS_CONFIGOBJ = False


def _project_root() -> Path:
    """ common.py 의 get_project_root() 를 lazy 로 호출한다. (circular import 방지 + PyInstaller EXE 패키징 호환) """
    try:
        from utils.common import get_project_root
        return Path(get_project_root())
    except ImportError:
        return Path(__file__).parent.parent


class ConfigLoader:
    """config.ini 를 파싱하고 각종 설정값을 제공하는 싱글톤 클래스."""

    def __init__(self, config_path: str | Path = None):
        if config_path is None:
            config_path = _project_root() / "configs" / "config.ini"

        self._path = Path(config_path)
        self._raw  = {}
        self._load()

        # 자주 사용하는 설정을 속성으로 노출
        self.weights_dir  = self._resolve_weights_dir()
        self.weights_path = str(self.weights_dir)   # 레거시 str 호환 (realesrgan 등)
        
        # token_key를 통해 복호화된 값 사용
        try:
            from utils.token_key import get_valid_hf_token
            self.hf_token = get_valid_hf_token() or ""
        except Exception:
            self.hf_token = self.get_config_value("Settings", "hf_token", "")
            
        try:
            from utils.token_key import get_valid_api_key
            self.api_key = get_valid_api_key() or ""
        except Exception:
            self.api_key = self.get_config_value("Settings", "api_key", "")

    # 로딩 ───────────────────────────────────────────────────────────
    def _load(self):
        """ 설정 파일 로드
            - ConfigObj 파싱 시 UTF-8 인코딩을 명시하여 한글(주석/시나리오 등) 로드 실패 방지
        """
        if not self._path.exists():
            print(f"[ConfigLoader] 경고: 설정 파일을 찾을 수 없습니다: {self._path}")
            return

        if _HAS_CONFIGOBJ:
            self._raw = ConfigObj(str(self._path), encoding="utf-8", interpolation=False)
        else:
            cp = configparser.RawConfigParser()
            cp.read(str(self._path), encoding="utf-8")
            self._raw = {s: dict(cp[s]) for s in cp.sections()}

    # 조회 메서드 ──────────────────────────────────────────────────────
    def get_config_value(self, section: str, key: str, default=None):
        """단순 키-값 조회. 없으면 default 반환."""
        try:
            return self._raw.get(section, {}).get(key, default)
        except Exception:
            return default

    def get_scenarios(self) -> dict:
        """[Scenarios] 섹션 전체를 dict 로 반환한다."""
        if not _HAS_CONFIGOBJ:
            return {}
        scenarios = {}
        raw_sec = self._raw.get("Scenarios", {})
        for key, val in raw_sec.items():
            if isinstance(val, dict):
                scenarios[val.get("label", key)] = {
                    "prompt":   val.get("prompt",   ""),
                    "negative": val.get("negative", ""),
                    "tags":     val.get("tags",     ""),
                }
        return scenarios

    def get_models(self, group: str = "generation") -> dict:
        """ [Models] 하위의 지정된 그룹 모델 설정을 평면 dict 로 반환
            - generation 그룹처럼 뎁스가 깊은 경우(Depth 4)와 sam2, depth 처럼 얕은 경우(Depth 3)를 모두 지원
            - bg_composer.py 의 필터링 조건에 맞게 'key'와 'category' 속성을 dict 내부에 강제 주입하도록 수정
            - use = True 인 항목만 필터링하여 반환
        """
        if not _HAS_CONFIGOBJ:
            return {}
        
        result = {}
        try:
            model_root = self._raw.get("Models", {}).get(group, {})
            for key1, val1 in model_root.items():
                if isinstance(val1, dict):
                    # val1이 바로 모델 설정(Depth 3)인지, 하위 분류(Depth 4)인지 판별
                    if "use" in val1 or "short_name" in val1:
                        use_flag = str(val1.get("use", "True")).strip().lower()
                        if use_flag == "true":
                            cfg = dict(val1)
                            # 단일 뎁스 모델의 경우 고유 key 주입 및 기본 카테고리 지정
                            cfg['key'] = key1
                            if 'category' not in cfg:
                                cfg['category'] = "general"
                            result[key1] = cfg
                    else:
                        # 하위 그룹(예: image_to_image)인 경우 한 번 더 순회 (Depth 4)
                        for model_key, model_cfg in val1.items():
                            if isinstance(model_cfg, dict):
                                use_flag = str(model_cfg.get("use", "True")).strip().lower()
                                if use_flag == "true":
                                    cfg = dict(model_cfg)
                                    # 콤보박스 선택 시 사용할 고유 key 속성 강제 주입
                                    cfg['key'] = model_key
                                    # UI 필터링(T2I/I2I) 조건 매핑을 위해 상위 그룹명(key1)을 category 속성으로 강제 주입
                                    cfg['category'] = key1
                                    result[model_key] = cfg
        except Exception as e:
            print(f"[ConfigLoader] get_models 오류: {e}")
            
        return result

    # 내부 헬퍼 ────────────────────────────────────────────────────────
    def _resolve_weights_dir(self) -> Path:
        """ config.ini 의 weights_path 값을 절대 경로로 변환한다.
            상대 경로면 프로젝트 루트 기준으로 해석한다.
            예: "./models/weights" → <project_root>/models/weights
        """
        raw = self.get_config_value("Settings", "weights_path", "models/weights")
        p   = Path(str(raw))
        if not p.is_absolute():
            # "./models/weights" 또는 "models/weights" 처리
            p = _project_root() / str(raw).lstrip("./").lstrip("/")
        p.mkdir(parents=True, exist_ok=True)
        return p


# 전역 싱글톤 (모든 모듈에서 from utils.config_loader import config 로 사용)
config = ConfigLoader()