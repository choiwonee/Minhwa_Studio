import os
from configobj import ConfigObj
from typing import Any, List, Dict

class ConfigLoader:
    def __init__(self, config_path="config.ini"):
        """ 설정 로더 초기화
            - 프로젝트 루트를 기준으로 configs/config.ini 절대 경로를 계산하여 로드함 """
        # utils 폴더 기준 상위 폴더(루트) 확보
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        # 구조에 맞게 configs 폴더 경로 결합
        self.config_path = os.path.join(project_root, "configs", config_path)

        if os.path.exists(self.config_path):
            self.config = ConfigObj(self.config_path, encoding='utf-8')
            print(f"[Config] Successfully loaded: {self.config_path}")
        else:
            # 루트에 직접 있는 경우를 위한 Fallback
            root_config = os.path.join(project_root, config_path)
            if os.path.exists(root_config):
                self.config_path = root_config
                self.config = ConfigObj(self.config_path, encoding='utf-8')
            else:
                print(f"[Config] Warning: File not found. Using defaults.")
                self.config = ConfigObj()

    def get_config_value(self, section: str, key: str, default=None):
        try:
            return self.config[section][key]
        
        except KeyError:
            return default

    def get_section(self, section_name: str):
        return self.config.get(section_name, {})

    # ==========================================================================
    # [통합] 모델 로딩 메서드 (Group -> Item 또는 Group -> Category -> Item 지원)
    # ==========================================================================
    def get_models(self, target_group: str) -> Dict[str, Any]:
        """ 특정 그룹(target_group) 하위의 모든 모델을 가져와 평탄화된 딕셔너리로 반환.
        지원 구조:
        1. 2단 구조: [Models] -> [[sam2]] -> [[[model_key]]]
        2. 3단 구조: [Models] -> [[generation]] -> [[[image_to_image]]] -> [[[[model_key]]]]
        Returns:
            Dict[str, dict]: { "model_key": { ...model_info... }, ... }
        """
        models = {}
        model_root = self.config.get("Models", {})
        
        # 대소문자 구분 없이 그룹 찾기
        group_section = None
        
        for key in model_root.keys():
            if key.lower() == target_group.lower():
                group_section = model_root[key]
                break
        
        if not group_section or not isinstance(group_section, dict):
            return models

        for key, value in group_section.items():
            if not isinstance(value, dict): continue

            # [판별 로직] value가 '모델'인지 '카테고리'인지 확인
            # 'repo_id'나 'short_name' 키가 있다면 이를 모델(Leaf Node)로 간주
            is_leaf_model = ("repo_id" in value) or ("short_name" in value) or ("pipeline_type" in value)

            if is_leaf_model:
                # 2단 구조 (Group -> Model). category는 별도로 없으므로 'general' 또는 그룹명 사용
                model_info = self._parse_model_entry(key, value, target_group, category="general")
                # 키를 소문자로 저장하여 외부 호출 시 대소문자 무력화
                models[key.lower()] = model_info
            else:
                # 3단 구조 (Group -> Category -> Model)
                category_name = key
                for sub_key, sub_val in value.items():
                    if isinstance(sub_val, dict):
                        model_info = self._parse_model_entry(sub_key, sub_val, target_group, category=category_name)
                        # 키를 소문자로 저장하여 외부 호출 시 대소문자 무력화
                        models[sub_key.lower()] = model_info

        return models

    def _parse_model_entry(self, model_key: str, data: dict, group: str, category: str) -> dict:
        """ 단일 모델 엔트리 파싱 """
        return {
            "key": model_key,
            "group": group.lower(),
            "category": category,
            
            # 기본 정보
            "short_name": self._flatten_text(data.get("short_name", model_key)),
            "description": self._flatten_text(data.get("description", "")),
            "use": data.as_bool("use") if "use" in data else True,
            "is_default": data.as_bool("default") if "default" in data else False,
            
            # 모델 식별 및 로드 정보
            "repo_id": data.get("repo_id", data.get("id", "")),
            "pipeline_type": str(data.get("pipeline_type", "default")).lower(),
            "file_name": data.get("file_name", None),
            
            # 실행 모드 및 Remote 설정 (provider, api_model_uri 추가)
            "mode": data.get("mode", "local").lower(),
            "provider": data.get("provider", None),
            "api_model_uri": data.get("api_model_uri", None),
            "remote_profile": data.get("remote_profile", None),
            "remote_url": data.get("remote_url", None),
            
            # 프롬프트 및 파라미터 제어
            "prompt_mode": data.get("prompt_mode", "two_way").lower(),
            "prompt_style": data.get("prompt_style", "descriptive").lower(),
            "accepts_negative": data.as_bool("accepts_negative") if "accepts_negative" in data else True,
            "accepts_image": data.as_bool("accepts_image") if "accepts_image" in data else True,
            "accepts_mask": data.as_bool("accepts_mask") if "accepts_mask" in data else True,
            
            # 하드웨어 및 옵션
            "options": self._parse_options(data.get("options", [])),
            "vram_requirement": data.as_float("vram_requirement") if "vram_requirement" in data else 0.0,
            "fallback_precision": data.get("fallback_precision", None),
        }

    def _parse_options(self, raw: Any) -> List[str]:
        tokens: List[str] = []
        if raw is None:
            return tokens

        if isinstance(raw, str):
            parts = raw.split(",")
            tokens = [p.strip() for p in parts if p.strip()]
        elif isinstance(raw, list):
            for item in raw:
                if item is None:
                    continue
                
                if isinstance(item, str) and ("," in item):
                    tokens.extend([p.strip() for p in item.split(",") if p.strip()])
                else:
                    s = str(item).strip()
                    if s: tokens.append(s)
        else:
            s = str(raw).strip()
            if s: tokens = [p.strip() for p in s.split(",") if p.strip()]

        return [t.lower() for t in tokens]

    def get_model_info(self, model_key):
        """ 모델 키를 통해 전체 그룹을 탐색하여 모델 정보 반환 """
        model_root = self.config.get("Models", {})
        
        for group_name in model_root.keys():
            # get_models 재사용하여 탐색
            group_models = self.get_models(group_name)
            
            if model_key in group_models:
                return group_models[model_key]
            
        return None

    def _flatten_text(self, text):
        if isinstance(text, list): return " ".join(text)
        if isinstance(text, str): return " ".join(text.replace('\n', ' ').split())
        
        return str(text)

    def get_scenarios(self):
        scenarios = {}
        scenario_section = self.config.get("Scenarios", {})
        
        for key, data in scenario_section.items():
            if isinstance(data, dict):
                label = data.get("label", key)
                scenarios[label] = {
                    "label": label,
                    "prompt": self._flatten_text(data.get("prompt", "")),
                    "negative": self._flatten_text(data.get("negative", "")),
                    "tags": ", ".join(data.get("tags", "")) if isinstance(data.get("tags"), list) else data.get("tags", "")
                }
                
        return scenarios
    
    def get_aspect_ratios(self) -> Dict[float, str]:
        """ Config.ini의 [AspectRatios] 섹션을 로드하여 {float_value: 'w:h'} 형태의 맵 반환 
            - Config 파일에 정의된 값을 우선 사용하며, 없을 경우 기본 하드코딩 값 제공 (Fallback)
        """
        section = self.config.get("AspectRatios", {})
        ratio_map = {}
        
        # ConfigObj는 모든 값을 문자열로 읽으므로 float 변환 필요
        for key, val in section.items():
            try:
                # ini: "16:9" = "1.77" -> python dict: { 1.77: "16:9" }
                ratio_map[float(val)] = key
            except ValueError:
                continue
        
        # 섹션이 비어있거나 파일이 없을 경우를 대비한 기본값
        if not ratio_map:
            return {
                1.0: "1:1", 0.66: "2:3", 1.5: "3:2", 0.75: "3:4", 1.33: "4:3",
                0.8: "4:5", 1.25: "5:4", 0.56: "9:16", 1.77: "16:9", 2.33: "21:9", 0.43: "9:21"
            }
            
        return ratio_map

# =====================================================================================================================
# 모듈 레벨에서 인스턴스 생성: 다른 파일에서 'from utils.config_loader import config' 처리할 수 있게 함.
# =====================================================================================================================
config = ConfigLoader()

# 자체 설정 테스트용 메인
if __name__ == "__main__":
    config = ConfigLoader()
    print("\n[Test] Checking 'generation' group models...")
    
    # 통합된 get_models 메서드 테스트
    # config.ini의 [Models] -> [[generation]] 섹션을 읽어옵니다.
    models = config.get_models("generation")
    
    if not models:
        print("❌ No models found in 'generation' group.")
    else:
        print(f"✅ Found {len(models)} models.")
        for key, item in models.items():
            print(f"  - [{item['category'].upper()}] {item['short_name']}")
            print(f"    Key: {key} | Mode: {item['mode']} | Pipeline: {item['pipeline_type']}")
            
            if item['remote_profile']:
                print(f"    Remote Profile: {item['remote_profile']}")
            print("-" * 50)

    print("\n[Test] Checking 'Scenarios'...")
    scenarios = config.get_scenarios()
    
    for key, s in scenarios.items():
        print(f"  - {s['label']} (Prompt Len: {len(s['prompt'])})")