from deep_translator import GoogleTranslator

class PromptTranslator:
    def __init__(self):
        # deep_translator는 별도의 인스턴스 초기화가 크게 필요 없으나 호환성을 위해 유지
        pass

    def translate(self, text, src='ko', dest='en'):
        """ deep-translator를 사용하여 한글 텍스트를 영문으로 번역한다. 기존 googletrans보다 연결 안정성이 높다. """
        if not text:
            return ""
        
        try:
            # deep_translator는 동기식(Synchronous)으로 동작하므로 asyncio 불필요
            # source='auto'로 설정하면 언어 감지 후 번역
            result = GoogleTranslator(source='auto', target=dest).translate(text)
            return result
            
        except Exception as e:
            print(f"[Translator] Error: {e}")
            # 번역 실패 시 프로세스가 멈추지 않도록 원문 반환
            return text

# 전역 인스턴스 생성
translator = PromptTranslator()