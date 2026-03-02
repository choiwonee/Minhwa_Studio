import os
import json
import re
import numpy as np
import faiss

from utils.config_loader import config
from sentence_transformers import SentenceTransformer
from google import genai

class RAGPrompter:
    def __init__(self, recipes_filename="recipes_korean_trad_200.jsonl"):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.recipes_path = os.path.join(base_dir, recipes_filename)
        
        self.embed_model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        self.gemini_model = "gemini-3-flash-preview"
        
        self.docs = []
        self.model = None
        self.index = None
        self.is_ready = False

        # [복원 완료] 원본 파일에 있던 강력한 가드레일 태그 100% 복구
        self.must_have_any_tags = [
            "region:korea",
            "texture:hanji", "texture:mural", "texture:silk",
            "palette:obangsaek", "rule:palette_limit", "rule:saturation", "rule:no_gradient",
            "negative:common", "negative:text", "negative:photoreal", 
            "negative:storybook", "negative:modern", "guardrail:domain"
        ]

    def build_index(self):
        try:
            with open(self.recipes_path, "r", encoding="utf-8") as f:
                self.docs = [json.loads(line.strip()) for line in f if line.strip()]
                
            self.model = SentenceTransformer(self.embed_model_name)
            embed_texts = [f"{d.get('title', '')}\nTags: {' '.join(d.get('tags', []))}\n{d.get('text', '')}" for d in self.docs]
            
            emb = self.model.encode(embed_texts, normalize_embeddings=True)
            emb = np.asarray(emb, dtype=np.float32)
            
            self.index = faiss.IndexFlatIP(emb.shape[1])
            self.index.add(emb)
            self.is_ready = True
            print("[RAG] Index successfully built.")
        except Exception as e:
            print(f"[RAG] Failed to build index: {e}")

    def _infer_query_tags(self, query: str):
        # [복원 완료] 원본의 세밀한 태그 추론 로직 100% 복구
        s = query.lower()
        tags = set()
        
        if any(k in s for k in ["봄", "spring"]): tags.add("season:spring")
        if any(k in s for k in ["여름", "summer", "덥", "무더"]): tags.add("season:summer")
        if any(k in s for k in ["가을", "autumn", "fall", "단풍"]): tags.add("season:autumn")
        if any(k in s for k in ["겨울", "winter", "눈", "추운", "한파"]): tags.add("season:winter")

        if any(k in s for k in ["민화", "minhwa"]): tags.add("style:minhwa")
        if any(k in s for k in ["조선", "joseon"]): tags.add("era:joseon")
        if any(k in s for k in ["십장생", "sipjangsaeng", "장수"]): tags.add("theme:sipjangsaeng")
        if any(k in s for k in ["호작도", "호랑이", "까치"]): tags.add("theme:hojakdo")
        if any(k in s for k in ["책가도", "책거리", "책"]): tags.add("theme:chaekgado")
        if any(k in s for k in ["화조도", "꽃", "새"]): tags.add("theme:hwajodo")
        if any(k in s for k in ["연꽃", "lotus"]): tags.add("motif:lotus")
        if any(k in s for k in ["모란", "peony"]): tags.add("motif:peony")
        if any(k in s for k in ["매화", "plum"]): tags.add("motif:plum_blossom")
        if any(k in s for k in ["대나무", "bamboo"]): tags.add("motif:bamboo")
        if any(k in s for k in ["소나무", "pine"]): tags.add("motif:pine")

        return sorted(tags)

    def generate_enhanced_prompt(self, user_input: str, api_key: str, **kwargs):
        if not self.is_ready:
            return {"positive": user_input, "negative": "Error: RAG index not ready yet. Please wait a moment and try again."}
        if not api_key:
            return {"positive": user_input, "negative": "Error: API key missing. Please set your Google API key."}

        try:
            qtags = self._infer_query_tags(user_input)
            
            # 1. FAISS 검색 및 보너스 점수 적용
            q_emb = np.asarray(self.model.encode([user_input], normalize_embeddings=True), dtype=np.float32)
            scores, idxs = self.index.search(q_emb, 18) 
            
            reranked = []
            for score, i in zip(scores[0], idxs[0]):
                doc = self.docs[int(i)]
                bonus = sum(0.06 for t in qtags if t in doc.get("tags", []))
                reranked.append((float(score) + bonus, doc))
            reranked.sort(key=lambda x: x[0], reverse=True)
            top_docs = reranked[:4] 

            # 2. 필수 가드레일 추출
            must = []
            must_tag_set = set(self.must_have_any_tags)
            for d in self.docs:
                d_tags = set(d.get("tags", []))
                if d_tags.intersection(must_tag_set):
                    must.append(d)
            
            def score_m(d):
                return len(set(d.get("tags", [])).intersection(must_tag_set))
                
            must_sorted = sorted(must, key=score_m, reverse=True)
            must_docs = must_sorted[:8]
            
            # 3. Evidence 텍스트 빌드
            used_ids = set()
            chunks = []
            for d in must_docs + [x[1] for x in top_docs]:
                doc_id = d.get("id", "unknown")
                if doc_id not in used_ids:
                    used_ids.add(doc_id)
                    title = d.get("title", "")
                    tags_show = " ".join(d.get("tags", [])[:8])
                    text = d.get("text", "")[:700]
                    chunks.append(f"[{doc_id}] | {title}\nTags: {tags_show}\n{text}")
                    
            evidence = "\n\n".join(chunks).strip()[:5200]

            # 4. config.ini 에서 지시문 및 템플릿 동적 로드
            client = genai.Client(api_key=api_key)
            
            # config.ini에 누락되었을 경우를 대비한 최소한의 fallback 문자열 제공
            fallback_instructions = "너는 한국 전통 회화 프롬프트 엔지니어다. 결과를 JSON으로 출력해라."
            fallback_template = "[사용자 입력]\n{user_input}\n\n[레시피]\n{evidence}\n위 내용을 바탕으로 JSON을 만들어라."
            
            instructions_raw = config.get_config_value("RAG_Templates", "instructions", fallback_instructions)
            prompt_template_raw = config.get_config_value("RAG_Templates", "prompt_template", fallback_template)

            # 파이썬 동적 변수 주입 (.format 사용)
            instructions = instructions_raw.strip()
            prompt = prompt_template_raw.format(
                user_input=user_input.strip(), 
                evidence=evidence
            ).strip()
            
            # Gemini API 호출
            response = client.models.generate_content(
                model=self.gemini_model, 
                contents=f"{instructions}\n\n{prompt}"
            )
            raw = (response.text or "").strip()
            
            m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            raw_json = m.group(0).strip() if m else raw
            
            try:
                data = json.loads(raw_json)
                return data 
            except Exception:
                return {"change": raw, "negative": ""}

        except Exception as e:
            return {"change": user_input, "negative": f"RAG Error: {e}"}