import os
import json
import re
import numpy as np
import faiss
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
        # [안전장치] **kwargs 추가 완료
        if not self.is_ready or not api_key:
            return {"positive": user_input, "negative": "Error: RAG not ready or API key missing."}

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

            # 2. 필수 가드레일 추출 [복원 완료: 매칭 태그가 많은 순서대로 정렬]
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

            # 4. Gemini 호출 지시문
            client = genai.Client(api_key=api_key)
            instructions = (
                "너는 '한국 전통 회화(조선 민화 포함)' 전용 프롬프트 엔지니어다.\n"
                "너의 최우선 목표는 결과가 반드시 한국 전통 회화 분위기(한지/벽화/비단, 먹선, 담채/석채, 평면 채색)를 유지하는 것이다.\n"
                "아래 '레시피(근거)'는 규칙이다. 근거에 없는 스타일(동화책/실사/현대)을 추정해서 추가하지 마라.\n"
                "특히 다음을 절대 금지한다: 동화책/아동 일러스트(kawaii/chibi/patchwork/bedtime/stars/moon), 실사/시네마틱(DSLR/HDR/3D), 텍스트/로고/워터마크, 현대 오브젝트.\n"
                "출력은 오직 JSON 1개만 반환해라.\n"
                "JSON 스키마: {\"positive\": \"...\", \"negative\": \"...\"}\n"
                "- positive: 이미지 생성용 프롬프트(가능하면 영어 토큰 중심, 쉼표로 구분). 한국 고유명은 유지 가능.\n"
                "- negative: 금지 요소를 쉼표로 나열.\n"
                "추가 설명/문장/코드블록을 절대 붙이지 마라.\n"
            )

            prompt = (
                f"[사용자 입력]\n{user_input.strip()}\n\n"
                f"[레시피(근거) - 반드시 준수]\n{evidence}\n\n"
                "작업 지시:\n"
                "1) 근거를 읽고, 사용자 의도에 가장 맞는 '장르/계열'을 1개 선택해라.\n"
                "2) positive에는 반드시 매체/기법 앵커를 포함해라:\n"
                "   - hanji paper OR aged plaster mural OR silk scroll 중 1개\n"
                "   - ink brush outlines, mineral pigments, flat color fills, minimal shading\n"
                "3) positive에는 사용자 입력의 핵심 주제(오브젝트/모티프)를 1~2개만 포함해 과밀을 피하라.\n"
                "4) negative에는 아래를 반드시 포함해라(그리고 근거에 있는 금지항목도 추가):\n"
                "   - children’s book, storybook, kawaii, chibi, cartoon, anime, patchwork, bedtime, stars, moon, galaxy, rainbow\n"
                "   - photorealistic, DSLR, cinematic lighting, HDR, 3D render\n"
                "   - text, logo, watermark, signature\n"
                "   - modern objects, city skyline, neon, sci-fi\n"
                "5) 결과는 JSON만 출력.\n"
                "\n출력 예시(JSON 형식 그대로):\n{\"positive\":\"...\",\"negative\":\"...\"}\n"
            )
            
            response = client.models.generate_content(
                model=self.gemini_model, 
                contents=f"{instructions}\n\n{prompt}"
            )
            raw = (response.text or "").strip()
            
            m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            raw_json = m.group(0).strip() if m else raw
            
            try:
                data = json.loads(raw_json)
                positive = str(data.get("positive", "")).strip()
                negative = str(data.get("negative", "")).strip()
                if not positive:
                    raise ValueError("positive가 비어 있습니다.")
                return {"positive": positive, "negative": negative}
            except Exception:
                return {"positive": raw, "negative": ""}

        except Exception as e:
            return {"positive": user_input, "negative": f"RAG Error: {e}"}