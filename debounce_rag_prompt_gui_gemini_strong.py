"""
debounce_rag_prompt_gui_gemini_strong.py
- "고정력 강화" 버전:
  1) 레시피 임베딩에 (title + tags + text) 또는 (title + text) 를 합쳐서 사용 가능
  2) '조선/민화 고정'에 필요한 핵심 레시피(가드레일)를 항상 evidence에 포함(강제)
  3) RAG 검색 결과는 top-k + 태그/키워드 보너스로 재정렬(간단한 rerank)

설치:
  pip install -U sentence-transformers faiss-cpu numpy google-genai

API 키:
  환경변수 GEMINI_API_KEY 권장 (또는 AppConfig.gemini_api_key에 직접 넣기)

실행:
  python debounce_rag_prompt_gui_gemini_strong.py
"""

from __future__ import annotations

import json
import re
import threading
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import ttk
from typing import Dict, List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

try:
    from google import genai
except ImportError as e:
    raise ImportError(
        "google-genai가 설치되어 있지 않습니다.\n"
        "다음으로 설치하세요:\n"
        "  pip install -U google-genai\n"
    ) from e


# =========================
# 1) 설정
# =========================
@dataclass
class AppConfig:
    # 디바운스(입력 멈춘 뒤 호출까지 대기, ms)
    debounce_ms: int = 1000

    # RAG 검색 개수(최종 top-k)
    rag_top_k: int = 4

    # 1차 FAISS 검색 풀(여기서 뽑아놓고 rerank)
    rag_pool_k: int = 18

    # 레시피 파일(민화 100개)
    recipes_path: str = "recipes_20.jsonl"

    # 임베딩 모델(다국어)
    embed_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # Gemini 모델명
    gemini_model: str = "gemini-3-flash-preview"

    # (선택) 키를 코드에 직접 넣고 싶다면 문자열로 지정
    # 보안상 권장 X. (깃허브 업로드하면 키 유출)
    gemini_api_key: str | None = None

    # ===== 고정력 강화 옵션 =====

    # 임베딩에 tags까지 넣을지 여부
    # - True: title + " ".join(tags) + text를 임베딩
    # - False: title + text만 임베딩 (tags는 별도 rerank/강제포함에 사용)
    include_tags_in_embedding: bool = True

    # (강제 포함) evidence에 항상 들어가야 하는 '가드레일' 태그들
    # - 조선/민화 스타일 고정 + 실사 방지 + 텍스트/워터마크 방지 등
    must_have_any_tags: List[str] = field(default_factory=lambda: [
    # ---- 도메인/스타일 고정 ----
    # ✅ 새 200 레시피 기본 스타일 태그
    "region:korea",

    # ---- 매체(질감) 고정: 한지/벽화/비단 중 최소 1개는 항상 들어오게 ----
    "texture:hanji",
    "texture:mural",
    "texture:silk",

    # ---- 팔레트/채색 규칙 ----
    "palette:obangsaek",
    "rule:palette_limit",
    "rule:saturation",
    "rule:no_gradient",

    # ---- 절대 금지(누수 방지) ----
    "negative:common",
    "negative:text",
    "negative:photoreal",
    "negative:storybook",         # ✅ 동화책/귀여움 방지 핵심
    "negative:modern",            # ✅ 현대 오브젝트 방지

    # ---- 기타 가드레일 ----
    "guardrail:domain",
])

    # 강제 포함 문서 수 상한 (너무 많이 넣으면 LLM이 산만해질 수 있음)
    must_have_max_docs: int = 8

    # evidence 길이 제한(너무 길면 LLM이 규칙을 놓침)
    per_doc_char_limit: int = 700   # 문서 하나당 잘라서 보낼 최대 글자 수
    max_evidence_chars: int = 5200  # evidence 전체 최대 글자 수

    # Gemini 출력 JSON 강제
    force_json: bool = True


# =========================
# 2) 유틸: 사용자 입력에서 '의도 태그' 추출(간단)
# =========================
def infer_query_tags(user_input: str) -> List[str]:
    """
    사용자의 자연어 입력에서 힌트가 되는 태그를 추정(단순 키워드 매칭).
    * 이건 'AI 추론'이 아니라 그냥 규칙 기반 문자열 포함 체크.
    """
    s = user_input.lower()
    tags = set()

    # 계절
    if any(k in s for k in ["봄", "spring"]):
        tags.add("season:spring")
    if any(k in s for k in ["여름", "summer", "덥", "무더"]):
        tags.add("season:summer")
    if any(k in s for k in ["가을", "autumn", "fall", "단풍"]):
        tags.add("season:autumn")
    if any(k in s for k in ["겨울", "winter", "눈", "추운", "한파"]):
        tags.add("season:winter")

    # 대표 테마/모티프
    if any(k in s for k in ["민화", "minhwa"]):
        tags.add("style:minhwa")
    if any(k in s for k in ["조선", "joseon"]):
        tags.add("era:joseon")
    if any(k in s for k in ["십장생", "sipjangsaeng", "장수"]):
        tags.add("theme:sipjangsaeng")
    if any(k in s for k in ["호작도", "호랑이", "까치"]):
        tags.add("theme:hojakdo")
    if any(k in s for k in ["책가도", "책거리", "책"]):
        tags.add("theme:chaekgado")
    if any(k in s for k in ["화조도", "꽃", "새"]):
        tags.add("theme:hwajodo")
    if any(k in s for k in ["연꽃", "lotus"]):
        tags.add("motif:lotus")
    if any(k in s for k in ["모란", "peony"]):
        tags.add("motif:peony")
    if any(k in s for k in ["매화", "plum"]):
        tags.add("motif:plum_blossom")
    if any(k in s for k in ["대나무", "bamboo"]):
        tags.add("motif:bamboo")
    if any(k in s for k in ["소나무", "pine"]):
        tags.add("motif:pine")

    return sorted(tags)


# =========================
# 3) RAG 인덱스
# =========================
class RAGRecipeIndex:
    """
    문서(레시피)를 임베딩하여 FAISS에서 검색.
    - normalize_embeddings=True + IndexFlatIP => 코사인 유사도 유사
    """

    def __init__(self, recipes_path: str, embed_model_name: str, include_tags_in_embedding: bool):
        self.recipes_path = recipes_path
        self.embed_model_name = embed_model_name
        self.include_tags_in_embedding = include_tags_in_embedding

        self.docs: List[dict] = []
        self.model: SentenceTransformer | None = None
        self.index: faiss.Index | None = None
        self.embeddings: np.ndarray | None = None

    def load_docs(self) -> None:
        docs = []
        with open(self.recipes_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                docs.append(json.loads(line))
        if not docs:
            raise ValueError("레시피 문서가 비어 있습니다. recipes 파일 내용을 확인하세요.")
        self.docs = docs

    def _build_embed_text(self, doc: dict) -> str:
        """
        어떤 필드를 임베딩에 넣을지 결정하는 함수.
        - include_tags_in_embedding=True면 title + tags + text
        - False면 title + text
        """
        title = doc.get("title", "")
        tags = doc.get("tags", [])
        text = doc.get("text", "")

        if self.include_tags_in_embedding:
            return f"{title}\nTags: {' '.join(tags)}\n{text}"
        return f"{title}\n{text}"

    def build(self) -> None:
        self.load_docs()
        self.model = SentenceTransformer(self.embed_model_name)

        embed_texts = [self._build_embed_text(d) for d in self.docs]
        emb = self.model.encode(embed_texts, normalize_embeddings=True)
        emb = np.asarray(emb, dtype=np.float32)

        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb)

        self.embeddings = emb
        self.index = index

    def search_pool(self, query: str, pool_k: int) -> List[Tuple[float, dict]]:
        if self.index is None or self.model is None:
            raise RuntimeError("인덱스가 아직 build되지 않았습니다.")
        if pool_k <= 0:
            raise ValueError("pool_k는 1 이상이어야 합니다.")

        q_emb = self.model.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype=np.float32)

        scores, idxs = self.index.search(q_emb, pool_k)

        results: List[Tuple[float, dict]] = []
        for score, i in zip(scores[0], idxs[0]):
            doc = self.docs[int(i)]
            results.append((float(score), doc))
        return results


# =========================
# 4) rerank + 강제 포함
# =========================
def tag_bonus(doc: dict, query_tags: List[str]) -> float:
    """
    간단한 보너스:
    - 문서 tags에 query_tags가 포함되면 가산점
    """
    d_tags = set(doc.get("tags", []))
    bonus = 0.0
    for t in query_tags:
        if t in d_tags:
            bonus += 0.06  # 너무 크면 임베딩 점수를 덮어씀 → 소량만
    return bonus


def pick_must_have_docs(all_docs: List[dict], must_have_any_tags: List[str], max_docs: int) -> List[dict]:
    """
    must_have_any_tags 중 하나라도 포함하는 문서들을 뽑되,
    개수가 너무 많으면 max_docs까지만.
    """
    must = []
    must_tag_set = set(must_have_any_tags)
    for d in all_docs:
        d_tags = set(d.get("tags", []))
        if d_tags.intersection(must_tag_set):
            must.append(d)

    # 너무 많으면: 태그 매칭 수가 큰 문서를 우선 (간단한 휴리스틱)
    def score_m(d):
        d_tags = set(d.get("tags", []))
        return len(d_tags.intersection(must_tag_set))

    must_sorted = sorted(must, key=score_m, reverse=True)
    return must_sorted[:max_docs]


def build_evidence(
    must_docs: List[dict],
    ranked_docs: List[Tuple[float, dict]],
    per_doc_char_limit: int,
    max_evidence_chars: int,
) -> Tuple[str, List[str]]:
    """
    evidence 문자열 생성.
    - must_docs는 항상 먼저 넣음
    - 그 다음 ranked_docs에서 중복 없는 doc을 추가
    Returns:
      evidence_text, used_doc_ids
    """
    used_ids = set()
    chunks = []

    def add_doc(doc: dict, score: float | None):
        doc_id = doc.get("id", "unknown")
        if doc_id in used_ids:
            return
        used_ids.add(doc_id)

        title = doc.get("title", "")
        tags = doc.get("tags", [])
        text = doc.get("text", "")

        if len(text) > per_doc_char_limit:
            text = text[:per_doc_char_limit] + "…(truncated)"

        head = f"[{doc_id}]"
        if score is not None:
            head += f" score={score:.4f}"
        if title:
            head += f" | {title}"

        tags_show = " ".join(tags[:8]) + (" ..." if len(tags) > 8 else "")
        chunk = f"{head}\nTags: {tags_show}\n{text}\n"
        chunks.append(chunk)

    for d in must_docs:
        add_doc(d, score=None)

    for s, d in ranked_docs:
        add_doc(d, score=s)

    evidence = "\n".join(chunks).strip()

    if len(evidence) > max_evidence_chars:
        evidence = evidence[:max_evidence_chars] + "\n…(evidence truncated)"

    return evidence, sorted(used_ids)


# =========================
# 5) Gemini 호출(JSON)
# =========================
def call_gemini_json(
    user_input: str,
    evidence: str,
    model: str,
    api_key: str | None = None,
    force_json: bool = True,
) -> Dict[str, str]:
    client = genai.Client(api_key=api_key) if api_key else genai.Client()

    instructions = (
        "너는 '한국 전통 회화(조선 민화 포함)' 전용 프롬프트 엔지니어다.\n"
        "너의 최우선 목표는 결과가 반드시 한국 전통 회화 분위기(한지/벽화/비단, 먹선, 담채/석채, 평면 채색)를 유지하는 것이다.\n"
        "아래 '레시피(근거)'는 규칙이다. 근거에 없는 스타일(동화책/실사/현대)을 추정해서 추가하지 마라.\n"
        "특히 다음을 절대 금지한다: 동화책/아동 일러스트(kawaii/chibi/patchwork/bedtime/stars/moon), 실사/시네마틱(DSLR/HDR/3D), 텍스트/로고/워터마크, 현대 오브젝트.\n"
    )
    if force_json:
        instructions += (
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
        "   (예: 조선 민화/미인도/초충도/풍속화(씨름도)/산수 두루마리/고구려 사신도 벽화/신라 천마도)\n"
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
    )

    # (강력 권장) JSON 예시를 prompt 끝에 붙이면 파싱 실패가 줄어듭니다.
    if force_json:
        prompt += "\n출력 예시(JSON 형식 그대로):\n{\"positive\":\"...\",\"negative\":\"...\"}\n"

    response = client.models.generate_content(
        model=model,
        contents=f"{instructions}\n\n{prompt}",
    )
    raw = (response.text or "").strip()

    if force_json:
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

    return {"positive": raw, "negative": ""}


# =========================
# 6) Tkinter 앱
# =========================
class DebounceRAGGeminiStrongApp:
    def __init__(self, root: tk.Tk, config: AppConfig):
        self.root = root
        self.config = config

        self.root.title("Debounce + RAG(Strong) + Gemini Prompt Recommender")
        self.root.geometry("1120x720")

        self._after_id = None
        self._last_request_id = 0

        self.rag = RAGRecipeIndex(
            config.recipes_path,
            config.embed_model_name,
            include_tags_in_embedding=config.include_tags_in_embedding,
        )

        self._build_ui()
        self._build_index_async()

    def _build_ui(self):
        header = ttk.Label(
            self.root,
            text="(고정력 강화) 입력 멈춘 뒤 1초 후: RAG 검색 + 필수 규칙 강제 포함 → Gemini 추천",
            font=("Segoe UI", 14, "bold"),
        )
        header.pack(pady=(12, 6))

        self.status_var = tk.StringVar(value="임베딩 모델/인덱스 로딩 중… (처음 1회는 조금 걸릴 수 있어요)")
        ttk.Label(self.root, textvariable=self.status_var).pack(pady=(0, 10))

        input_frame = ttk.Frame(self.root)
        input_frame.pack(fill="x", padx=12)

        ttk.Label(input_frame, text="사용자 프롬프트:").pack(anchor="w")
        self.text = tk.Text(input_frame, height=6, wrap="word")
        self.text.pack(fill="x", pady=(4, 0))
        self.text.bind("<KeyRelease>", self.on_text_change)

        row = ttk.Frame(self.root)
        row.pack(fill="x", padx=12, pady=10)
        self.btn_now = ttk.Button(row, text="지금 추천", command=self.suggest_now, state="disabled")
        self.btn_now.pack(side="right")

        out_frame = ttk.Frame(self.root)
        out_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        out_frame.columnconfigure(0, weight=1)
        out_frame.columnconfigure(1, weight=1)
        out_frame.columnconfigure(2, weight=1)

        ttk.Label(out_frame, text="Evidence(강제 포함 + RAG top-k):").grid(row=0, column=0, sticky="w")
        ttk.Label(out_frame, text="추천 Positive Prompt:").grid(row=0, column=1, sticky="w")
        ttk.Label(out_frame, text="추천 Negative Prompt:").grid(row=0, column=2, sticky="w")

        self.out_evidence = tk.Text(out_frame, wrap="word")
        self.out_positive = tk.Text(out_frame, wrap="word")
        self.out_negative = tk.Text(out_frame, wrap="word")

        self.out_evidence.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(4, 0))
        self.out_positive.grid(row=1, column=1, sticky="nsew", padx=(0, 6), pady=(4, 0))
        self.out_negative.grid(row=1, column=2, sticky="nsew", pady=(4, 0))

        for t in (self.out_evidence, self.out_positive, self.out_negative):
            t.configure(state="disabled")

    def _build_index_async(self):
        def worker():
            try:
                self.rag.build()
                ok = True
                msg = "준비 완료. 입력을 멈추면 1초 후 자동 추천됩니다. (고정력 강화 버전)"
            except Exception as e:
                ok = False
                msg = f"로딩 실패: {e}"
            self.root.after(0, lambda: self._on_index_ready(ok, msg))

        threading.Thread(target=worker, daemon=True).start()

    def _on_index_ready(self, ok: bool, msg: str):
        self.status_var.set(msg)
        self.btn_now.configure(state=("normal" if ok else "disabled"))

    def on_text_change(self, event=None):
        self.schedule_suggest()

    def schedule_suggest(self):
        if self.btn_now["state"] == "disabled":
            return

        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

        self.status_var.set(f"입력 감지… {self.config.debounce_ms/1000:.1f}초 후 추천")
        self._after_id = self.root.after(self.config.debounce_ms, self._trigger_suggest)

    def suggest_now(self):
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None
        self._trigger_suggest()

    def _trigger_suggest(self):
        user_input = self.text.get("1.0", "end").strip()

        self._last_request_id += 1
        request_id = self._last_request_id
        self.status_var.set("RAG 검색 + 필수 규칙 포함 + Gemini 추천 생성 중…")

        def worker():
            try:
                qtags = infer_query_tags(user_input)

                pool = self.rag.search_pool(user_input, pool_k=self.config.rag_pool_k)

                reranked = []
                for score, doc in pool:
                    reranked.append((score + tag_bonus(doc, qtags), doc))
                reranked.sort(key=lambda x: x[0], reverse=True)

                top = reranked[: self.config.rag_top_k]

                must_docs = pick_must_have_docs(
                    all_docs=self.rag.docs,
                    must_have_any_tags=self.config.must_have_any_tags,
                    max_docs=self.config.must_have_max_docs,
                )

                evidence, _used = build_evidence(
                    must_docs=must_docs,
                    ranked_docs=top,
                    per_doc_char_limit=self.config.per_doc_char_limit,
                    max_evidence_chars=self.config.max_evidence_chars,
                )

                llm_out = call_gemini_json(
                    user_input=user_input,
                    evidence=evidence,
                    model=self.config.gemini_model,
                    api_key=self.config.gemini_api_key,
                    force_json=self.config.force_json,
                )

                bundle = {
                    "evidence": evidence,
                    "positive": llm_out.get("positive", ""),
                    "negative": llm_out.get("negative", ""),
                }

            except Exception as e:
                bundle = {"evidence": "", "positive": f"(오류)\n{e}", "negative": ""}

            self.root.after(0, lambda: self._apply_result(request_id, bundle))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_result(self, request_id: int, bundle: dict):
        if request_id != self._last_request_id:
            return

        self._set_text(self.out_evidence, bundle.get("evidence", ""))
        self._set_text(self.out_positive, bundle.get("positive", ""))
        self._set_text(self.out_negative, bundle.get("negative", ""))

        self.status_var.set("완료 (입력을 더 하면 1초 후 다시 추천됩니다.)")

    @staticmethod
    def _set_text(widget: tk.Text, text: str):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")


def main():
    # 키를 직접 넣고 싶다면 여기서 문자열로 넣어도 됨 (권장 X)
    # api_key = "YOUR_GEMINI_KEY"
    api_key = ""

    root = tk.Tk()
    config = AppConfig(
        debounce_ms=1000,
        rag_top_k=4,
        rag_pool_k=18,
        recipes_path="recipes_20.jsonl",
        embed_model_name="paraphrase-multilingual-MiniLM-L12-v2",
        gemini_model="gemini-3-flash-preview",
        gemini_api_key=api_key,
        include_tags_in_embedding=True,   # 요청한 방식: title + tags + text
        must_have_max_docs=8,             # 규칙이 많으면 6으로 줄이기 권장
    )

    DebounceRAGGeminiStrongApp(root, config)
    root.mainloop()


if __name__ == "__main__":
    main()
