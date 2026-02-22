# 고정력 강화 버전: Debounce + RAG + Gemini

## 무엇이 강화됐나?
1) 임베딩에 `title + tags + text`를 합쳐서 검색할 수 있음  
2) 민화/조선 스타일을 깨는 것을 막기 위한 **필수 규칙(가드레일)** 문서를 **항상 evidence에 포함**  
3) 1차 검색 결과를 `태그 보너스`로 간단 rerank

## 실행
```bash
pip install -U sentence-transformers faiss-cpu numpy google-genai
python debounce_rag_prompt_gui_gemini_strong.py
```

## 추천 튜닝 포인트
- 너무 '규칙이 많다'고 느껴지면: `must_have_max_docs`를 8 → 6으로 줄이기
- 검색이 엉뚱하면: `include_tags_in_embedding=False`로 바꿔서 tags를 임베딩에서 빼고 rerank만 사용
- 더 강하게 고정하고 싶으면:
  - must_have_any_tags에 추가
  - infer_query_tags()에 키워드-태그 매핑 추가
