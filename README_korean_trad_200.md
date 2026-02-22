# Korean Traditional / Joseon Minhwa RAG Recipes (200)

목표: '민화풍'이 동화책/귀여운 일러스트로 새는 문제를 막으면서,
첨부 이미지처럼 **다양한 한국 전통 회화 계열**이 프롬프트 추천에 반영되도록 만든 200개 레시피(JSONL)입니다.

## 포함 계열(예시)
- 고구려 고분벽화 사신도(청룡/백호/주작/현무) — mural/plaster 질감
- 신라 천마도(천마총 장니) — aged leather/paper 질감
- 조선 초충도(벌레+풀) — 섬세 선묘 + 담채
- 조선 풍속화/씨름도 — 군상 선묘
- 산수 두루마리(몽유도원도 계열) — 파노라마 스크롤, 담묵+연채
- (기존) 조선 민화 기본 규칙/팔레트/네거티브 가드레일

## 파일
- recipes_korean_trad_200.jsonl  (MH001~MH200)

## 구조(JSONL)
각 줄 = 1개 레시피(JSON)
- id: MH001~MH200
- title: 제목
- tags: 분류/가드레일/장르 태그
- text: 검색 임베딩 + Gemini evidence에 전달되는 본문(규칙/힌트/네거티브 포함)

## 적용(데모 코드)
AppConfig에서:
- recipes_path="recipes_korean_trad_200.jsonl"
- top_k(또는 rag_top_k): 3~5 권장

## 팁
- '고정력 강화(strong) 버전'을 쓰면
  must_have_any_tags에 core/negative:storybook/negative:photoreal/negative:text 같은 태그를 넣어
  항상 evidence에 포함되게 할 수 있습니다.
