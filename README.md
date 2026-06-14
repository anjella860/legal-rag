# ⚖️ 법령 RAG 문서 검색 시스템

> 자연어로 묻고, 법령에서 찾고, 쉽게 이해하는 노동법령 AI 검색 서비스

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)
![Vue.js](https://img.shields.io/badge/Vue.js-3.4-4FC08D?style=flat&logo=vue.js&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=flat)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-FF6B35?style=flat)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?style=flat&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)

---

## 📌 프로젝트 소개

일반 시민이 연차·퇴직금·최저임금 같은 노동 관련 궁금증을 해결하려면 방대한 법령 원문을 직접 찾아 읽어야 한다. 전문 용어가 많고 조문 구조가 복잡해 원하는 정보를 빠르게 파악하기 어렵다.

**법령 RAG 문서 검색 시스템**은 RAG(Retrieval-Augmented Generation) 기술을 활용하여 자연어 질문에 대해 관련 법령 조문을 검색하고, LLM이 누구나 이해하기 쉬운 답변을 생성한다. 답변과 함께 근거 조문을 함께 제공하여 신뢰성을 높였다.

> ※ 본 서비스의 답변은 참고용이며 법적 효력이 있는 해석이 아닙니다.

---

## 🎯 주요 기능

### 비회원
- 자연어로 노동법령 질문 입력
- 관련 조문 기반 AI 답변 + 출처 조문 확인
- 법령별 필터 검색 (근로기준법, 최저임금법 등)

### 회원
- 비회원 기능 전체 포함
- 질의응답 히스토리 저장 및 조회
- 이전 질문 다시 확인

---

## 🏗️ 시스템 아키텍처

```
[Vue.js Frontend]
       │ REST API
       ▼
[FastAPI Backend]
       │
       ├─ 질문 임베딩 (ko-sroberta-multitask)
       ├─ 벡터 검색 (ChromaDB)
       ├─ 프롬프트 구성 (LangChain)
       └─ 답변 생성 (GPT-4o-mini)
       │
       ├─ [PostgreSQL] — 회원 정보, QA 히스토리
       └─ [ChromaDB]   — 법령 조문 벡터
```

---

## 🤖 RAG 파이프라인

### 법령 적재 흐름 (최초 1회)
```
국가법령정보센터 Open API
→ 조문(Article) 단위 파싱
→ 메타데이터 태깅 (법령명·조문번호·조문제목)
→ ko-sroberta-multitask 임베딩
→ ChromaDB 저장
```

### 질의응답 흐름
```
자연어 질문
→ 질문 임베딩 (ko-sroberta)
→ ChromaDB 유사도 검색 (Top-4)
→ LangChain 프롬프트 구성
→ GPT-4o-mini 답변 생성
→ 답변 + 출처 조문 반환
```

**강사님 베이스 코드 대비 주요 변경사항**

| 항목 | 베이스 코드 | 본 프로젝트 |
|------|------------|------------|
| 데이터 소스 | 사용자 파일 업로드 | 국가법령정보센터 API 사전 적재 |
| 임베딩 모델 | OpenAI text-embedding-3-small | jhgan/ko-sroberta-multitask (무료, 한국어 특화) |
| 청크 방식 | 800자 단위 분할 | 조문 단위 파싱 |
| LLM | GPT-4o | GPT-4o-mini (비용 절감) |

---

## 🛠️ 기술 스택

| 분류 | 기술 | 선택 이유 |
|------|------|-----------|
| AI / RAG | LangChain | RAG 파이프라인 표준 프레임워크 |
| 임베딩 | jhgan/ko-sroberta-multitask | 한국어 특화, 무료 로컬 실행 |
| 벡터 DB | ChromaDB | 로컬 실행, 개인 프로젝트 규모에 적합 |
| LLM | GPT-4o-mini | 비용 효율적, 한국어 답변 품질 우수 |
| 백엔드 | FastAPI (Python 3.11) | 비동기 처리, AI 라이브러리 연동 용이 |
| 프론트엔드 | Vue.js 3 + Vite | 익숙한 스택, 빠른 개발 |
| 스타일링 | Tailwind CSS | 빠른 UI 구성 |
| DB | PostgreSQL 16 | 회원·히스토리 관계형 데이터 관리 |
| 인증 | JWT (PyJWT + bcrypt) | Stateless 인증 |
| 컨테이너 | Docker Compose | 환경 일관성 보장 |
| 법령 데이터 | 국가법령정보센터 Open API | 무료, 공식 법령 원문 |

---

## 📁 프로젝트 구조

```
legal-rag/
├── backend/
│   ├── main.py              # FastAPI 앱 진입점
│   ├── auth.py              # 인증 API
│   ├── rag_pipeline.py      # RAG 파이프라인
│   ├── law_loader.py        # 법령 적재 스크립트
│   ├── models.py            # SQLAlchemy ORM
│   ├── database.py          # DB 연결
│   ├── jwt_utils.py         # JWT 유틸리티
│   ├── schemas.py           # Pydantic 스키마
│   ├── response.py          # 공통 응답 포맷
│   ├── requirements.txt
│   └── .env
├── frontend/
│   ├── src/
│   │   ├── views/
│   │   │   ├── LoginView.vue    # 로그인·회원가입
│   │   │   ├── MainView.vue     # 메인 검색 화면
│   │   │   └── HistoryView.vue  # 히스토리 조회
│   │   ├── stores/
│   │   │   └── auth.js          # Pinia 인증 스토어
│   │   └── router/
│   │       └── index.js         # Vue Router
│   ├── package.json
│   └── vite.config.js
├── docs/
│   ├── 01_프로젝트_개요서.md
│   ├── 02_요구사항_정의서.md
│   ├── 03_시스템_아키텍처_설계서.md
│   ├── 04_RAG_파이프라인_설계서.md
│   └── 05_API_명세서.md
├── chroma_db/               # ChromaDB 벡터 저장소
└── docker-compose.yml
```

---

## 🚀 실행 방법

### 사전 준비

- Docker Desktop 설치
- OpenAI API 키 발급
- 국가법령정보센터 Open API 키 발급

### 1. 저장소 클론

```bash
git clone https://github.com/anjella860/legal-rag.git
cd legal-rag
```

### 2. 환경변수 설정

```bash
cp backend/.env.example backend/.env
# .env 파일에 API 키 입력
```

```env
OPENAI_API_KEY=your-openai-api-key
LAW_API_KEY=your-law-api-key
SECRET_KEY=your-jwt-secret-key
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/legal_rag_db
```

### 3. Docker Compose 실행

```bash
docker-compose up -d
```

### 4. 법령 데이터 적재 (최초 1회)

```bash
docker exec -it legal-rag-backend python law_loader.py
```

### 5. 접속

| 서비스 | URL |
|--------|-----|
| 프론트엔드 | http://localhost:5173 |
| FastAPI Swagger | http://localhost:8001/docs |

---

## 📋 적용 법령

| 법령명 | 주요 내용 |
|--------|-----------|
| 근로기준법 | 근로계약, 임금, 근로시간, 휴가, 해고 |
| 최저임금법 | 최저임금 기준, 적용 대상 |
| 근로자퇴직급여 보장법 | 퇴직금·퇴직연금 지급 기준 |
| 남녀고용평등법 | 육아휴직, 출산휴가, 성차별 금지 |
| 고용보험법 | 실업급여 수급 조건 및 절차 |

---

## 🔍 기술적 고민 & 해결

**Q. 왜 조문 단위로 청크를 나눴나요?**
항(①②③) 단위로 더 잘게 쪼갤 수도 있었지만, 항은 단독으로 의미가 완결되지 않는 경우가 많아 조문 전체를 하나의 청크로 유지했다. 검색 시 관련 항이 함께 반환되어 컨텍스트 손실을 방지할 수 있다.

**Q. 왜 OpenAI 임베딩 대신 ko-sroberta를 선택했나요?**
한국어 법률 용어 간 의미 유사도를 더 정확하게 잡아낸다. "연차"와 "유급휴가"처럼 동의어 검색이 잘 되고, 로컬 실행이라 임베딩 비용이 없다.

**Q. 비회원도 질문할 수 있게 한 이유는?**
법령 정보는 누구나 접근할 수 있어야 한다는 서비스 철학 때문이다. 단, 히스토리 저장은 회원 전용으로 차별화하여 회원가입 유인을 만들었다.

---

## 🔧 향후 개선사항

- 법령 개정 자동 감지 및 재적재
- 연관 질문 자동 추천 기능
- 자주 묻는 질문 TOP 10 통계
- 소셜 로그인 (카카오·네이버)

---

## 📚 참고 자료

- [국가법령정보센터 Open API](https://www.law.go.kr/LSO/openApi/intro.do)
- [LangChain 공식 문서](https://python.langchain.com)
- [ChromaDB 공식 문서](https://docs.trychroma.com)
- [jhgan/ko-sroberta-multitask](https://huggingface.co/jhgan/ko-sroberta-multitask)
