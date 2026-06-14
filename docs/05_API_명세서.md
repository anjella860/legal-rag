# 📋 API 명세서

> **프로젝트명:** 법령 RAG 문서 검색 시스템 (Legal RAG Document Search System)
> **작성일:** 2026년 6월
> **작성자:** 전보경
> **버전:** v1.0
> **Base URL:** `http://localhost:8001/api/v1`

---

## 1. 공통 사항

### 1.1 응답 포맷

모든 API는 다음 공통 응답 포맷을 따른다.

```json
{
  "success": true,
  "message": "요청이 성공적으로 처리되었습니다.",
  "data": { },
  "error": null,
  "timestamp": "2026-06-14T10:00:00.000000"
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| success | boolean | 요청 성공 여부 |
| message | string | 성공 메시지 |
| data | object \| null | 응답 데이터 |
| error | string \| null | 에러 메시지 (실패 시) |
| timestamp | string | 응답 시각 (ISO 8601) |

### 1.2 인증

회원 전용 API는 Authorization 헤더에 JWT Bearer Token을 포함해야 한다.

```
Authorization: Bearer {access_token}
```

### 1.3 HTTP 상태 코드

| 코드 | 설명 |
|------|------|
| 200 | 요청 성공 |
| 201 | 생성 성공 |
| 400 | 잘못된 요청 (유효성 검사 실패) |
| 401 | 인증 실패 (토큰 없음 또는 만료) |
| 403 | 권한 없음 |
| 404 | 리소스 없음 |
| 409 | 충돌 (이미 존재하는 데이터) |
| 500 | 서버 내부 오류 |

---

## 2. 인증 API

### 2.1 회원가입

```
POST /auth/signup
```

**Request Body**

```json
{
  "username": "bokyung",
  "email": "bokyung@example.com",
  "password": "password1234!"
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| username | string | ✅ | 사용자명 (2~50자) |
| email | string | ✅ | 이메일 형식 |
| password | string | ✅ | 비밀번호 (8자 이상) |

**Response (201)**

```json
{
  "success": true,
  "message": "회원가입이 완료되었습니다.",
  "data": {
    "user_id": 1,
    "username": "bokyung",
    "email": "bokyung@example.com"
  },
  "error": null,
  "timestamp": "2026-06-14T10:00:00.000000"
}
```

**Error Cases**

| 상태 코드 | 에러 메시지 |
|----------|------------|
| 409 | 이미 사용 중인 이메일입니다. |
| 409 | 이미 사용 중인 사용자명입니다. |
| 400 | 비밀번호는 8자 이상이어야 합니다. |

---

### 2.2 로그인

```
POST /auth/login
```

**Request Body**

```json
{
  "email": "bokyung@example.com",
  "password": "password1234!"
}
```

**Response (200)**

```json
{
  "success": true,
  "message": "로그인이 완료되었습니다.",
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIs...",
    "token_type": "bearer",
    "username": "bokyung"
  },
  "error": null,
  "timestamp": "2026-06-14T10:00:00.000000"
}
```

**Error Cases**

| 상태 코드 | 에러 메시지 |
|----------|------------|
| 401 | 이메일 또는 비밀번호가 올바르지 않습니다. |

---

## 3. 질의응답 API

### 3.1 질문하기

```
POST /qa/ask
```

**권한:** 비회원·회원 모두 사용 가능 (회원은 히스토리 저장)

**Request Header (회원인 경우)**

```
Authorization: Bearer {access_token}
```

**Request Body**

```json
{
  "question": "연차는 며칠 쓸 수 있나요?",
  "law_names": ["근로기준법"]
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| question | string | ✅ | 자연어 질문 (1~500자) |
| law_names | array | ❌ | 검색 대상 법령 필터 (미입력 시 전체 검색) |

**Response (200)**

```json
{
  "success": true,
  "message": "요청이 성공적으로 처리되었습니다.",
  "data": {
    "answer": "근로기준법 제60조에 따르면, 1년간 80% 이상 출근한 근로자에게는 15일의 유급휴가가 주어집니다...\n\n※ 본 답변은 참고용이며 법적 효력이 있는 해석이 아닙니다.",
    "sources": [
      {
        "law_name": "근로기준법",
        "article_no": "제60조",
        "article_title": "연차 유급휴가",
        "content": "사용자는 1년간 80퍼센트 이상 출근한 근로자에게 15일의 유급휴가를 주어야 한다..."
      },
      {
        "law_name": "근로기준법",
        "article_no": "제61조",
        "article_title": "연차 유급휴가의 사용 촉진",
        "content": "사용자가 제60조에 따른 유급휴가의 사용을 촉진하기 위하여..."
      }
    ],
    "saved": true
  },
  "error": null,
  "timestamp": "2026-06-14T10:00:00.000000"
}
```

| 응답 필드 | 타입 | 설명 |
|----------|------|------|
| answer | string | LLM 생성 답변 (면책 문구 포함) |
| sources | array | 근거 조문 목록 |
| sources[].law_name | string | 법령명 |
| sources[].article_no | string | 조문 번호 |
| sources[].article_title | string | 조문 제목 |
| sources[].content | string | 조문 원문 (200자 이내) |
| saved | boolean | 히스토리 저장 여부 (비회원: false) |

**Error Cases**

| 상태 코드 | 에러 메시지 |
|----------|------------|
| 400 | 질문을 입력해주세요. |
| 400 | 질문은 500자 이내로 입력해주세요. |
| 404 | 관련 법령 조문을 찾을 수 없습니다. |

---

### 3.2 히스토리 조회

```
GET /qa/history
```

**권한:** 회원 전용

**Request Header**

```
Authorization: Bearer {access_token}
```

**Query Parameters**

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| page | integer | ❌ | 페이지 번호 (기본값: 1) |
| size | integer | ❌ | 페이지 크기 (기본값: 10, 최대: 50) |

**Response (200)**

```json
{
  "success": true,
  "message": "요청이 성공적으로 처리되었습니다.",
  "data": {
    "items": [
      {
        "id": 5,
        "question": "연차는 며칠 쓸 수 있나요?",
        "answer": "근로기준법 제60조에 따르면...",
        "sources": [...],
        "created_at": "2026-06-14T10:00:00"
      }
    ],
    "total": 5,
    "page": 1,
    "size": 10
  },
  "error": null,
  "timestamp": "2026-06-14T10:00:00.000000"
}
```

**Error Cases**

| 상태 코드 | 에러 메시지 |
|----------|------------|
| 401 | 로그인이 필요합니다. |

---

## 4. 법령 관리 API (관리자 전용)

### 4.1 법령 데이터 적재

```
POST /admin/laws/load
```

**권한:** ADMIN 역할 필요

**Request Header**

```
Authorization: Bearer {admin_access_token}
```

**Request Body**

```json
{
  "law_names": ["근로기준법", "최저임금법"]
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| law_names | array | ❌ | 적재할 법령명 목록 (미입력 시 전체 5개 적재) |

**Response (200)**

```json
{
  "success": true,
  "message": "법령 데이터 적재가 완료되었습니다.",
  "data": {
    "loaded": ["근로기준법", "최저임금법"],
    "skipped": [],
    "total_chunks": 140
  },
  "error": null,
  "timestamp": "2026-06-14T10:00:00.000000"
}
```

---

### 4.2 적재된 법령 목록 조회

```
GET /admin/laws
```

**권한:** ADMIN 역할 필요

**Response (200)**

```json
{
  "success": true,
  "message": "요청이 성공적으로 처리되었습니다.",
  "data": [
    {
      "law_name": "근로기준법",
      "chunk_count": 110,
      "collected_at": "2026-06-14"
    },
    {
      "law_name": "최저임금법",
      "chunk_count": 30,
      "collected_at": "2026-06-14"
    }
  ],
  "error": null,
  "timestamp": "2026-06-14T10:00:00.000000"
}
```

---

## 5. API 목록 요약

| Method | Endpoint | 설명 | 권한 |
|--------|----------|------|------|
| POST | /auth/signup | 회원가입 | 비회원 |
| POST | /auth/login | 로그인 | 비회원 |
| POST | /qa/ask | 질문하기 | 비회원·회원 |
| GET | /qa/history | 히스토리 조회 | 회원 전용 |
| POST | /admin/laws/load | 법령 데이터 적재 | 관리자 |
| GET | /admin/laws | 적재 법령 목록 조회 | 관리자 |
