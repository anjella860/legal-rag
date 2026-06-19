from fastapi import FastAPI, Depends, HTTPException
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from database import Base, engine, get_db
from models import User, QAHistory
from schemas import AskRequest, SignupRequest, LoginRequest
from response import ApiResponse
from auth import get_current_user, TokenData
from jwt_utils import (
    hash_password, verify_password,
    create_access_token, decode_token
)
from rag_pipeline import (
    ask_question,
    search_precedent,
    search_interpretation,
    search_constitutional,
    search_integrated
)
from contextlib import asynccontextmanager
from pydantic import BaseModel
import json
from dotenv import load_dotenv
load_dotenv()

@asynccontextmanager
async def lifespan(app):
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(title="Legal RAG API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

AVAILABLE_LAWS = {
    "노동/고용": [
        "근로기준법",
        "최저임금법",
        "근로자퇴직급여 보장법",
        "남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률",
        "고용보험법",
    ],
    "주거/임대": [
        "주택임대차보호법",
        "상가건물 임대차보호법",
        "집합건물의 소유 및 관리에 관한 법률",
    ],
    "소비자/생활": [
        "소비자기본법",
        "전자상거래 등에서의 소비자보호에 관한 법률",
        "할부거래에 관한 법률",
    ],
    "개인정보/디지털": [
        "개인정보 보호법",
        "정보통신망 이용촉진 및 정보보호 등에 관한 법률",
    ],
    "청년/취업": [
        "청년고용촉진 특별법",
        "직업안정법",
    ],
}

CATEGORIES = ["노동/고용", "주거/임대", "소비자/생활", "개인정보/디지털", "청년/취업"]

security = HTTPBearer(auto_error=False)

class SearchRequest(BaseModel):
    question: str
    category: Optional[str] = None

class IntegratedSearchRequest(BaseModel):
    question: str
    category: Optional[str] = None

async def get_optional_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[TokenData]:
    if not credentials:
        return None
    try:
        payload = decode_token(credentials.credentials)
        if payload.get("type") != "access":
            return None
        return TokenData(
            user_id  = payload["user_id"],
            username = payload["username"],
            role     = payload.get("role", "USER")
        )
    except Exception:
        return None

@app.post("/api/v1/auth/signup", status_code=201)
async def signup(req: SignupRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(409, "이미 사용 중인 이메일입니다.")
    if db.query(User).filter(User.username == req.username).first():
        raise HTTPException(409, "이미 사용 중인 사용자명입니다.")
    user = User(
        username      = req.username,
        email         = req.email,
        password_hash = hash_password(req.password),
        role          = "USER"
    )
    db.add(user); db.commit(); db.refresh(user)
    return ApiResponse.ok(
        data    = {"user_id": user.id, "username": user.username, "email": user.email},
        message = "회원가입이 완료되었습니다."
    )

@app.post("/api/v1/auth/login")
async def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(401, "이메일 또는 비밀번호가 올바르지 않습니다.")
    token = create_access_token({
        "user_id":  user.id,
        "username": user.username,
        "role":     user.role
    })
    return ApiResponse.ok(
        data    = {"access_token": token, "token_type": "bearer", "username": user.username},
        message = "로그인이 완료되었습니다."
    )

@app.get("/api/v1/laws")
async def get_laws():
    return ApiResponse.ok(data=AVAILABLE_LAWS)

@app.get("/api/v1/categories")
async def get_categories():
    return ApiResponse.ok(data=CATEGORIES)

@app.post("/api/v1/qa/ask")
async def ask(
    req: AskRequest,
    db: Session = Depends(get_db),
    cur: Optional[TokenData] = Depends(get_optional_user)
):
    result = await ask_question(
        question  = req.question,
        user_id   = cur.user_id if cur else None,
        law_names = req.law_names
    )
    if cur:
        qa = QAHistory(
            user_id  = cur.user_id,
            question = req.question,
            answer   = result["answer"],
            sources  = json.dumps(result["sources"], ensure_ascii=False)
        )
        db.add(qa); db.commit()
    return ApiResponse.ok(data=result)

@app.post("/api/v1/prec/search")
async def prec_search(
    req: SearchRequest,
    db: Session = Depends(get_db),
    cur: Optional[TokenData] = Depends(get_optional_user)
):
    result = await search_precedent(
        question = req.question,
        user_id  = cur.user_id if cur else None,
        category = req.category
    )
    if cur:
        qa = QAHistory(
            user_id  = cur.user_id,
            question = f"[판례] {req.question}",
            answer   = result["answer"],
            sources  = json.dumps(result["sources"], ensure_ascii=False)
        )
        db.add(qa); db.commit()
    return ApiResponse.ok(data=result)

@app.post("/api/v1/expc/search")
async def expc_search(
    req: SearchRequest,
    db: Session = Depends(get_db),
    cur: Optional[TokenData] = Depends(get_optional_user)
):
    result = await search_interpretation(
        question = req.question,
        user_id  = cur.user_id if cur else None,
        category = req.category
    )
    if cur:
        qa = QAHistory(
            user_id  = cur.user_id,
            question = f"[법령해석례] {req.question}",
            answer   = result["answer"],
            sources  = json.dumps(result["sources"], ensure_ascii=False)
        )
        db.add(qa); db.commit()
    return ApiResponse.ok(data=result)

@app.post("/api/v1/detc/search")
async def detc_search(
    req: SearchRequest,
    db: Session = Depends(get_db),
    cur: Optional[TokenData] = Depends(get_optional_user)
):
    result = await search_constitutional(
        question = req.question,
        user_id  = cur.user_id if cur else None,
        category = req.category
    )
    if cur:
        qa = QAHistory(
            user_id  = cur.user_id,
            question = f"[헌재결정례] {req.question}",
            answer   = result["answer"],
            sources  = json.dumps(result["sources"], ensure_ascii=False)
        )
        db.add(qa); db.commit()
    return ApiResponse.ok(data=result)

@app.post("/api/v1/integrated/search")
async def integrated_search(
    req: IntegratedSearchRequest,
    db: Session = Depends(get_db),
    cur: Optional[TokenData] = Depends(get_optional_user)
):
    result = await search_integrated(
        question = req.question,
        user_id  = cur.user_id if cur else None,
        category = req.category
    )
    if cur:
        qa = QAHistory(
            user_id  = cur.user_id,
            question = f"[통합검색] {req.question}",
            answer   = result["answer"],
            sources  = json.dumps(result["sources"], ensure_ascii=False)
        )
        db.add(qa); db.commit()
    return ApiResponse.ok(data=result)

@app.get("/api/v1/qa/history")
async def qa_history(
    page: int = 1,
    size: int = 10,
    db: Session = Depends(get_db),
    cur: TokenData = Depends(get_current_user)
):
    offset = (page - 1) * size
    items = (
        db.query(QAHistory)
        .filter(QAHistory.user_id == cur.user_id)
        .order_by(QAHistory.created_at.desc())
        .offset(offset)
        .limit(size)
        .all()
    )
    total = db.query(QAHistory).filter(QAHistory.user_id == cur.user_id).count()
    return ApiResponse.ok(data={
        "items": [
            {
                "id":         i.id,
                "question":   i.question,
                "answer":     i.answer,
                "sources":    json.loads(i.sources) if i.sources else [],
                "created_at": i.created_at.isoformat()
            }
            for i in items
        ],
        "total": total,
        "page":  page,
        "size":  size
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
