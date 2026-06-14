"""
공통 JWT 유틸리티 (FastAPI 프로젝트 전용)
모든 FastAPI 기반 프로젝트에서 공유하여 사용
"""
from datetime import datetime, timedelta
from typing import Optional
import jwt
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel

SECRET_KEY  = "your-256-bit-secret-key-change-in-production"
ALGORITHM   = "HS256"
ACCESS_EXP  = 60 * 24        # 24시간(분)
REFRESH_EXP = 60 * 24 * 7   # 7일(분)

pwd_ctx  = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

class TokenData(BaseModel):
    user_id: int
    username: str
    role: str = "USER"

def hash_password(plain: str) -> str:
    return pwd_ctx.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)

def create_access_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"]  = datetime.utcnow() + timedelta(minutes=ACCESS_EXP)
    payload["type"] = "access"
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"]  = datetime.utcnow() + timedelta(minutes=REFRESH_EXP)
    payload["type"] = "refresh"
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="토큰이 만료되었습니다.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> TokenData:
    payload = decode_token(credentials.credentials)
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Access Token이 필요합니다.")
    return TokenData(
        user_id  = payload["user_id"],
        username = payload["username"],
        role     = payload.get("role", "USER")
    )

async def require_admin(current: TokenData = Security(get_current_user)) -> TokenData:
    if current.role != "ADMIN":
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")
    return current
