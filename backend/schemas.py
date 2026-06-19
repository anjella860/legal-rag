from pydantic import BaseModel, EmailStr
from typing import Optional, List

class SignupRequest(BaseModel):
    username: str
    email:    EmailStr
    password: str

class LoginRequest(BaseModel):
    email:    EmailStr
    password: str

class AskRequest(BaseModel):
    question:  str
    law_names: Optional[List[str]] = None
