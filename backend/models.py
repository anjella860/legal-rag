from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from datetime import datetime
from database import Base

class User(Base):
    __tablename__ = "users"
    id            = Column(Integer, primary_key=True)
    username      = Column(String(50), unique=True, nullable=False)
    email         = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role          = Column(String(20), default="USER")
    created_at    = Column(DateTime, default=datetime.utcnow)

class QAHistory(Base):
    __tablename__ = "qa_history"
    id         = Column(Integer, primary_key=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    question   = Column(Text, nullable=False)
    answer     = Column(Text, nullable=False)
    sources    = Column(Text)
    feedback   = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

