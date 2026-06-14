"""프로젝트 02 — RAG 문서 검색 시스템"""
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import Base, engine, get_db
from models import User, Document, QAHistory
from schemas import AskRequest, QAResponse
from auth import get_current_user, TokenData
from rag_pipeline import process_document, ask_question
import shutil, os, hashlib
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(title="RAG 문서 검색 시스템", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/v1/documents/upload", status_code=201)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    cur: TokenData = Depends(get_current_user)
):
    if file.content_type not in ["application/pdf", "text/plain",
                                  "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        raise HTTPException(400, "PDF, TXT, DOCX 파일만 업로드 가능합니다.")
    content  = await file.read()
    md5      = hashlib.md5(content).hexdigest()
    existing = db.query(Document).filter(Document.md5_hash == md5,
                                          Document.user_id == cur.user_id).first()
    if existing:
        raise HTTPException(409, "동일한 파일이 이미 업로드되어 있습니다.")
    file_path = os.path.join(UPLOAD_DIR, f"{cur.user_id}_{md5}_{file.filename}")
    with open(file_path, "wb") as f:
        f.write(content)
    doc = Document(
        user_id   = cur.user_id,
        filename  = file.filename,
        file_path = file_path,
        file_size = len(content),
        md5_hash  = md5,
        status    = "PROCESSING"
    )
    db.add(doc); db.commit(); db.refresh(doc)
    background_tasks.add_task(process_document, doc.id, file_path, db)
    return {"id": doc.id, "filename": doc.filename, "status": "PROCESSING"}

@app.get("/api/v1/documents")
async def list_documents(db: Session = Depends(get_db),
                         cur: TokenData = Depends(get_current_user)):
    docs = db.query(Document).filter(Document.user_id == cur.user_id).all()
    return [{"id": d.id, "filename": d.filename, "status": d.status,
             "chunk_count": d.chunk_count, "file_size": d.file_size} for d in docs]

@app.delete("/api/v1/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: int, db: Session = Depends(get_db),
                           cur: TokenData = Depends(get_current_user)):
    doc = db.query(Document).filter(Document.id == doc_id,
                                     Document.user_id == cur.user_id).first()
    if not doc: raise HTTPException(404, "문서를 찾을 수 없습니다.")
    if os.path.exists(doc.file_path): os.remove(doc.file_path)
    db.delete(doc); db.commit()

@app.post("/api/v1/qa/ask")
async def ask(req: AskRequest, db: Session = Depends(get_db),
              cur: TokenData = Depends(get_current_user)):
    result = await ask_question(req.question, cur.user_id, req.document_ids)
    qa = QAHistory(user_id=cur.user_id, question=req.question,
                   answer=result["answer"], sources=str(result["sources"]))
    db.add(qa); db.commit()
    return result

@app.get("/api/v1/qa/history")
async def qa_history(db: Session = Depends(get_db),
                     cur: TokenData = Depends(get_current_user)):
    items = (db.query(QAHistory).filter(QAHistory.user_id == cur.user_id)
               .order_by(QAHistory.created_at.desc()).limit(50).all())
    return [{"id": i.id, "question": i.question, "answer": i.answer,
             "created_at": i.created_at} for i in items]

if __name__ == "__main__":
    import uvicorn; uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
