"""RAG 파이프라인 — 문서 처리 + 벡터 검색 + GPT 응답 생성"""
import os
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from sqlalchemy.orm import Session
from models import Document

CHROMA_DIR  = "./chroma_db"
EMBED_MODEL = OpenAIEmbeddings(model="text-embedding-3-small")

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""아래 문서 내용을 바탕으로 질문에 답변하세요.
문서에 없는 내용은 "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 답변하세요.

문서 내용:
{context}

질문: {question}

답변:"""
)

def process_document(doc_id: int, file_path: str, db: Session):
    """문서를 청크로 분할하고 벡터 DB에 저장"""
    try:
        # 파일 로드
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")

        pages    = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks   = splitter.split_documents(pages)

        # 메타데이터에 doc_id 추가
        for chunk in chunks:
            chunk.metadata["doc_id"] = str(doc_id)

        # ChromaDB에 저장
        Chroma.from_documents(
            documents        = chunks,
            embedding        = EMBED_MODEL,
            persist_directory= CHROMA_DIR,
            collection_name  = "documents"
        )

        # DB 상태 업데이트
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc:
            doc.chunk_count = len(chunks)
            doc.status      = "READY"
            db.commit()

    except Exception as e:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc:
            doc.status = "ERROR"
            db.commit()

async def ask_question(question: str, user_id: int,
                       document_ids: Optional[List[int]] = None) -> dict:
    """질문에 대해 RAG 기반 답변 생성"""
    vectordb = Chroma(
        persist_directory = CHROMA_DIR,
        embedding_function= EMBED_MODEL,
        collection_name   = "documents"
    )

    # 검색 필터 (특정 문서만 검색)
    search_kwargs = {"k": 4}
    if document_ids:
        search_kwargs["filter"] = {"doc_id": {"$in": [str(d) for d in document_ids]}}

    retriever = vectordb.as_retriever(search_kwargs=search_kwargs)

    llm  = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = RetrievalQA.from_chain_type(
        llm            = llm,
        retriever      = retriever,
        chain_type_kwargs = {"prompt": RAG_PROMPT},
        return_source_documents = True
    )

    result  = chain.invoke({"query": question})
    sources = []
    for doc in result.get("source_documents", []):
        sources.append({
            "doc_id":  doc.metadata.get("doc_id"),
            "page":    doc.metadata.get("page", 0),
            "content": doc.page_content[:200]
        })

    return {"answer": result["result"], "sources": sources}
