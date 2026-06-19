import os
from typing import List, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import torch

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"임베딩 모델 디바이스: {device}")

_embed_model = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)
print("임베딩 모델 로딩 완료 (캐시됨)")

LAW_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""당신은 한국 법령 전문 AI 어시스턴트입니다.
참고 법령 조문을 기반으로 답변하세요.

1. 관련된 법령 조문 내용을 바탕으로 답변해주세요.
2. 조문에 없는 내용은 "해당 법령에서 해당 정보를 찾을 수 없습니다."라고 답변해주세요.
3. 답변은 이해하기 쉬운 말로 작성해주세요.
4. 답변 마지막에 참고한 조문을 포함해주세요.
   "이 답변은 참고용이며 법적 효력이 있는 해석이 아닙니다."

[참고 법령 조문]
{context}

[질문]
{question}

[답변]"""
)

PREC_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""당신은 한국 판례 전문 AI 어시스턴트입니다.
관련 판례를 기반으로 다음 순서로 분석해서 답변하세요.

1. 사건의 핵심 쟁점을 설명하세요.
2. 법원이 어떻게 판단했는지 설명하세요.
3. 판단 이유를 쉬운 말로 설명하세요.
4. 관련 법령 조문이 있으면 함께 정리하세요.
5. 사용자가 실무에서 참고할 수 있는 포인트를 정리하세요.
6. 판례에 없는 내용은 추측하지 마세요.
7. 마지막에 "이 답변은 참고용이며 법적 효력이 있는 해석이 아닙니다."를 포함하세요.

[참고 판례]
{context}

[질문]
{question}

[답변]"""
)

def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

async def ask_question(
    question: str,
    user_id: Optional[int] = None,
    law_names: Optional[List[str]] = None
) -> dict:

    vectordb = Chroma(
        persist_directory  = CHROMA_DIR,
        embedding_function = _embed_model,
        collection_name    = "law_articles"
    )

    search_kwargs = {"k": 3}
    if law_names:
        search_kwargs["filter"] = {"law_name": {"$in": law_names}}

    retriever = vectordb.as_retriever(search_kwargs=search_kwargs)

    chain = RetrievalQA.from_chain_type(
        llm               = get_llm(),
        retriever         = retriever,
        chain_type_kwargs = {"prompt": LAW_PROMPT},
        return_source_documents = True
    )

    result = chain.invoke({"query": question})

    sources = []
    seen = set()
    for doc in result.get("source_documents", []):
        law_name      = doc.metadata.get("law_name", "")
        article_no    = doc.metadata.get("article_no", "")
        article_title = doc.metadata.get("article_title", "")
        key = f"{law_name}_{article_no}"

        if key in seen:
            continue
        seen.add(key)

        sources.append({
            "law_name":      law_name,
            "article_no":    article_no,
            "article_title": article_title,
            "content":       doc.page_content[:300]
        })

    return {
        "answer":  result["result"],
        "sources": sources,
        "saved":   user_id is not None
    }

async def search_precedent(
    question: str,
    user_id: Optional[int] = None,
    category: Optional[str] = None
) -> dict:

    vectordb = Chroma(
        persist_directory  = CHROMA_DIR,
        embedding_function = _embed_model,
        collection_name    = "precedents"
    )

    search_kwargs = {"k": 3}
    if category:
        search_kwargs["filter"] = {"category": category}

    retriever = vectordb.as_retriever(search_kwargs=search_kwargs)

    chain = RetrievalQA.from_chain_type(
        llm               = get_llm(),
        retriever         = retriever,
        chain_type_kwargs = {"prompt": PREC_PROMPT},
        return_source_documents = True
    )

    result = chain.invoke({"query": question})

    sources = []
    seen = set()
    for doc in result.get("source_documents", []):
        case_name = doc.metadata.get("case_name", "")
        case_no   = doc.metadata.get("case_no", "")
        court     = doc.metadata.get("court", "")
        date      = doc.metadata.get("date", "")
        case_type = doc.metadata.get("case_type", "")
        key       = f"{case_no}"

        if key in seen:
            continue
        seen.add(key)

        sources.append({
            "case_name": case_name,
            "case_no":   case_no,
            "court":     court,
            "date":      date,
            "case_type": case_type,
            "content":   doc.page_content[:300]
        })

    return {
        "answer":  result["result"],
        "sources": sources,
        "saved":   user_id is not None
    }
