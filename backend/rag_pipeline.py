import os
from typing import List, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

SYSTEM_PROMPT = """당신은 한국 노동법령 전문 AI 어시스턴트입니다.
참고 법령 조문을 기반으로 답변하세요.

1. 관련된 법령 조문 내용을 바탕으로 답변해주세요.
2. 조문에 없는 내용은 "해당 법령에서 해당 정보를 찾을 수 없습니다."라고 답변해주세요.
3. 답변은 이해하기 쉬운 쉬운 말로 작성해주세요.
4. 답변 마지막에 참고한 조문을 포함해주세요.
   "이 답변은 참고용이며 법적 효력이 있는 해석이 아닙니다."
"""

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=SYSTEM_PROMPT + """
[참고 법령 조문]
{context}

[질문]
{question}

[답변]"""
)

def get_embed_model():
    return HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

async def ask_question(
    question: str,
    user_id: Optional[int] = None,
    law_names: Optional[List[str]] = None
) -> dict:
    embed_model = get_embed_model()

    vectordb = Chroma(
        persist_directory  = CHROMA_DIR,
        embedding_function = embed_model,
        collection_name    = "law_articles"
    )

    search_kwargs = {"k": 4}

    if law_names:
        search_kwargs["filter"] = {
            "law_name": {"$in": law_names}
        }

    retriever = vectordb.as_retriever(search_kwargs=search_kwargs)

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    chain = RetrievalQA.from_chain_type(
        llm               = llm,
        retriever         = retriever,
        chain_type_kwargs = {"prompt": RAG_PROMPT},
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
