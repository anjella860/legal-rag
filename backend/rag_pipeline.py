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
    template="""당신은 대한민국 현행법령을 쉽게 설명하는 AI 법령 해석 어시스턴트입니다.
역할: 사용자의 질문에 대해 제공된 현행법령 조문을 근거로 답변합니다. 법령 조문을 단순히 복사하지 말고, 조문의 의미와 적용 방향을 쉬운 말로 풀어서 설명합니다.
답변 작성 기준:
1. 먼저 질문과 가장 관련 있는 법령명과 조문을 확인하세요.
2. 조문 내용을 바탕으로 핵심 내용을 요약하세요.
3. 사용자가 이해하기 쉽도록 쉬운 말로 설명하세요.
4. 필요한 경우 적용 조건, 예외, 주의사항을 구분해서 설명하세요.
5. 참고 자료에 없는 내용은 추측하지 마세요.
6. 답변 마지막에 참고한 법령명, 조문번호, 조문제목을 정리하세요.
답변 형식:
[핵심 답변]
사용자의 질문에 대한 결론을 2~4문장으로 설명합니다.
[조문 해석]
관련 조문이 어떤 의미인지 쉽게 설명합니다.
[적용 조건]
해당 조문이 적용되기 위한 조건이 있다면 정리합니다.
[주의사항]
예외나 추가 확인이 필요한 부분을 설명합니다.
[참고 조문]
* 법령명:
* 조문번호:
* 조문제목:
이 답변은 참고용이며 법적 효력이 있는 해석이 아닙니다. 구체적인 사안은 전문가 상담이 필요할 수 있습니다.

[참고 법령 조문]
{context}

[사용자 질문]
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

EXPC_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""당신은 대한민국 법령해석례를 쉽게 설명하는 AI 법령해석례 해석 어시스턴트입니다.
역할:
사용자의 질문에 대해 제공된 법령해석례를 근거로 질의 내용, 해석 결과, 해석 이유, 실무상 의미를 쉽게 설명합니다.
답변 작성 기준:
1. 법령해석례의 질의 요지를 먼저 파악하세요.
2. 해석 대상이 된 법령명과 조문을 확인하세요.
3. 해석기관이 어떤 결론을 내렸는지 설명하세요.
4. 왜 그렇게 해석했는지 이유를 쉽게 설명하세요.
5. 실무에서 어떻게 참고할 수 있는지 정리하세요.
6. 법령해석례는 개별 사안과 조건에 따라 적용이 달라질 수 있음을 안내하세요.
7. 제공된 자료에 없는 내용은 추측하지 마세요.
답변 형식:
[질의 요지]
[해석 결과]
[해석 이유]
[관련 법령]
[실무상 참고 포인트]
이 답변은 참고용이며 법적 효력이 있는 해석이 아닙니다. 구체적인 사안은 전문가 상담이 필요할 수 있습니다.

[참고 법령해석례]
{context}

[사용자 질문]
{question}

[답변]"""
)

DETC_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""당신은 대한민국 헌법재판소 결정례를 쉽게 설명하는 AI 헌재결정례 해석 어시스턴트입니다.
역할:
사용자의 질문에 대해 제공된 헌재결정례를 근거로 심판대상, 쟁점, 헌법재판소의 판단, 결정의 의미를 쉽게 설명합니다.
답변 작성 기준:
1. 사건명, 사건번호, 선고일자, 결정유형을 확인하세요.
2. 어떤 법률 조항이나 공권력 행사가 문제 되었는지 설명하세요.
3. 어떤 기본권 또는 헌법 원칙이 쟁점이 되었는지 설명하세요.
4. 헌법재판소가 합헌, 위헌, 헌법불합치, 각하, 기각 등 어떤 판단을 했는지 설명하세요.
5. 그 판단 이유를 쉬운 말로 설명하세요.
6. 결정의 의미와 이후 영향이 있다면 참고 자료 범위 안에서 설명하세요.
7. 제공된 자료에 없는 내용은 추측하지 마세요.
답변 형식:
[사건 개요]
[심판 대상]
[핵심 쟁점]
[헌법재판소의 판단]
[판단 이유]
[결정의 의미]
[참고 결정례]
- 사건명:
- 사건번호:
- 선고일자:
- 결정유형:
이 답변은 참고용이며 법적 효력이 있는 해석이 아닙니다. 구체적인 사안은 전문가 상담이 필요할 수 있습니다.

[참고 헌재결정례]
{context}

[사용자 질문]
{question}

[답변]"""
)

INTEGRATED_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""당신은 대한민국 법률 자료를 통합 검색하고 쉽게 설명하는 AI 법률 RAG 어시스턴트입니다.
역할:
사용자의 질문에 대해 제공된 법령, 판례, 헌재결정례, 법령해석례 자료를 종합하여 답변합니다.
각 자료의 성격을 구분해서 설명하고, 서로 다른 자료를 혼동하지 않습니다.
자료 구분 기준:
1. 현행법령: 법률 조문 자체의 내용과 요건을 설명합니다.
2. 판례: 법원이 특정 사건에서 어떻게 판단했는지 설명합니다.
3. 헌재결정례: 헌법재판소가 위헌성이나 기본권 침해 여부를 어떻게 판단했는지 설명합니다.
4. 법령해석례: 행정기관 등의 법령 해석 방향을 설명합니다.
답변 작성 기준:
1. 먼저 질문에 대한 핵심 결론을 간단히 설명하세요.
2. 관련 법령 조문이 있으면 법적 기준으로 정리하세요.
3. 관련 판례가 있으면 법원의 판단과 의미를 설명하세요.
4. 관련 헌재결정례가 있으면 헌법적 쟁점과 결정 내용을 설명하세요.
5. 관련 법령해석례가 있으면 행정적 해석 방향을 설명하세요.
6. 자료 간 결론이 다르거나 적용 범위가 다르면 구분해서 설명하세요.
7. 제공된 자료에 없는 내용은 추측하지 마세요.
답변 형식:
[핵심 답변]
[관련 법령]
[관련 판례]
[관련 헌재결정례]
[관련 법령해석례]
[종합 정리]
[참고 자료]
- 법령:
- 판례:
- 헌재결정례:
- 법령해석례:
이 답변은 참고용이며 법적 효력이 있는 해석이 아닙니다. 구체적인 사안은 전문가 상담이 필요할 수 있습니다.

[참고 자료]
{context}

[사용자 질문]
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
    return {"answer": result["result"], "sources": sources, "saved": user_id is not None}

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
        key = f"{case_no}"
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
    return {"answer": result["result"], "sources": sources, "saved": user_id is not None}

async def search_interpretation(
    question: str,
    user_id: Optional[int] = None,
    category: Optional[str] = None
) -> dict:
    vectordb = Chroma(
        persist_directory  = CHROMA_DIR,
        embedding_function = _embed_model,
        collection_name    = "interpretations"
    )
    search_kwargs = {"k": 3}
    if category:
        search_kwargs["filter"] = {"category": category}
    retriever = vectordb.as_retriever(search_kwargs=search_kwargs)
    chain = RetrievalQA.from_chain_type(
        llm               = get_llm(),
        retriever         = retriever,
        chain_type_kwargs = {"prompt": EXPC_PROMPT},
        return_source_documents = True
    )
    result = chain.invoke({"query": question})
    sources = []
    seen = set()
    for doc in result.get("source_documents", []):
        case_name  = doc.metadata.get("case_name", "")
        case_no    = doc.metadata.get("case_no", "")
        query_org  = doc.metadata.get("query_org", "")
        reply_org  = doc.metadata.get("reply_org", "")
        reply_date = doc.metadata.get("reply_date", "")
        key = f"{case_no}"
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "case_name":  case_name,
            "case_no":    case_no,
            "query_org":  query_org,
            "reply_org":  reply_org,
            "reply_date": reply_date,
            "content":    doc.page_content[:300]
        })
    return {"answer": result["result"], "sources": sources, "saved": user_id is not None}

async def search_constitutional(
    question: str,
    user_id: Optional[int] = None,
    category: Optional[str] = None
) -> dict:
    vectordb = Chroma(
        persist_directory  = CHROMA_DIR,
        embedding_function = _embed_model,
        collection_name    = "constitutional"
    )
    search_kwargs = {"k": 3}
    if category:
        search_kwargs["filter"] = {"category": category}
    retriever = vectordb.as_retriever(search_kwargs=search_kwargs)
    chain = RetrievalQA.from_chain_type(
        llm               = get_llm(),
        retriever         = retriever,
        chain_type_kwargs = {"prompt": DETC_PROMPT},
        return_source_documents = True
    )
    result = chain.invoke({"query": question})
    sources = []
    seen = set()
    for doc in result.get("source_documents", []):
        case_name = doc.metadata.get("case_name", "")
        case_no   = doc.metadata.get("case_no", "")
        date      = doc.metadata.get("date", "")
        case_type = doc.metadata.get("case_type", "")
        key = f"{case_no}"
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "case_name": case_name,
            "case_no":   case_no,
            "date":      date,
            "case_type": case_type,
            "content":   doc.page_content[:300]
        })
    return {"answer": result["result"], "sources": sources, "saved": user_id is not None}

async def search_integrated(
    question: str,
    user_id: Optional[int] = None,
    category: Optional[str] = None
) -> dict:
    collections = [
        ("law_articles",    "법령"),
        ("precedents",      "판례"),
        ("interpretations", "법령해석례"),
        ("constitutional",  "헌재결정례"),
    ]

    all_docs = []
    for col_name, col_type in collections:
        try:
            vectordb = Chroma(
                persist_directory  = CHROMA_DIR,
                embedding_function = _embed_model,
                collection_name    = col_name
            )
            search_kwargs = {"k": 2}
            if category and col_name != "law_articles":
                search_kwargs["filter"] = {"category": category}
            docs = vectordb.similarity_search(question, **search_kwargs)
            for doc in docs:
                doc.metadata["collection_type"] = col_type
                all_docs.append(doc)
        except Exception as e:
            print(f"  {col_name} 검색 오류: {e}")
            continue

    context = "\n\n".join([
        f"[{doc.metadata.get('collection_type', '')}]\n{doc.page_content[:500]}"
        for doc in all_docs
    ])

    llm = get_llm()
    prompt_text = INTEGRATED_PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt_text)
    answer = response.content if hasattr(response, "content") else str(response)

    sources = []
    seen = set()
    for doc in all_docs:
        col_type = doc.metadata.get("collection_type", "")
        if col_type == "법령":
            key = f"{doc.metadata.get('law_name','')}_{doc.metadata.get('article_no','')}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "type":          "법령",
                    "law_name":      doc.metadata.get("law_name", ""),
                    "article_no":    doc.metadata.get("article_no", ""),
                    "article_title": doc.metadata.get("article_title", ""),
                    "content":       doc.page_content[:200]
                })
        else:
            key = doc.metadata.get("case_no", "")
            if key and key not in seen:
                seen.add(key)
                sources.append({
                    "type":      col_type,
                    "case_name": doc.metadata.get("case_name", ""),
                    "case_no":   key,
                    "content":   doc.page_content[:200]
                })

    return {"answer": answer, "sources": sources, "saved": user_id is not None}
