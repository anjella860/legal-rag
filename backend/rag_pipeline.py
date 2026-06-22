import os
import re
from langchain.chains.question_answering import load_qa_chain
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
    encode_kwargs={"normalize_embeddings": True},
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
7. 임차권등기명령은 소유권 보호가 아니라 대항력과 우선변제권 유지를 위한 제도임을 구분해서 설명하세요.
8. 여러 법령이 함께 검색된 경우, 먼저 질문과 가장 직접 관련된 기본 법령을 중심으로 답변하고, 다른 법령은 추가 참고로만 구분해서 설명하세요.
9. 연차휴가 수당 여부는 사용촉진 조치 여부, 미사용 사유, 근로관계 종료 여부에 따라 달라질 수 있으므로 단정하지 말고 조건별로 설명하세요.
10. 참고 조문 중 질문과 직접 관련 없는 조문은 답변에 사용하지 마세요. 직접 관련성이 낮은 조문은 참고 조문 목록에도 포함하지 마세요.
11. 같은 문장이나 같은 의미의 설명을 반복하지 마세요.
12. 답변은 핵심 답변, 조문 해석, 적용 조건, 주의사항, 참고 조문을 포함하되 각 항목은 3문장 이내로 작성하세요.
13. 여러 참고 조문이 제공된 경우, 첫 번째 참고 조문을 가장 핵심 근거로 보고 답변하세요. 뒤의 조문은 보조 근거로만 사용하세요.
14. 참고 조문 목록에 여러 조문이 있더라도, 질문에 직접 답하는 조문이 아니면 핵심 답변의 중심으로 삼지 마세요.
15. 같은 의미의 문장을 반복하지 말고, 핵심 답변에서는 사용자의 질문에 대한 실제 대응 방법을 함께 제시하세요.
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

[답변]""",
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

[답변]""",
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

[답변]""",
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

[답변]""",
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

[답변]""",
)


def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant", temperature=0, api_key=os.getenv("GROQ_API_KEY")
    )


def expand_query(question: str) -> str:
    keyword_map = {
        "임금체불": "임금체불 임금 체불 임금 지급 금품 청산 체불임금 근로기준법 제36조 제43조 제109조",
        "임금 체불": "임금체불 임금 체불 임금 지급 금품 청산 체불임금 근로기준법 제36조 제43조 제109조",
        "퇴직금": "퇴직금 퇴직 급여 금품 청산 근로자퇴직급여 보장법",
        "연차": "연차휴가 유급휴가 연차 유급휴가 근로기준법 제60조",
        "부당해고": "부당해고 부당 해고 해고 제한 노동위원회 구제신청",
        "부당 해고": "부당해고 부당 해고 해고 제한 노동위원회 구제신청",
        "보증금": "보증금 반환 임대차 임차권등기명령 주택임대차보호법 상가건물 임대차보호법",
        "개인정보 유출": "개인정보 유출 개인정보 침해 손해배상 개인정보 보호법",
        "개인정보침해": "개인정보 침해 개인정보 유출 손해배상 개인정보 보호법",
        "근로계약서": "근로계약서 근로 계약서 근로조건 명시 서면 명시 근로기준법 제17조 임금 소정근로시간 휴일 연차 유급휴가",
        "근로계약": "근로계약 근로 계약 근로조건 명시 서면 명시 근로기준법 제17조",
        "계약서": "근로계약서 근로조건 명시 서면 명시 근로기준법 제17조",
    }

    expanded_question = question

    for keyword, extra_terms in keyword_map.items():
        if keyword in question:
            expanded_question += " " + extra_terms

    return expanded_question


def select_law_names(question: str) -> Optional[List[str]]:
    law_rules = [
        {
            "keywords": [
                "임금",
                "임금체불",
                "임금 체불",
                "월급",
                "급여",
                "연차",
                "연차휴가",
                "연차 수당",
                "부당해고",
                "부당 해고",
                "해고",
                "근로계약서",
                "근로 계약서",
                "근로계약",
                "근로시간",
                "야근",
                "연장근로",
                "휴일근로",
                "직장 내 괴롭힘",
                "직장내괴롭힘",
                "괴롭힘",
            ],
            "laws": ["근로기준법"],
        },
        {
            "keywords": [
                "퇴직금",
                "퇴직급여",
                "퇴직 연금",
                "퇴직연금",
            ],
            "laws": ["근로자퇴직급여 보장법", "근로기준법"],
        },
        {
            "keywords": [
                "최저임금",
                "최저 임금",
                "수습기간",
                "수습 기간",
            ],
            "laws": ["최저임금법", "근로기준법"],
        },
        {
            "keywords": [
                "육아휴직",
                "육아 휴직",
                "육아기",
                "출산휴가",
                "출산 휴가",
                "배우자 출산",
                "일가정",
                "일ㆍ가정",
            ],
            "laws": ["남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률"],
        },
        {
            "keywords": [
                "주택",
                "전세",
                "월세",
                "집주인",
                "임대인",
                "임차인",
                "보증금",
                "계약갱신",
                "계약 갱신",
                "임차권등기",
                "임차권 등기",
            ],
            "laws": ["주택임대차보호법"],
        },
        {
            "keywords": [
                "상가",
                "권리금",
                "상가 임대차",
                "상가임대차",
            ],
            "laws": ["상가건물 임대차보호법"],
        },
        {
            "keywords": [
                "온라인 쇼핑몰",
                "인터넷 쇼핑",
                "전자상거래",
                "통신판매",
                "배송",
                "배송 지연",
                "환불",
                "청약철회",
                "청약 철회",
            ],
            "laws": ["전자상거래 등에서의 소비자보호에 관한 법률"],
        },
        {
            "keywords": [
                "방문판매",
                "전화권유판매",
            ],
            "laws": ["방문판매 등에 관한 법률"],
        },
        {
            "keywords": [
                "할부",
                "할부거래",
            ],
            "laws": ["할부거래에 관한 법률"],
        },
        {
            "keywords": [
                "개인정보",
                "개인 정보",
                "개인정보 유출",
                "개인정보 침해",
                "동의 철회",
                "처리정지",
                "제3자 제공",
                "개인정보 제공",
            ],
            "laws": ["개인정보 보호법"],
        },
        {
            "keywords": [
                "신용정보",
                "개인신용정보",
            ],
            "laws": ["신용정보의 이용 및 보호에 관한 법률"],
        },
        {
            "keywords": [
                "위치정보",
                "개인위치정보",
            ],
            "laws": ["위치정보의 보호 및 이용 등에 관한 법률"],
        },
        {
            "keywords": [
                "채용",
                "채용서류",
                "채용 서류",
                "구직자",
                "채용절차",
            ],
            "laws": ["채용절차의 공정화에 관한 법률"],
        },
        {
            "keywords": [
                "산업재해",
                "산재",
                "안전조치",
                "보건조치",
                "작업중지",
                "안전보건",
                "근로자가 다치",
            ],
            "laws": ["산업안전보건법"],
        },
    ]

    selected = []

    for rule in law_rules:
        if any(keyword in question for keyword in rule["keywords"]):
            selected.extend(rule["laws"])

    selected = list(dict.fromkeys(selected))

    return selected if selected else None


def extract_query_terms(question: str) -> List[str]:
    terms = re.findall(r"[가-힣A-Za-z0-9]+", question)

    stopwords = {
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "에",
        "에서",
        "으로",
        "로",
        "하고",
        "하면",
        "되나요",
        "되면",
        "수",
        "있나요",
        "있는지",
        "어떻게",
        "무엇",
        "언제",
        "왜",
        "회사",
        "근로자",
        "소비자",
        "임차인",
        "사업자",
        "사업주",
    }

    filtered_terms = []

    for term in terms:
        if len(term) <= 1:
            continue

        if term in stopwords:
            continue

        filtered_terms.append(term)

    return list(dict.fromkeys(filtered_terms))


def rerank_docs(question: str, docs: list):
    priority_map = {
        "임금체불": [
            "제36조",
            "제37조",
            "제43조",
            "제43조의2",
            "제43조의8",
            "제109조",
            "금품 청산",
            "미지급 임금",
            "임금 지급",
            "체불 임금",
            "벌칙",
        ],
        "임금 체불": [
            "제36조",
            "제37조",
            "제43조",
            "제43조의2",
            "제43조의8",
            "제109조",
            "금품 청산",
            "미지급 임금",
            "임금 지급",
            "체불 임금",
            "벌칙",
        ],
        "퇴직금": [
            "제9조",
            "퇴직금의 지급",
            "14일 이내",
            "퇴직금",
            "퇴직급여",
            "제8조",
            "제44조",
            "벌칙",
            "금품 청산",
        ],
        "연차": [
            "연차",
            "유급휴가",
            "제60조",
            "근로기준법",
        ],
        "부당해고": [
            "해고",
            "부당해고",
            "해고 제한",
            "노동위원회",
            "구제신청",
            "제23조",
            "제28조",
        ],
        "보증금": [
            "보증금 반환",
            "임차권등기",
            "임대차",
            "주택임대차보호법",
            "상가건물 임대차보호법",
            "대항력",
            "우선변제권",
        ],
        "전세": [
            "전세보증금",
            "보증금 반환",
            "임차권등기",
            "주택임대차보호법",
            "대항력",
            "우선변제권",
        ],
        "개인정보": [
            "개인정보",
            "개인정보 보호법",
            "개인정보 유출",
            "개인정보 침해",
            "안전조치",
            "손해배상",
            "제3자 제공",
        ],
        "환불": [
            "환불",
            "청약철회",
            "전자상거래",
            "소비자",
            "손해배상",
            "소비자보호",
        ],
        "근로계약서": [
            "제17조",
            "근로조건의 명시",
            "근로계약서",
            "서면",
            "임금",
            "소정근로시간",
            "휴일",
            "연차 유급휴가",
        ],
        "근로계약": [
            "제17조",
            "근로조건의 명시",
            "근로계약",
            "서면",
            "임금",
            "소정근로시간",
        ],
    }

    priority_terms = []

    for keyword, terms in priority_map.items():
        if keyword in question:
            priority_terms.extend(terms)

    query_terms = extract_query_terms(question)

    scored_docs = []

    for doc in docs:
        text = doc.page_content
        metadata = doc.metadata
        score = 0

        law_name = metadata.get("law_name", "")
        article_title = metadata.get("article_title", "")
        article_no = metadata.get("article_no", "")

        # 1. 기존 주제별 보정 점수
        for term in priority_terms:
            if term in text:
                score += 10

        # 2. 일반형 rerank: 질문 단어가 조문 제목/본문에 포함되면 자동 가점
        for term in query_terms:
            if term in article_title:
                score += 8

            if term in law_name:
                score += 5

            if term in text:
                score += 3

        # 3. 조문 제목이 질문에 직접 포함되면 강한 가점
        if article_title and article_title in question:
            score += 15

        # 4. 조문번호가 질문에 직접 포함되면 가점
        if article_no and article_no in question:
            score += 10

        if "임금체불" in question or "임금 체불" in question:
            weak_terms = [
                "제19조",
                "근로조건의 위반",
                "제45조",
                "비상시 지급",
            ]

            for term in weak_terms:
                if term in text:
                    score -= 8

        if (
            "근로계약서" in question
            or "근로 계약서" in question
            or "근로계약" in question
        ):
            weak_terms = [
                "제26조",
                "해고의 예고",
                "파견근로자",
                "근로자파견",
                "계약의 해지",
                "근로시간 면제",
                "노동조합",
                "제93조",
                "취업규칙",
            ]

            for term in weak_terms:
                if term in text:
                    score -= 10

        scored_docs.append((score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)

    return [doc for score, doc in scored_docs]


async def ask_question(
    question: str, user_id: Optional[int] = None, law_names: Optional[List[str]] = None
) -> dict:
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=_embed_model,
        collection_name="law_articles",
    )

    search_question = expand_query(question)

    selected_laws = law_names if law_names else select_law_names(question)

    filter_arg = None
    if selected_laws:
        filter_arg = {"law_name": {"$in": selected_laws}}

    docs = vectordb.max_marginal_relevance_search(
        search_question,
        k=12,
        fetch_k=30,
        lambda_mult=0.7,
        filter=filter_arg,
    )

    # 자동 법령 필터를 적용했는데 결과가 너무 적으면 전체 법령에서 다시 검색
    if selected_laws and len(docs) < 3:
        docs = vectordb.max_marginal_relevance_search(
            search_question,
            k=12,
            fetch_k=30,
            lambda_mult=0.7,
        )

    reranked_docs = rerank_docs(question, docs)
    selected_docs = reranked_docs[:3]

    if not selected_docs:
        return {
            "answer": "질문과 직접 관련된 법령 조문을 찾지 못했습니다. 질문을 조금 더 구체적으로 입력해 주세요.",
            "sources": [],
            "saved": False,
        }

    chain = load_qa_chain(
        llm=get_llm(),
        chain_type="stuff",
        prompt=LAW_PROMPT,
    )

    result = chain.invoke(
        {
            "input_documents": selected_docs,
            "question": question,
        }
    )

    sources = []
    seen = set()

    for doc in selected_docs:
        law_name = doc.metadata.get("law_name", "")
        article_no = doc.metadata.get("article_no", "")
        article_title = doc.metadata.get("article_title", "")
        key = f"{law_name}_{article_no}"

        if key in seen:
            continue

        seen.add(key)
        sources.append(
            {
                "law_name": law_name,
                "article_no": article_no,
                "article_title": article_title,
                "content": doc.page_content[:300],
            }
        )

    return {
        "answer": result["output_text"],
        "sources": sources,
        "saved": user_id is not None,
    }


async def search_precedent(
    question: str, user_id: Optional[int] = None, category: Optional[str] = None
) -> dict:
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=_embed_model,
        collection_name="precedents",
    )
    search_kwargs = {"k": 8}
    if category:
        search_kwargs["filter"] = {"category": category}
    retriever = vectordb.as_retriever(search_kwargs=search_kwargs)
    chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        retriever=retriever,
        chain_type_kwargs={"prompt": PREC_PROMPT},
        return_source_documents=True,
    )
    result = chain.invoke({"query": expand_query(question)})
    sources = []
    seen = set()
    for doc in result.get("source_documents", []):
        case_name = doc.metadata.get("case_name", "")
        case_no = doc.metadata.get("case_no", "")
        court = doc.metadata.get("court", "")
        date = doc.metadata.get("date", "")
        case_type = doc.metadata.get("case_type", "")
        key = f"{case_no}"
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "case_name": case_name,
                "case_no": case_no,
                "court": court,
                "date": date,
                "case_type": case_type,
                "content": doc.page_content[:300],
            }
        )
    return {
        "answer": result["result"],
        "sources": sources,
        "saved": user_id is not None,
    }


async def search_interpretation(
    question: str, user_id: Optional[int] = None, category: Optional[str] = None
) -> dict:
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=_embed_model,
        collection_name="interpretations",
    )
    search_kwargs = {"k": 8}
    if category:
        search_kwargs["filter"] = {"category": category}
    retriever = vectordb.as_retriever(search_kwargs=search_kwargs)
    chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        retriever=retriever,
        chain_type_kwargs={"prompt": EXPC_PROMPT},
        return_source_documents=True,
    )
    result = chain.invoke({"query": question})
    sources = []
    seen = set()
    for doc in result.get("source_documents", []):
        case_name = doc.metadata.get("case_name", "")
        case_no = doc.metadata.get("case_no", "")
        query_org = doc.metadata.get("query_org", "")
        reply_org = doc.metadata.get("reply_org", "")
        reply_date = doc.metadata.get("reply_date", "")
        key = f"{case_no}"
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "case_name": case_name,
                "case_no": case_no,
                "query_org": query_org,
                "reply_org": reply_org,
                "reply_date": reply_date,
                "content": doc.page_content[:300],
            }
        )
    return {
        "answer": result["result"],
        "sources": sources,
        "saved": user_id is not None,
    }


async def search_constitutional(
    question: str, user_id: Optional[int] = None, category: Optional[str] = None
) -> dict:
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=_embed_model,
        collection_name="constitutional",
    )
    search_kwargs = {"k": 3}
    if category:
        search_kwargs["filter"] = {"category": category}
    retriever = vectordb.as_retriever(search_kwargs=search_kwargs)
    chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        retriever=retriever,
        chain_type_kwargs={"prompt": DETC_PROMPT},
        return_source_documents=True,
    )
    result = chain.invoke({"query": question})
    sources = []
    seen = set()
    for doc in result.get("source_documents", []):
        case_name = doc.metadata.get("case_name", "")
        case_no = doc.metadata.get("case_no", "")
        date = doc.metadata.get("date", "")
        case_type = doc.metadata.get("case_type", "")
        key = f"{case_no}"
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "case_name": case_name,
                "case_no": case_no,
                "date": date,
                "case_type": case_type,
                "content": doc.page_content[:300],
            }
        )
    return {
        "answer": result["result"],
        "sources": sources,
        "saved": user_id is not None,
    }


async def search_integrated(
    question: str, user_id: Optional[int] = None, category: Optional[str] = None
) -> dict:
    collections = [
        ("law_articles", "법령"),
        ("precedents", "판례"),
        ("interpretations", "법령해석례"),
        ("constitutional", "헌재결정례"),
    ]

    all_docs = []
    for col_name, col_type in collections:
        try:
            vectordb = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=_embed_model,
                collection_name=col_name,
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

    context = "\n\n".join(
        [
            f"[{doc.metadata.get('collection_type', '')}]\n{doc.page_content[:500]}"
            for doc in all_docs
        ]
    )

    llm = get_llm()
    prompt_text = INTEGRATED_PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt_text)
    answer = response.content if hasattr(response, "content") else str(response)

    sources = []
    seen = set()
    for doc in all_docs:
        col_type = doc.metadata.get("collection_type", "")
        if col_type == "법령":
            key = (
                f"{doc.metadata.get('law_name','')}_{doc.metadata.get('article_no','')}"
            )
            if key not in seen:
                seen.add(key)
                sources.append(
                    {
                        "type": "법령",
                        "law_name": doc.metadata.get("law_name", ""),
                        "article_no": doc.metadata.get("article_no", ""),
                        "article_title": doc.metadata.get("article_title", ""),
                        "content": doc.page_content[:200],
                    }
                )
        else:
            key = doc.metadata.get("case_no", "")
            if key and key not in seen:
                seen.add(key)
                sources.append(
                    {
                        "type": col_type,
                        "case_name": doc.metadata.get("case_name", ""),
                        "case_no": key,
                        "content": doc.page_content[:200],
                    }
                )

    return {"answer": answer, "sources": sources, "saved": user_id is not None}
