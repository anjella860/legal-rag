import os
import re
import time
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

LAW_API_KEY = os.getenv("LAW_API_KEY", "legalrag2026")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
BASE_URL = "http://www.law.go.kr/DRF"

SEARCH_KEYWORDS = [
    # 노동/고용
    {"keyword": "연차휴가", "category": "노동/고용"},
    {"keyword": "연차 휴가", "category": "노동/고용"},
    {"keyword": "유급휴가", "category": "노동/고용"},
    {"keyword": "유급 휴가", "category": "노동/고용"},
    {"keyword": "부당해고", "category": "노동/고용"},
    {"keyword": "부당 해고", "category": "노동/고용"},
    {"keyword": "징계해고", "category": "노동/고용"},
    {"keyword": "징계 해고", "category": "노동/고용"},
    {"keyword": "정리해고", "category": "노동/고용"},
    {"keyword": "정리 해고", "category": "노동/고용"},
    {"keyword": "해고예고", "category": "노동/고용"},
    {"keyword": "해고 예고", "category": "노동/고용"},
    {"keyword": "임금체불", "category": "노동/고용"},
    {"keyword": "임금 체불", "category": "노동/고용"},
    {"keyword": "퇴직금", "category": "노동/고용"},
    {"keyword": "퇴직 급여", "category": "노동/고용"},
    {"keyword": "최저임금", "category": "노동/고용"},
    {"keyword": "최저 임금", "category": "노동/고용"},
    {"keyword": "통상임금", "category": "노동/고용"},
    {"keyword": "통상 임금", "category": "노동/고용"},
    {"keyword": "평균임금", "category": "노동/고용"},
    {"keyword": "평균 임금", "category": "노동/고용"},
    {"keyword": "근로시간", "category": "노동/고용"},
    {"keyword": "근로 시간", "category": "노동/고용"},
    {"keyword": "연장근로", "category": "노동/고용"},
    {"keyword": "연장 근로", "category": "노동/고용"},
    {"keyword": "야간근로", "category": "노동/고용"},
    {"keyword": "야간 근로", "category": "노동/고용"},
    {"keyword": "휴일근로", "category": "노동/고용"},
    {"keyword": "휴일 근로", "category": "노동/고용"},
    {"keyword": "육아휴직", "category": "노동/고용"},
    {"keyword": "육아 휴직", "category": "노동/고용"},
    {"keyword": "출산휴가", "category": "노동/고용"},
    {"keyword": "출산 휴가", "category": "노동/고용"},
    {"keyword": "직장내괴롭힘", "category": "노동/고용"},
    {"keyword": "직장 내 괴롭힘", "category": "노동/고용"},
    {"keyword": "직장 괴롭힘", "category": "노동/고용"},
    {"keyword": "근로계약", "category": "노동/고용"},
    {"keyword": "근로 계약", "category": "노동/고용"},
    {"keyword": "근로계약서", "category": "노동/고용"},
    {"keyword": "근로 계약서", "category": "노동/고용"},
    {"keyword": "수습기간", "category": "노동/고용"},
    {"keyword": "수습 기간", "category": "노동/고용"},
    {"keyword": "근로자성", "category": "노동/고용"},
    {"keyword": "근로자 성", "category": "노동/고용"},
    # 주거/임대
    {"keyword": "임대차", "category": "주거/임대"},
    {"keyword": "임대차계약", "category": "주거/임대"},
    {"keyword": "임대차 계약", "category": "주거/임대"},
    {"keyword": "주택임대차", "category": "주거/임대"},
    {"keyword": "주택 임대차", "category": "주거/임대"},
    {"keyword": "상가임대차", "category": "주거/임대"},
    {"keyword": "상가 임대차", "category": "주거/임대"},
    {"keyword": "보증금", "category": "주거/임대"},
    {"keyword": "보증금반환", "category": "주거/임대"},
    {"keyword": "보증금 반환", "category": "주거/임대"},
    {"keyword": "전세보증금", "category": "주거/임대"},
    {"keyword": "전세 보증금", "category": "주거/임대"},
    {"keyword": "전월세", "category": "주거/임대"},
    {"keyword": "전세", "category": "주거/임대"},
    {"keyword": "월세", "category": "주거/임대"},
    {"keyword": "명도소송", "category": "주거/임대"},
    {"keyword": "명도 소송", "category": "주거/임대"},
    {"keyword": "임대차보호", "category": "주거/임대"},
    {"keyword": "임대차 보호", "category": "주거/임대"},
    {"keyword": "임차권등기", "category": "주거/임대"},
    {"keyword": "임차권 등기", "category": "주거/임대"},
    {"keyword": "계약갱신", "category": "주거/임대"},
    {"keyword": "계약 갱신", "category": "주거/임대"},
    {"keyword": "차임연체", "category": "주거/임대"},
    {"keyword": "차임 연체", "category": "주거/임대"},
    # 소비자/생활
    {"keyword": "소비자", "category": "소비자/생활"},
    {"keyword": "소비자분쟁", "category": "소비자/생활"},
    {"keyword": "소비자 분쟁", "category": "소비자/생활"},
    {"keyword": "통신판매", "category": "소비자/생활"},
    {"keyword": "통신 판매", "category": "소비자/생활"},
    {"keyword": "전자상거래", "category": "소비자/생활"},
    {"keyword": "전자 상거래", "category": "소비자/생활"},
    {"keyword": "불공정계약", "category": "소비자/생활"},
    {"keyword": "불공정 계약", "category": "소비자/생활"},
    {"keyword": "제조물책임", "category": "소비자/생활"},
    {"keyword": "제조물 책임", "category": "소비자/생활"},
    {"keyword": "방문판매", "category": "소비자/생활"},
    {"keyword": "방문 판매", "category": "소비자/생활"},
    {"keyword": "청약철회", "category": "소비자/생활"},
    {"keyword": "청약 철회", "category": "소비자/생활"},
    {"keyword": "환불", "category": "소비자/생활"},
    {"keyword": "손해배상", "category": "소비자/생활"},
    {"keyword": "손해 배상", "category": "소비자/생활"},
    {"keyword": "약관", "category": "소비자/생활"},
    # 개인정보/디지털
    {"keyword": "개인정보", "category": "개인정보/디지털"},
    {"keyword": "개인정보유출", "category": "개인정보/디지털"},
    {"keyword": "개인정보 유출", "category": "개인정보/디지털"},
    {"keyword": "개인정보침해", "category": "개인정보/디지털"},
    {"keyword": "개인정보 침해", "category": "개인정보/디지털"},
    {"keyword": "개인정보처리", "category": "개인정보/디지털"},
    {"keyword": "개인정보 처리", "category": "개인정보/디지털"},
    {"keyword": "개인정보제공", "category": "개인정보/디지털"},
    {"keyword": "개인정보 제공", "category": "개인정보/디지털"},
    {"keyword": "개인정보 제3자 제공", "category": "개인정보/디지털"},
    {"keyword": "정보통신망", "category": "개인정보/디지털"},
    {"keyword": "정보 통신망", "category": "개인정보/디지털"},
    {"keyword": "정보통신서비스", "category": "개인정보/디지털"},
    {"keyword": "정보통신 서비스", "category": "개인정보/디지털"},
    {"keyword": "사이버범죄", "category": "개인정보/디지털"},
    {"keyword": "사이버 범죄", "category": "개인정보/디지털"},
    {"keyword": "위치정보", "category": "개인정보/디지털"},
    {"keyword": "위치 정보", "category": "개인정보/디지털"},
    {"keyword": "명예훼손", "category": "개인정보/디지털"},
    {"keyword": "명예 훼손", "category": "개인정보/디지털"},
    # 청년/취업
    {"keyword": "고용", "category": "청년/취업"},
    {"keyword": "고용차별", "category": "청년/취업"},
    {"keyword": "고용 차별", "category": "청년/취업"},
    {"keyword": "취업차별", "category": "청년/취업"},
    {"keyword": "취업 차별", "category": "청년/취업"},
    {"keyword": "취업규칙", "category": "청년/취업"},
    {"keyword": "취업 규칙", "category": "청년/취업"},
    {"keyword": "직업안정", "category": "청년/취업"},
    {"keyword": "직업 안정", "category": "청년/취업"},
    {"keyword": "채용절차", "category": "청년/취업"},
    {"keyword": "채용 절차", "category": "청년/취업"},
    {"keyword": "채용비리", "category": "청년/취업"},
    {"keyword": "채용 비리", "category": "청년/취업"},
    {"keyword": "비정규직", "category": "청년/취업"},
    {"keyword": "비정규 근로자", "category": "청년/취업"},
]


def get_embed_model():
    return HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def clean_html(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def fetch_expc_list(keyword: str, display: int = 100) -> list[dict]:
    url = f"{BASE_URL}/lawSearch.do"
    params = {
        "OC": LAW_API_KEY,
        "target": "expc",
        "type": "XML",
        "query": keyword,
        "display": display,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.encoding = "utf-8"
    except requests.RequestException as e:
        print(f"  요청 실패: {keyword} / {e}")
        return []

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        print(f"  XML 파싱 오류: {keyword}")
        print(resp.text[:300])
        return []

    results = []

    for expc in root.findall(".//expc"):
        expc_id = expc.findtext("법령해석례일련번호", "").strip()
        case_name = expc.findtext("안건명", "").strip()
        case_no = expc.findtext("안건번호", "").strip()
        query_org = expc.findtext("질의기관명", "").strip()
        reply_org = expc.findtext("회신기관명", "").strip()
        reply_date = expc.findtext("회신일자", "").strip()

        if expc_id:
            results.append(
                {
                    "id": expc_id,
                    "case_name": case_name,
                    "case_no": case_no,
                    "query_org": query_org,
                    "reply_org": reply_org,
                    "reply_date": reply_date,
                }
            )

    return results


def fetch_expc_content(expc_id: str) -> dict | None:
    url = f"{BASE_URL}/lawService.do"
    params = {
        "OC": LAW_API_KEY,
        "target": "expc",
        "ID": expc_id,
        "type": "XML",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.encoding = "utf-8"
    except requests.RequestException:
        return None

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return None

    return {
        "질의요지": clean_html(root.findtext("질의요지", "")),
        "회답": clean_html(root.findtext("회답", "")),
        "이유": clean_html(root.findtext("이유", ""))[:3000],
    }


def load_to_chroma(chunks: list[dict], embed_model):
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embed_model,
        collection_name="interpretations",
    )

    existing = vectordb.get()
    existing_ids = set(existing["ids"]) if existing["ids"] else set()
    new_chunks = [c for c in chunks if c["id"] not in existing_ids]

    if not new_chunks:
        print("  이미 적재된 데이터입니다. 건너뜁니다.")
        return 0

    texts = [c["content"] for c in new_chunks]
    metadatas = [c["metadata"] for c in new_chunks]
    ids = [c["id"] for c in new_chunks]

    vectordb.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    print(f"  {len(new_chunks)}개 법령해석례 적재 완료")
    return len(new_chunks)


def main():
    print("=" * 50)
    print("법령해석례 데이터 적재 시작")
    print("=" * 50)

    print("\n임베딩 모델 로딩 중...")
    embed_model = get_embed_model()
    print("임베딩 모델 로딩 완료!")

    total = 0
    seen_expc_ids = set()

    for item in SEARCH_KEYWORDS:
        keyword = item["keyword"]
        category = item["category"]
        print(f"\n[{category}] [{keyword}] 법령해석례 검색 중...")

        expc_list = fetch_expc_list(keyword, display=100)
        print(f"  검색된 법령해석례 수: {len(expc_list)}개")

        if not expc_list:
            print("  법령해석례 없음. 건너뜁니다.")
            continue

        chunks = []

        for expc in expc_list:
            if expc["id"] in seen_expc_ids:
                continue

            seen_expc_ids.add(expc["id"])

            content = fetch_expc_content(expc["id"])
            time.sleep(0.1)

            if not content:
                continue

            full_text = f"""[법령해석례] {expc['case_name']}

안건번호: {expc['case_no']}
질의기관: {expc['query_org']} | 회신기관: {expc['reply_org']} | 회신일자: {expc['reply_date']}

[질의요지]
{content['질의요지']}

[회답]
{content['회답']}

[이유]
{content['이유']}"""

            chunks.append(
                {
                    "id": f"expc_{expc['id']}",
                    "content": full_text,
                    "metadata": {
                        "type": "interpretation",
                        "category": category,
                        "keyword": keyword,
                        "case_name": expc["case_name"][:100],
                        "case_no": expc["case_no"],
                        "query_org": expc["query_org"],
                        "reply_org": expc["reply_org"],
                        "reply_date": expc["reply_date"],
                    },
                }
            )

        count = load_to_chroma(chunks, embed_model)
        total += count

    print("\n" + "=" * 50)
    print(f"적재 완료! 총 {total}개 법령해석례가 ChromaDB에 저장되었습니다.")
    print("=" * 50)


if __name__ == "__main__":
    main()
