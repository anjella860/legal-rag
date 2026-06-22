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

KEYWORD_TEXT = """
근로|노동/고용
근로자|노동/고용
근로계약|노동/고용
근로 계약|노동/고용
근로시간|노동/고용
근로 시간|노동/고용
해고|노동/고용
부당해고|노동/고용
부당 해고|노동/고용
징계해고|노동/고용
징계 해고|노동/고용
정리해고|노동/고용
정리 해고|노동/고용
임금|노동/고용
임금체불|노동/고용
임금 체불|노동/고용
최저임금|노동/고용
최저 임금|노동/고용
통상임금|노동/고용
통상 임금|노동/고용
퇴직금|노동/고용
연차휴가|노동/고용
연차 휴가|노동/고용
유급휴가|노동/고용
유급 휴가|노동/고용
육아휴직|노동/고용
육아 휴직|노동/고용
출산휴가|노동/고용
출산 휴가|노동/고용
직장내괴롭힘|노동/고용
직장 내 괴롭힘|노동/고용
고용차별|노동/고용
고용 차별|노동/고용
임대차|주거/임대
임대차계약|주거/임대
임대차 계약|주거/임대
주택임대차|주거/임대
주택 임대차|주거/임대
상가임대차|주거/임대
상가 임대차|주거/임대
주거|주거/임대
보증금|주거/임대
보증금반환|주거/임대
보증금 반환|주거/임대
전세보증금|주거/임대
전세 보증금|주거/임대
전월세|주거/임대
전세|주거/임대
월세|주거/임대
명도소송|주거/임대
명도 소송|주거/임대
임차권등기|주거/임대
임차권 등기|주거/임대
계약갱신|주거/임대
계약 갱신|주거/임대
소비자|소비자/생활
소비자분쟁|소비자/생활
소비자 분쟁|소비자/생활
통신판매|소비자/생활
통신 판매|소비자/생활
전자상거래|소비자/생활
전자 상거래|소비자/생활
불공정계약|소비자/생활
불공정 계약|소비자/생활
제조물책임|소비자/생활
제조물 책임|소비자/생활
방문판매|소비자/생활
방문 판매|소비자/생활
청약철회|소비자/생활
청약 철회|소비자/생활
환불|소비자/생활
손해배상|소비자/생활
손해 배상|소비자/생활
개인정보|개인정보/디지털
개인정보유출|개인정보/디지털
개인정보 유출|개인정보/디지털
개인정보침해|개인정보/디지털
개인정보 침해|개인정보/디지털
개인정보처리|개인정보/디지털
개인정보 처리|개인정보/디지털
개인정보제공|개인정보/디지털
개인정보 제공|개인정보/디지털
개인정보 제3자 제공|개인정보/디지털
정보통신망|개인정보/디지털
정보 통신망|개인정보/디지털
정보통신서비스|개인정보/디지털
정보통신 서비스|개인정보/디지털
인터넷|개인정보/디지털
사이버범죄|개인정보/디지털
사이버 범죄|개인정보/디지털
위치정보|개인정보/디지털
위치 정보|개인정보/디지털
명예훼손|개인정보/디지털
명예 훼손|개인정보/디지털
청년|청년/취업
고용|청년/취업
취업|청년/취업
고용차별|청년/취업
고용 차별|청년/취업
취업차별|청년/취업
취업 차별|청년/취업
취업규칙|청년/취업
취업 규칙|청년/취업
직업안정|청년/취업
직업 안정|청년/취업
채용절차|청년/취업
채용 절차|청년/취업
채용비리|청년/취업
채용 비리|청년/취업
비정규직|청년/취업
비정규 근로자|청년/취업
"""

SEARCH_KEYWORDS = [
    {"keyword": keyword.strip(), "category": category.strip()}
    for keyword, category in (
        line.split("|", 1)
        for line in KEYWORD_TEXT.strip().splitlines()
        if "|" in line
    )
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


def fetch_detc_list(keyword: str, display: int = 100) -> list[dict]:
    url = f"{BASE_URL}/lawSearch.do"
    params = {
        "OC": LAW_API_KEY,
        "target": "detc",
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

    for detc in root.findall(".//Detc"):
        detc_id = detc.findtext("헌재결정례일련번호", "").strip()
        case_name = detc.findtext("사건명", "").strip()
        case_no = detc.findtext("사건번호", "").strip()
        date = detc.findtext("종국일자", "").strip()

        if detc_id:
            results.append(
                {
                    "id": detc_id,
                    "case_name": case_name,
                    "case_no": case_no,
                    "date": date,
                }
            )

    return results


def fetch_detc_content(detc_id: str) -> dict | None:
    url = f"{BASE_URL}/lawService.do"
    params = {
        "OC": LAW_API_KEY,
        "target": "detc",
        "ID": detc_id,
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
        "사건종류명": root.findtext("사건종류명", "").strip(),
        "판시사항": clean_html(root.findtext("판시사항", "")),
        "결정요지": clean_html(root.findtext("결정요지", "")),
        "전문": clean_html(root.findtext("전문", ""))[:3000],
    }


def load_to_chroma(chunks: list[dict], embed_model):
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embed_model,
        collection_name="constitutional",
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
    print(f"  {len(new_chunks)}개 헌재결정례 적재 완료")
    return len(new_chunks)


def main():
    print("=" * 50)
    print("헌재결정례 데이터 적재 시작")
    print("=" * 50)

    print("\n임베딩 모델 로딩 중...")
    embed_model = get_embed_model()
    print("임베딩 모델 로딩 완료!")

    total = 0
    seen_detc_ids = set()

    for item in SEARCH_KEYWORDS:
        keyword = item["keyword"]
        category = item["category"]
        print(f"\n[{category}] [{keyword}] 헌재결정례 검색 중...")

        detc_list = fetch_detc_list(keyword, display=100)
        print(f"  검색된 헌재결정례 수: {len(detc_list)}개")

        if not detc_list:
            print("  헌재결정례 없음. 건너뜁니다.")
            continue

        chunks = []

        for detc in detc_list:
            if detc["id"] in seen_detc_ids:
                continue

            seen_detc_ids.add(detc["id"])

            content = fetch_detc_content(detc["id"])
            time.sleep(0.1)

            if not content:
                continue

            full_text = f"""[헌재결정례] {detc['case_name']}

사건번호: {detc['case_no']} | 종국일자: {detc['date']} | 사건종류: {content['사건종류명']}

[판시사항]
{content['판시사항']}

[결정요지]
{content['결정요지']}

[결정 전문]
{content['전문']}"""

            chunks.append(
                {
                    "id": f"detc_{detc['id']}",
                    "content": full_text,
                    "metadata": {
                        "type": "constitutional",
                        "category": category,
                        "keyword": keyword,
                        "case_name": detc["case_name"][:100],
                        "case_no": detc["case_no"],
                        "date": detc["date"],
                        "case_type": content["사건종류명"],
                    },
                }
            )

        count = load_to_chroma(chunks, embed_model)
        total += count

    print("\n" + "=" * 50)
    print(f"적재 완료! 총 {total}개 헌재결정례가 ChromaDB에 저장되었습니다.")
    print("=" * 50)


if __name__ == "__main__":
    main()
