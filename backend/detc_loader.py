import os
import re
import requests
import xml.etree.ElementTree as ET
import datetime
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

LAW_API_KEY = os.getenv("LAW_API_KEY", "legalrag2026")
CHROMA_DIR  = os.getenv("CHROMA_DIR", "./chroma_db")
BASE_URL    = "http://www.law.go.kr/DRF"

SEARCH_KEYWORDS = [
    {"keyword": "근로",          "category": "노동/고용"},
    {"keyword": "해고",          "category": "노동/고용"},
    {"keyword": "임금",          "category": "노동/고용"},
    {"keyword": "육아휴직",      "category": "노동/고용"},
    {"keyword": "최저임금",      "category": "노동/고용"},
    {"keyword": "임대차",        "category": "주거/임대"},
    {"keyword": "주거",          "category": "주거/임대"},
    {"keyword": "소비자",        "category": "소비자/생활"},
    {"keyword": "개인정보",      "category": "개인정보/디지털"},
    {"keyword": "인터넷",        "category": "개인정보/디지털"},
    {"keyword": "청년",          "category": "청년/취업"},
    {"keyword": "고용",          "category": "청년/취업"},
]

def get_embed_model():
    return HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def clean_html(text: str) -> str:
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()

def fetch_detc_list(keyword: str, display: int = 20) -> list[dict]:
    url = f"{BASE_URL}/lawSearch.do"
    params = {
        "OC":      LAW_API_KEY,
        "target":  "detc",
        "type":    "XML",
        "query":   keyword,
        "display": display,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.encoding = "utf-8"

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        print(f"  XML 파싱 오류: {keyword}")
        return []

    results = []
    for detc in root.findall(".//Detc"):
        detc_id   = detc.findtext("헌재결정례일련번호", "").strip()
        case_name = detc.findtext("사건명", "").strip()
        case_no   = detc.findtext("사건번호", "").strip()
        date      = detc.findtext("종국일자", "").strip()

        if detc_id:
            results.append({
                "id":        detc_id,
                "case_name": case_name,
                "case_no":   case_no,
                "date":      date,
            })
    return results

def fetch_detc_content(detc_id: str) -> dict | None:
    url = f"{BASE_URL}/lawService.do"
    params = {
        "OC":     LAW_API_KEY,
        "target": "detc",
        "ID":     detc_id,
        "type":   "XML",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.encoding = "utf-8"

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return None

    return {
        "사건종류명": root.findtext("사건종류명", "").strip(),
        "판시사항":   clean_html(root.findtext("판시사항", "")),
        "결정요지":   clean_html(root.findtext("결정요지", "")),
        "전문":       clean_html(root.findtext("전문", ""))[:2000],
    }

def load_to_chroma(chunks: list[dict], embed_model):
    vectordb = Chroma(
        persist_directory  = CHROMA_DIR,
        embedding_function = embed_model,
        collection_name    = "constitutional"
    )

    existing = vectordb.get()
    existing_ids = set(existing["ids"]) if existing["ids"] else set()
    new_chunks = [c for c in chunks if c["id"] not in existing_ids]

    if not new_chunks:
        print("  이미 적재된 데이터입니다. 건너뜁니다.")
        return 0

    texts     = [c["content"]  for c in new_chunks]
    metadatas = [c["metadata"] for c in new_chunks]
    ids       = [c["id"]       for c in new_chunks]

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
    for item in SEARCH_KEYWORDS:
        keyword  = item["keyword"]
        category = item["category"]
        print(f"\n[{category}] [{keyword}] 헌재결정례 검색 중...")

        detc_list = fetch_detc_list(keyword, display=20)
        print(f"  검색된 헌재결정례 수: {len(detc_list)}개")

        if not detc_list:
            print("  헌재결정례 없음. 건너뜁니다.")
            continue

        chunks = []
        for detc in detc_list:
            content = fetch_detc_content(detc["id"])
            if not content:
                continue

            full_text = f"""[헌재결정례] {detc['case_name']}
사건번호: {detc['case_no']} | 종국일자: {detc['date']} | 사건종류: {content['사건종류명']}

[판시사항]
{content['판시사항']}

[결정요지]
{content['결정요지']}

[전문 요약]
{content['전문']}"""

            chunks.append({
                "id":      f"detc_{detc['id']}",
                "content": full_text,
                "metadata": {
                    "type":      "constitutional",
                    "category":  category,
                    "keyword":   keyword,
                    "case_name": detc["case_name"][:100],
                    "case_no":   detc["case_no"],
                    "date":      detc["date"],
                    "case_type": content["사건종류명"],
                }
            })

        count = load_to_chroma(chunks, embed_model)
        total += count

    print("\n" + "=" * 50)
    print(f"적재 완료! 총 {total}개 헌재결정례가 ChromaDB에 저장되었습니다.")
    print("=" * 50)

if __name__ == "__main__":
    main()
