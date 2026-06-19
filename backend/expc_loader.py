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
    {"keyword": "연차휴가",      "category": "노동/고용"},
    {"keyword": "부당해고",      "category": "노동/고용"},
    {"keyword": "임금체불",      "category": "노동/고용"},
    {"keyword": "퇴직금",        "category": "노동/고용"},
    {"keyword": "최저임금",      "category": "노동/고용"},
    {"keyword": "육아휴직",      "category": "노동/고용"},
    {"keyword": "직장내괴롭힘",  "category": "노동/고용"},
    {"keyword": "근로계약",      "category": "노동/고용"},
    {"keyword": "임대차",        "category": "주거/임대"},
    {"keyword": "보증금",        "category": "주거/임대"},
    {"keyword": "소비자",        "category": "소비자/생활"},
    {"keyword": "개인정보",      "category": "개인정보/디지털"},
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

def fetch_expc_list(keyword: str, display: int = 20) -> list[dict]:
    url = f"{BASE_URL}/lawSearch.do"
    params = {
        "OC":      LAW_API_KEY,
        "target":  "expc",
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
    for expc in root.findall(".//expc"):
        expc_id      = expc.findtext("법령해석례일련번호", "").strip()
        case_name    = expc.findtext("안건명", "").strip()
        case_no      = expc.findtext("안건번호", "").strip()
        query_org    = expc.findtext("질의기관명", "").strip()
        reply_org    = expc.findtext("회신기관명", "").strip()
        reply_date   = expc.findtext("회신일자", "").strip()

        if expc_id:
            results.append({
                "id":         expc_id,
                "case_name":  case_name,
                "case_no":    case_no,
                "query_org":  query_org,
                "reply_org":  reply_org,
                "reply_date": reply_date,
            })
    return results

def fetch_expc_content(expc_id: str) -> dict | None:
    url = f"{BASE_URL}/lawService.do"
    params = {
        "OC":     LAW_API_KEY,
        "target": "expc",
        "ID":     expc_id,
        "type":   "XML",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.encoding = "utf-8"

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return None

    return {
        "질의요지": clean_html(root.findtext("질의요지", "")),
        "회답":     clean_html(root.findtext("회답", "")),
        "이유":     clean_html(root.findtext("이유", ""))[:2000],
    }

def load_to_chroma(chunks: list[dict], embed_model):
    vectordb = Chroma(
        persist_directory  = CHROMA_DIR,
        embedding_function = embed_model,
        collection_name    = "interpretations"
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
    for item in SEARCH_KEYWORDS:
        keyword  = item["keyword"]
        category = item["category"]
        print(f"\n[{category}] [{keyword}] 법령해석례 검색 중...")

        expc_list = fetch_expc_list(keyword, display=20)
        print(f"  검색된 법령해석례 수: {len(expc_list)}개")

        if not expc_list:
            print("  법령해석례 없음. 건너뜁니다.")
            continue

        chunks = []
        for expc in expc_list:
            content = fetch_expc_content(expc["id"])
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

            chunks.append({
                "id":      f"expc_{expc['id']}",
                "content": full_text,
                "metadata": {
                    "type":       "interpretation",
                    "category":   category,
                    "keyword":    keyword,
                    "case_name":  expc["case_name"][:100],
                    "case_no":    expc["case_no"],
                    "query_org":  expc["query_org"],
                    "reply_org":  expc["reply_org"],
                    "reply_date": expc["reply_date"],
                }
            })

        count = load_to_chroma(chunks, embed_model)
        total += count

    print("\n" + "=" * 50)
    print(f"적재 완료! 총 {total}개 법령해석례가 ChromaDB에 저장되었습니다.")
    print("=" * 50)

if __name__ == "__main__":
    main()
