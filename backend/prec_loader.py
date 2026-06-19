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
    {"keyword": "해고예고",      "category": "노동/고용"},
    {"keyword": "근로계약",      "category": "노동/고용"},
    {"keyword": "보증금반환",    "category": "주거/임대"},
    {"keyword": "임대차계약",    "category": "주거/임대"},
    {"keyword": "주택임대차",    "category": "주거/임대"},
    {"keyword": "전월세",        "category": "주거/임대"},
    {"keyword": "명도소송",      "category": "주거/임대"},
    {"keyword": "임대차보호",    "category": "주거/임대"},
    {"keyword": "소비자분쟁",    "category": "소비자/생활"},
    {"keyword": "통신판매",      "category": "소비자/생활"},
    {"keyword": "불공정계약",    "category": "소비자/생활"},
    {"keyword": "제조물책임",    "category": "소비자/생활"},
    {"keyword": "방문판매",      "category": "소비자/생활"},
    {"keyword": "개인정보유출",  "category": "개인정보/디지털"},
    {"keyword": "개인정보침해",  "category": "개인정보/디지털"},
    {"keyword": "정보통신망",    "category": "개인정보/디지털"},
    {"keyword": "사이버범죄",    "category": "개인정보/디지털"},
    {"keyword": "고용차별",      "category": "청년/취업"},
    {"keyword": "취업규칙",      "category": "청년/취업"},
    {"keyword": "직업안정",      "category": "청년/취업"},
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

def fetch_prec_list(keyword: str, display: int = 10) -> list[dict]:
    url = f"{BASE_URL}/lawSearch.do"
    params = {
        "OC":      LAW_API_KEY,
        "target":  "prec",
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
    for prec in root.findall(".//prec"):
        prec_id   = prec.findtext("판례일련번호", "").strip()
        case_name = prec.findtext("사건명", "").strip()
        case_no   = prec.findtext("사건번호", "").strip()
        court     = prec.findtext("법원명", "").strip()
        date      = prec.findtext("선고일자", "").strip()
        case_type = prec.findtext("사건종류명", "").strip()

        if prec_id:
            results.append({
                "id":        prec_id,
                "case_name": case_name,
                "case_no":   case_no,
                "court":     court,
                "date":      date,
                "case_type": case_type,
            })
    return results

def fetch_prec_content(prec_id: str) -> dict | None:
    url = f"{BASE_URL}/lawService.do"
    params = {
        "OC":     LAW_API_KEY,
        "target": "prec",
        "ID":     prec_id,
        "type":   "XML",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.encoding = "utf-8"

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return None

    return {
        "판시사항": clean_html(root.findtext("판시사항", "")),
        "판결요지": clean_html(root.findtext("판결요지", "")),
        "참조조문": clean_html(root.findtext("참조조문", "")),
        "판례내용": clean_html(root.findtext("판례내용", ""))[:2000],
    }

def load_to_chroma(chunks: list[dict], embed_model):
    vectordb = Chroma(
        persist_directory  = CHROMA_DIR,
        embedding_function = embed_model,
        collection_name    = "precedents"
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
    print(f"  {len(new_chunks)}개 판례 적재 완료")
    return len(new_chunks)

def main():
    print("=" * 50)
    print("판례 데이터 적재 시작")
    print("=" * 50)

    print("\n임베딩 모델 로딩 중...")
    embed_model = get_embed_model()
    print("임베딩 모델 로딩 완료!")

    total = 0
    for item in SEARCH_KEYWORDS:
        keyword  = item["keyword"]
        category = item["category"]
        print(f"\n[{category}] [{keyword}] 판례 검색 중...")

        prec_list = fetch_prec_list(keyword, display=20)
        print(f"  검색된 판례 수: {len(prec_list)}개")

        if not prec_list:
            print("  판례 없음. 건너뜁니다.")
            continue

        chunks = []
        for prec in prec_list:
            content = fetch_prec_content(prec["id"])
            if not content:
                continue

            full_text = f"""[판례] {prec['case_name']}
사건번호: {prec['case_no']}
법원: {prec['court']} | 선고일자: {prec['date']} | 사건종류: {prec['case_type']}

[판시사항]
{content['판시사항']}

[판결요지]
{content['판결요지']}

[참조조문]
{content['참조조문']}"""

            chunks.append({
                "id":      f"prec_{prec['id']}",
                "content": full_text,
                "metadata": {
                    "type":      "precedent",
                    "category":  category,
                    "keyword":   keyword,
                    "case_name": prec["case_name"][:100],
                    "case_no":   prec["case_no"],
                    "court":     prec["court"],
                    "date":      prec["date"],
                    "case_type": prec["case_type"],
                }
            })

        count = load_to_chroma(chunks, embed_model)
        total += count

    print("\n" + "=" * 50)
    print(f"적재 완료! 총 {total}개 판례가 ChromaDB에 저장되었습니다.")
    print("=" * 50)

if __name__ == "__main__":
    main()
