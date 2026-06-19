import os
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

TARGET_LAWS = [
    {"query": "근로기준법",         "name": "근로기준법",         "category": "노동/고용"},
    {"query": "최저임금법",         "name": "최저임금법",         "category": "노동/고용"},
    {"query": "근로자퇴직급여보장법", "name": "근로자퇴직급여 보장법", "category": "노동/고용"},
    {"query": "남녀고용평등",        "name": "남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률", "category": "노동/고용"},
    {"query": "고용보험법",         "name": "고용보험법",         "category": "노동/고용"},
    {"query": "주택임대차보호법",    "name": "주택임대차보호법",   "category": "주거/임대"},
    {"query": "상가건물임대차보호법", "name": "상가건물 임대차보호법", "category": "주거/임대"},
    {"query": "집합건물",           "name": "집합건물의 소유 및 관리에 관한 법률", "category": "주거/임대"},
    {"query": "소비자기본법",        "name": "소비자기본법",       "category": "소비자/생활"},
    {"query": "전자상거래",         "name": "전자상거래 등에서의 소비자보호에 관한 법률", "category": "소비자/생활"},
    {"query": "할부거래",           "name": "할부거래에 관한 법률", "category": "소비자/생활"},
    {"query": "개인정보보호법",      "name": "개인정보 보호법",    "category": "개인정보/디지털"},
    {"query": "정보통신망법",        "name": "정보통신망 이용촉진 및 정보보호 등에 관한 법률", "category": "개인정보/디지털"},
    {"query": "청년고용촉진특별법",  "name": "청년고용촉진 특별법", "category": "청년/취업"},
    {"query": "직업안정법",         "name": "직업안정법",         "category": "청년/취업"},
]

def get_embed_model():
    return HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def fetch_law_id(query: str, law_name: str) -> str | None:
    url = f"{BASE_URL}/lawSearch.do"
    params = {
        "OC":      LAW_API_KEY,
        "target":  "law",
        "type":    "XML",
        "query":   query,
        "search":  1,
        "display": 5,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.encoding = "utf-8"
    root = ET.fromstring(resp.text)

    for law in root.findall(".//law"):
        name = law.findtext("법령명한글", "").strip()
        mst  = law.findtext("법령ID", "").strip()
        if name == law_name and mst:
            print(f"  법령 ID 조회 성공: {law_name} -> {mst}")
            return mst

    print(f"  법령 ID 조회 실패: {law_name}")
    return None

def fetch_law_content(mst: str) -> str | None:
    url = f"{BASE_URL}/lawService.do"
    params = {
        "OC":     LAW_API_KEY,
        "target": "law",
        "type":   "XML",
        "ID":     mst,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.encoding = "utf-8"
    return resp.text

def parse_articles(xml_text: str, law_name: str, category: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    chunks = []

    for jo in root.findall(".//조문단위"):
        jo_type = jo.findtext("조문여부", "").strip()
        if jo_type != "조문":
            continue

        jo_key     = jo.get("조문키", "").strip()
        jo_no      = jo.findtext("조문번호", "").strip()
        jo_title   = jo.findtext("조문제목", "").strip()
        jo_content = jo.findtext("조문내용", "").strip()

        contents = []
        for hang in jo.findall(".//항"):
            hang_content = hang.findtext("항내용", "").strip()
            if hang_content:
                contents.append(hang_content)

        if not contents and jo_content:
            contents.append(jo_content)

        if not contents:
            continue

        full_content = f"{jo_content}\n" + "\n".join(contents) if jo_content else "\n".join(contents)
        chunk_id = f"{law_name}_{jo_key}" if jo_key else f"{law_name}_{jo_no}"

        chunks.append({
            "id":      chunk_id,
            "content": full_content,
            "metadata": {
                "law_name":      law_name,
                "category":      category,
                "article_no":    f"제{jo_no}조",
                "article_title": jo_title,
                "collected_at":  datetime.date.today().isoformat(),
            }
        })

    return chunks

def load_to_chroma(chunks: list[dict], embed_model):
    vectordb = Chroma(
        persist_directory  = CHROMA_DIR,
        embedding_function = embed_model,
        collection_name    = "law_articles"
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
    print(f"  {len(new_chunks)}개 조문 적재 완료")
    return len(new_chunks)

def main():
    print("=" * 50)
    print("법령 데이터 적재 시작")
    print("=" * 50)

    print("\n임베딩 모델 로딩 중...")
    embed_model = get_embed_model()
    print("임베딩 모델 로딩 완료!")

    total = 0
    for law in TARGET_LAWS:
        print(f"\n[{law['category']}] [{law['name']}] 처리 중...")

        mst = fetch_law_id(law["query"], law["name"])
        if not mst:
            continue

        xml_text = fetch_law_content(mst)
        if not xml_text:
            print(f"  본문 조회 실패: {law['name']}")
            continue

        chunks = parse_articles(xml_text, law["name"], law["category"])
        print(f"  파싱된 조문 수: {len(chunks)}개")

        count = load_to_chroma(chunks, embed_model)
        total += count

    print("\n" + "=" * 50)
    print(f"적재 완료! 총 {total}개 조문이 ChromaDB에 저장되었습니다.")
    print("=" * 50)

if __name__ == "__main__":
    main()
