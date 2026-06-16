"""
국가법령정보센터 Open API를 통해 법령 원문을 수집하고
조문 단위로 파싱하여 ChromaDB에 적재하는 스크립트
"""
import os
import re
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

LAW_API_KEY = os.getenv("LAW_API_KEY", "legalrag2026")
CHROMA_DIR  = os.getenv("CHROMA_DIR", "./chroma_db")
BASE_URL    = "http://www.law.go.kr/DRF"

# 적재 대상 법령 목록
TARGET_LAWS = [
    "근로기준법",
    "최저임금법",
    "근로자퇴직급여 보장법",
    "남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률",
    "고용보험법",
]

# 임베딩 모델 초기화 (ko-sroberta-multitask)
def get_embed_model():
    return HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def fetch_law_id(law_name: str) -> str | None:
    """법령명으로 법령 ID(MST) 조회"""
    url = f"{BASE_URL}/lawSearch.do"
    params = {
        "OC":     LAW_API_KEY,
        "target": "law",
        "type":   "XML",
        "query":  law_name,
        "search": 1,
        "display": 5,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.encoding = "utf-8"
    root = ET.fromstring(resp.text)

    for law in root.findall(".//law"):
        name = law.findtext("법령명한글", "").strip()
        mst  = law.findtext("법령ID", "").strip()
        if name == law_name and mst:
            print(f"  법령 ID 조회 성공: {law_name} → {mst}")
            return mst

    print(f"  법령 ID 조회 실패: {law_name}")
    return None

def fetch_law_content(mst: str) -> str | None:
    """법령 ID로 법령 본문 XML 조회"""
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

def parse_articles(xml_text: str, law_name: str) -> list[dict]:
    """
    법령 본문 XML을 파싱하여 조문 단위 청크 리스트 반환
    반환 형식: [{"id": ..., "content": ..., "metadata": {...}}, ...]
    """
    root = ET.fromstring(xml_text)
    chunks = []
    import datetime

    for jo in root.findall(".//조문단위"):
        # 조문여부가 "조문"인 것만 파싱 (전문=장제목 제외)
        jo_type = jo.findtext("조문여부", "").strip()
        if jo_type != "조문":
            continue

        jo_key   = jo.get("조문키", "").strip()
        jo_no    = jo.findtext("조문번호", "").strip()
        jo_title = jo.findtext("조문제목", "").strip()
        jo_content = jo.findtext("조문내용", "").strip()

        # 항 내용 수집
        contents = []
        for hang in jo.findall(".//항"):
            hang_content = hang.findtext("항내용", "").strip()
            if hang_content:
                contents.append(hang_content)

        # 항이 없으면 조문내용 직접 사용
        if not contents and jo_content:
            contents.append(jo_content)

        if not contents:
            continue

        full_content = f"{jo_content}\n" + "\n".join(contents) if jo_content else "\n".join(contents)

        # 청크 ID: 법령명_조문번호 (중복 방지)
        chunk_id = f"{law_name}_{jo_key}" if jo_key else f"{law_name}_{jo_no}"

        chunks.append({
            "id":      chunk_id,
            "content": full_content,
            "metadata": {
                "law_name":      law_name,
                "article_no":    f"제{jo_no}조",
                "article_title": jo_title,
                "collected_at":  datetime.date.today().isoformat(),
            }
        })

    return chunks

def load_to_chroma(chunks: list[dict], embed_model):
    """청크를 ChromaDB에 적재 (중복 방지)"""
    vectordb = Chroma(
        persist_directory  = CHROMA_DIR,
        embedding_function = embed_model,
        collection_name    = "law_articles"
    )

    # 기존 ID 조회 (중복 방지)
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

    print("\n임베딩 모델 로딩 중... (최초 실행 시 다운로드 발생)")
    embed_model = get_embed_model()
    print("임베딩 모델 로딩 완료!")

    total = 0
    for law_name in TARGET_LAWS:
        print(f"\n[{law_name}] 처리 중...")

        mst = fetch_law_id(law_name)
        if not mst:
            continue

        xml_text = fetch_law_content(mst)
        if not xml_text:
            print(f"  본문 조회 실패: {law_name}")
            continue

        chunks = parse_articles(xml_text, law_name)
        print(f"  파싱된 조문 수: {len(chunks)}개")

        count = load_to_chroma(chunks, embed_model)
        total += count

    print("\n" + "=" * 50)
    print(f"적재 완료! 총 {total}개 조문이 ChromaDB에 저장됐습니다.")
    print("=" * 50)

if __name__ == "__main__":
    main()
