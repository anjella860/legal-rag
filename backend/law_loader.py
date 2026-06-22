import os
import re
import time
import datetime
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

LAW_API_KEY = os.getenv("LAW_API_KEY", "legalrag2026")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
BASE_URL = "http://www.law.go.kr/DRF"

TARGET_LAWS = [
    {"query": "근로기준법", "name": "근로기준법", "category": "노동/고용"},
    {"query": "최저임금법", "name": "최저임금법", "category": "노동/고용"},
    {
        "query": "근로자퇴직급여보장법",
        "name": "근로자퇴직급여 보장법",
        "category": "노동/고용",
    },
    {
        "query": "남녀고용평등",
        "name": "남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률",
        "category": "노동/고용",
    },
    {"query": "고용보험법", "name": "고용보험법", "category": "노동/고용"},
    {"query": "산업안전보건법", "name": "산업안전보건법", "category": "노동/고용"},
    {
        "query": "기간제 및 단시간근로자",
        "name": "기간제 및 단시간근로자 보호 등에 관한 법률",
        "category": "노동/고용",
    },
    {
        "query": "파견근로자보호",
        "name": "파견근로자 보호 등에 관한 법률",
        "category": "노동/고용",
    },
    {
        "query": "노동조합 및 노동관계조정법",
        "name": "노동조합 및 노동관계조정법",
        "category": "노동/고용",
    },
    {"query": "근로복지기본법", "name": "근로복지기본법", "category": "노동/고용"},
    {
        "query": "채용절차의 공정화",
        "name": "채용절차의 공정화에 관한 법률",
        "category": "노동/고용",
    },
    {"query": "주택임대차보호법", "name": "주택임대차보호법", "category": "주거/임대"},
    {
        "query": "상가건물임대차보호법",
        "name": "상가건물 임대차보호법",
        "category": "주거/임대",
    },
    {
        "query": "집합건물",
        "name": "집합건물의 소유 및 관리에 관한 법률",
        "category": "주거/임대",
    },
    {
        "query": "민간임대주택",
        "name": "민간임대주택에 관한 특별법",
        "category": "주거/임대",
    },
    {"query": "공동주택관리법", "name": "공동주택관리법", "category": "주거/임대"},
    {
        "query": "부동산 거래신고",
        "name": "부동산 거래신고 등에 관한 법률",
        "category": "주거/임대",
    },
    {"query": "공인중개사법", "name": "공인중개사법", "category": "주거/임대"},
    {"query": "소비자기본법", "name": "소비자기본법", "category": "소비자/생활"},
    {
        "query": "전자상거래",
        "name": "전자상거래 등에서의 소비자보호에 관한 법률",
        "category": "소비자/생활",
    },
    {"query": "할부거래", "name": "할부거래에 관한 법률", "category": "소비자/생활"},
    {"query": "방문판매", "name": "방문판매 등에 관한 법률", "category": "소비자/생활"},
    {
        "query": "약관의 규제",
        "name": "약관의 규제에 관한 법률",
        "category": "소비자/생활",
    },
    {"query": "제조물 책임법", "name": "제조물 책임법", "category": "소비자/생활"},
    {
        "query": "표시광고의 공정화",
        "name": "표시ㆍ광고의 공정화에 관한 법률",
        "category": "소비자/생활",
    },
    {
        "query": "개인정보보호법",
        "name": "개인정보 보호법",
        "category": "개인정보/디지털",
    },
    {
        "query": "정보통신망법",
        "name": "정보통신망 이용촉진 및 정보보호 등에 관한 법률",
        "category": "개인정보/디지털",
    },
    {
        "query": "위치정보",
        "name": "위치정보의 보호 및 이용 등에 관한 법률",
        "category": "개인정보/디지털",
    },
    {
        "query": "전자문서",
        "name": "전자문서 및 전자거래 기본법",
        "category": "개인정보/디지털",
    },
    {"query": "전자서명법", "name": "전자서명법", "category": "개인정보/디지털"},
    {
        "query": "신용정보",
        "name": "신용정보의 이용 및 보호에 관한 법률",
        "category": "개인정보/디지털",
    },
    {
        "query": "청년고용촉진특별법",
        "name": "청년고용촉진 특별법",
        "category": "청년/취업",
    },
    {"query": "직업안정법", "name": "직업안정법", "category": "청년/취업"},
    {"query": "고용정책기본법", "name": "고용정책 기본법", "category": "청년/취업"},
    {
        "query": "국민 평생 직업능력 개발법",
        "name": "국민 평생 직업능력 개발법",
        "category": "청년/취업",
    },
    {"query": "민법", "name": "민법", "category": "민사/계약"},
    {"query": "민사소송법", "name": "민사소송법", "category": "민사/계약"},
    {"query": "민사집행법", "name": "민사집행법", "category": "민사/계약"},
    {"query": "상법", "name": "상법", "category": "민사/계약"},
    {"query": "형법", "name": "형법", "category": "형사/안전"},
    {"query": "형사소송법", "name": "형사소송법", "category": "형사/안전"},
    {"query": "도로교통법", "name": "도로교통법", "category": "형사/안전"},
    {
        "query": "자동차손해배상 보장법",
        "name": "자동차손해배상 보장법",
        "category": "형사/안전",
    },
    {"query": "행정기본법", "name": "행정기본법", "category": "행정/공공"},
    {"query": "행정절차법", "name": "행정절차법", "category": "행정/공공"},
    {"query": "행정소송법", "name": "행정소송법", "category": "행정/공공"},
    {
        "query": "공공기관의 정보공개",
        "name": "공공기관의 정보공개에 관한 법률",
        "category": "행정/공공",
    },
]


def get_embed_model():
    return HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_law_name(name: str) -> str:
    return re.sub(r"\s+", "", name or "").strip()


def deduplicate_target_laws(laws: list[dict]) -> list[dict]:
    seen = set()
    result = []

    for law in laws:
        key = normalize_law_name(law["name"])
        if key in seen:
            continue
        seen.add(key)
        result.append(law)

    return result


def fetch_law_id(query: str, law_name: str) -> str | None:
    url = f"{BASE_URL}/lawSearch.do"
    params = {
        "OC": LAW_API_KEY,
        "target": "law",
        "type": "XML",
        "query": query,
        "search": 1,
        "display": 10,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.encoding = "utf-8"
    except requests.RequestException as e:
        print(f"  요청 실패: {law_name} / {e}")
        return None

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        print(f"  XML 파싱 오류: {law_name}")
        print(resp.text[:300])
        return None

    target_name = normalize_law_name(law_name)

    for law in root.findall(".//law"):
        name = law.findtext("법령명한글", "").strip()
        mst = law.findtext("법령ID", "").strip()

        if normalize_law_name(name) == target_name and mst:
            print(f"  법령 ID 조회 성공: {law_name} -> {mst}")
            return mst

    print(f"  법령 ID 조회 실패: {law_name}")
    return None


def fetch_law_content(mst: str, law_name: str) -> str | None:
    url = f"{BASE_URL}/lawService.do"
    params = {
        "OC": LAW_API_KEY,
        "target": "law",
        "type": "XML",
        "ID": mst,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.encoding = "utf-8"
        return resp.text
    except requests.RequestException as e:
        print(f"  본문 요청 실패: {law_name} / {e}")
        return None


def extract_article_text(jo: ET.Element) -> str:
    jo_content = clean_text(jo.findtext("조문내용", ""))
    contents = []

    if jo_content:
        contents.append(jo_content)

    for hang in jo.findall(".//항"):
        hang_content = clean_text(hang.findtext("항내용", ""))
        if hang_content:
            contents.append(hang_content)

        for ho in hang.findall(".//호"):
            ho_content = clean_text(ho.findtext("호내용", ""))
            if ho_content:
                contents.append(ho_content)

    return "\n".join(contents).strip()


def parse_articles(xml_text: str, law_name: str, category: str) -> list[dict]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        print(f"  본문 XML 파싱 오류: {law_name}")
        return []

    chunks = []

    for jo in root.findall(".//조문단위"):
        jo_type = jo.findtext("조문여부", "").strip()
        if jo_type != "조문":
            continue

        jo_key = jo.get("조문키", "").strip()
        jo_no = jo.findtext("조문번호", "").strip()
        jo_title = clean_text(jo.findtext("조문제목", ""))

        full_content = extract_article_text(jo)

        if not full_content:
            continue

        article_no = f"제{jo_no}조" if jo_no else ""
        article_title_text = f"({jo_title})" if jo_title else ""

        full_text = f"""[현행법령] {law_name}
분류: {category}

[조문]
{article_no} {article_title_text}

[내용]
{full_content}"""

        chunk_id = (
            f"law_{normalize_law_name(law_name)}_{jo_key}"
            if jo_key
            else f"law_{normalize_law_name(law_name)}_{jo_no}"
        )

        chunks.append(
            {
                "id": chunk_id,
                "content": full_text,
                "metadata": {
                    "type": "law",
                    "law_name": law_name,
                    "category": category,
                    "article_no": article_no,
                    "article_title": jo_title,
                    "collected_at": datetime.date.today().isoformat(),
                },
            }
        )

    return chunks


def load_to_chroma(chunks: list[dict], embed_model):
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embed_model,
        collection_name="law_articles",
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
    target_laws = deduplicate_target_laws(TARGET_LAWS)

    for law in target_laws:
        print(f"\n[{law['category']}] [{law['name']}] 처리 중...")

        mst = fetch_law_id(law["query"], law["name"])
        time.sleep(0.1)

        if not mst:
            continue

        xml_text = fetch_law_content(mst, law["name"])
        time.sleep(0.1)

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
