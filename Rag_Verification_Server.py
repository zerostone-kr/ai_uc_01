
import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import faiss
import numpy as np
import json
import logging
import time
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import requests
import re
import os
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# 로깅 설정
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# -----------------------
# 설정
# -----------------------
EMBEDDING_MODEL_NAME = "jhgan/ko-sbert-nli"
FAISS_INDEX_FILE = "regulation_index.faiss"
TEXT_LIST_FILE = "regulation_sentences.json"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "my-ko-model"
FIXED_PDF_PATH = "일반신용정보관리규약(제2025-4차 신용정보집중관리위원회).pdf"

sentence_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
faiss_index = None
loaded_sentences = None

if os.path.exists(FAISS_INDEX_FILE):
    faiss_index = faiss.read_index(FAISS_INDEX_FILE)
    with open(TEXT_LIST_FILE, encoding="utf-8") as f:
        loaded_sentences = json.load(f)

# -----------------------
# 규약 PDF → 텍스트 추출
# -----------------------
def extract_text_from_pdf(pdf_path):
    logging.info("PDF에서 텍스트 추출 시작...")
    start_time = time.time()
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    logging.info(f"텍스트 추출 완료. 소요 시간: {time.time() - start_time:.2f}초")
    return text

# -----------------------
# 텍스트 전체 → 문장 분할
# -----------------------
def extract_all_sentences(text):
    logging.info("문장 분할 시작...")
    start_time = time.time()
    sentences = re.split(r'[\n\r\.\?]+', text)
    result = [s.strip() for s in sentences if len(s.strip()) > 10]
    logging.info(f"문장 분할 완료. 소요 시간: {time.time() - start_time:.2f}초")
    return result

# -----------------------
# 문장 → 벡터 저장
# -----------------------
def build_faiss_index(sentences):
    global faiss_index, loaded_sentences
    logging.info("FAISS 인덱스 생성 및 저장 시작...")
    start_time = time.time()
    embeddings = sentence_model.encode(sentences, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(TEXT_LIST_FILE, "w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False)
    faiss_index = index
    loaded_sentences = sentences
    logging.info(f"인덱스 저장 완료. 소요 시간: {time.time() - start_time:.2f}초")
    return index, sentences

# -----------------------
# 유사 문장 검색
# -----------------------
def search_similar_sentences(query, top_k=5):
    logging.info(f"유사 문장 검색 시작... 쿼리: {query}")
    start_time = time.time()
    query_vec = sentence_model.encode([query]).astype("float32")
    D, I = faiss_index.search(query_vec, top_k)
    logging.info(f"유사 문장 검색 완료. 소요 시간: {time.time() - start_time:.2f}초")
    return [loaded_sentences[i] for i in I[0]]

# -----------------------
# 유사 문장 중복 제거
# -----------------------
def deduplicate_sentences(sentences, threshold=0.95):
    if len(sentences) <= 1:
        return sentences
    embeddings = sentence_model.encode(sentences, convert_to_numpy=True)
    sim_matrix = cosine_similarity(embeddings)
    unique_indices = []
    for i in range(len(sentences)):
        if all(sim_matrix[i][j] < threshold for j in unique_indices):
            unique_indices.append(i)
    return [sentences[i] for i in unique_indices]

# -----------------------
# JSON 사전 검증 로직
# -----------------------
def prevalidate_json(user_json):
    try:
        data = json.loads(user_json)
        violations = []

        if data.get("특수채권여부") == True:
            if not str(data.get("등록코드", "")).startswith("7"):
                violations.append("- [등록코드]: 특수채권인데 등록코드가 7로 시작하지 않음")

        return violations

    except Exception as e:
        return [f"⚠️ JSON 파싱 에러: {str(e)}"]

# -----------------------
# Ollama 로컬 API 호출
# -----------------------
def call_ollama_with_prompt(context, user_json):
    prompt = f"""
다음은 한국의 일반신용정보관리규약 핵심 요약입니다:
{chr(10).join([f"- {line}" for line in context])}

아래는 검토 대상 JSON입니다. 분석에만 참고하고 출력에는 절대 포함하지 마세요.

{user_json}

다음 질문에 대답하세요:

❓ 이 JSON 데이터에 규약 위반 항목이 있다면, 아래 형식으로만 간단히 작성해 주세요:

- [항목명]: 위반 사유

🛑 절대 다음 내용을 출력하지 마세요:
- "다음은", "예시", "출력 형식" 등의 설명 문장
- JSON 내용 또는 항목별 설명
- 출력 형식 안내
- 규약 위반이 없으면 아무 것도 출력하지 마세요.

단순히 규약 위반 항목과 그 사유만 출력하세요.
"""

    logging.info("📡 Ollama API 호출 시작...")
    start = time.time()
    response = requests.post(OLLAMA_API_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    })
    elapsed = time.time() - start
    logging.info(f"✅ Ollama 응답 완료. 소요 시간: {elapsed:.2f}초")

    try:
        raw_output = response.json().get("response", "⚠️ 응답에 'response' 항목이 없습니다.")
    except json.JSONDecodeError:
        return f"⚠️ 응답 JSON 파싱 실패:\n{response.text}"

    if isinstance(raw_output, str):
        filtered_lines = []
        for line in raw_output.splitlines():
            if any(bad_word in line for bad_word in [    "예시", "출력 형식", "다음은", "※", "설명", "결과입니다",    "아래와 같은 질문", "다음 질문에 대답", "❓"]):
                continue
            if line.strip() == "":
                continue
            filtered_lines.append(line)
        
        if not filtered_lines:
            return "✅ 검증 결과: 이상 없음"
        
        return "\n".join(filtered_lines).strip()

    return raw_output

# -----------------------
# Streamlit 인터페이스
# -----------------------
st.set_page_config(page_title="신용도판단정보 검증", layout="wide")
st.title("📄 일반신용정보관리규약 기반 검증 (로컬 Ollama API)")

# PDF 로딩은 최초 실행 시 1회만 수행
if 'initialized' not in st.session_state:
    with st.spinner("📘 규약 PDF 처리 중입니다..."):
        if not os.path.exists(FIXED_PDF_PATH):
            st.error("❌ 고정된 규약 PDF 파일이 존재하지 않습니다.")
            st.stop()
        else:
            text = extract_text_from_pdf(FIXED_PDF_PATH)
            all_sentences = extract_all_sentences(text)
            build_faiss_index(all_sentences)
            st.session_state['all_sentences'] = all_sentences
            st.session_state['initialized'] = True
            st.success(f"✅ 규약 PDF 처리 완료. 총 {len(all_sentences)}개의 문장을 벡터화했습니다.")

st.markdown("## 🧪 JSON 검증 테스트")

json_input = st.text_area("검증 대상 JSON 입력", height=300, value='''{
  "신용정보유형": "신용도판단정보",
  "등록기관": "국민은행",
  "등록일자": "2025-03-10",
  "식별정보": {
    "성명": "홍길동",
    "주민등록번호": "901010-1234567"
  },
  "정보유형": "연체정보",
  "등록코드": "0101",
  "연체기산일": "2025-01-01",
  "등록사유발생일": "2025-03-03",
  "등록금액": 1500000,
  "연체금액": 1500000,
  "특수채권여부": true,
  "관련인여부": false,
  "보전처분등존재여부": false,
  "해제사유": null,
  "소송중여부": false,
  "비고": "신용카드 청구대금 미결제에 따른 연체"
}''')

query = st.text_input("질문 요약 (예: 연체정보 등록 기준)", "연체정보 등록 조건과 특수채권 등록 기준 위반 여부 검토")

if st.button("🚀 검증 요청"):
    with st.spinner("검증 중입니다..."):
        precheck = prevalidate_json(json_input)
        top_sentences = deduplicate_sentences(search_similar_sentences(query, top_k=10))
        result = call_ollama_with_prompt(top_sentences, json_input)

        # ✅ 사전검증 및 LLM 응답 통합 처리 (이상 없음 출력 완전 차단 조건 포함)
        final_lines = []

        if precheck:
            final_lines.extend(precheck)

        if result and result.strip() and result.strip() != "✅ 검증 결과: 이상 없음":
            final_lines.append(result.strip())

        if not final_lines:
            final_lines.append("✅ 검증 결과: 이상 없음")

        st.markdown("### ✅ 검증 결과\n```markdown\n" + "\n".join(final_lines) + "\n```") 
