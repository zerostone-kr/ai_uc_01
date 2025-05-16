# Aws_Rag_Verification_Server.py
import streamlit as st
import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import boto3
from PyPDF2 import PdfReader

# ===============================
# AWS Bedrock LLM 호출 설정
# ===============================
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

def query_llm(prompt: str) -> str:
    body = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 2048,
            "temperature": 0.3,
            "topP": 0.9
        }
    }

    response = bedrock_client.invoke_model_with_response_stream(
        body=json.dumps(body),
        modelId="meta.llama3-70b-instruct-v1:0",
        inferenceConfiguration={
            "inferenceProfileArn": "arn:aws:bedrock:us-east-1:484907498824:inference-profile/us.amazon.nova-premier-v1:0"
        },
        accept="application/json",
        contentType="application/json"
    )

    result = ""
    for event in response['body']:
        if 'chunk' in event and 'bytes' in event['chunk']:
            result += event['chunk']['bytes'].decode("utf-8")
    return result

# ===============================
# PDF 규약집 임베딩 처리
# ===============================
@st.cache_resource
def load_pdf_sentences(pdf_path):
    reader = PdfReader(pdf_path)
    all_text = ""
    for page in reader.pages:
        all_text += page.extract_text() + "\n"
    return [line.strip() for line in all_text.split("\n") if len(line.strip()) > 20]

@st.cache_resource
def build_faiss_index(sentences):
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    embeddings = model.encode(sentences)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings, model

def search_similar_sentences(query, model, index, sentences, k=3):
    embedding = model.encode([query])
    D, I = index.search(np.array(embedding), k)
    return [sentences[i] for i in I[0]]

# ===============================
# Streamlit UI 시작
# ===============================
st.title("신용정보 규약 기반 LLM 검증 시스템 (AWS Bedrock 기반)")

pdf_path = "일반신용정보관리규약(제2025-4차 신용정보집중관리위원회).pdf"

if not os.path.exists(pdf_path):
    st.error("PDF 규약집 파일이 존재하지 않습니다.")
else:
    sentences = load_pdf_sentences(pdf_path)
    index, embeddings, model = build_faiss_index(sentences)

# 정보형태 선택
info_type = st.selectbox("정보형태를 선택하세요", ["연체정보", "금융질서문란정보"])

# 샘플 JSON 양식 정의
if info_type == "연체정보":
    default_json = {
        "정보형태": "연체정보",
        "등록사유코드": "0101",
        "등록일자": "2025-03-01",
        "발생금액": "2000000",
        "지연일수": "45",
        "계좌번호": "12345678901234",
        "금융기관명": "신한은행",
        "연체구분": "단기연체",
        "연체원금": "1800000",
        "연체이자": "200000",
        "상환여부": "N",
        "상환일자": "",
        "최종변제예정일": "2025-05-01",
        "성명": "홍길동",
        "주민등록번호": "800101-1234567"
    }
elif info_type == "금융질서문란정보":
    default_json = {
        "정보형태": "금융질서문란정보",
        "등록사유코드": "0702",
        "등록일자": "2025-04-10",
        "사건번호": "서울중앙지방법원2024형제1001",
        "금융기관명": "국민은행",
        "사유내용": "대출사기",
        "판결유형": "유죄",
        "처분일자": "2025-04-05",
        "처분내용": "벌금형",
        "성명": "김진수",
        "주민등록번호": "850505-1234567"
    }

input_text = st.text_area("검증 대상 JSON을 입력하세요:", value=json.dumps(default_json, indent=4, ensure_ascii=False), height=400)

if st.button("검증 실행"):
    try:
        data = json.loads(input_text)
        st.json(data)

        queries = [f"{key}: {value}" for key, value in data.items()]
        if '식별정보' in data and isinstance(data['식별정보'], dict):
            for k, v in data['식별정보'].items():
                queries.append(f"{k}: {v}")

        output_lines = []
        for query in queries:
            similar_sents = []
            for q in query.split(","):
                similar_sents += search_similar_sentences(q.strip(), model, index, sentences)
            similar_sents = list(set(similar_sents))

            context = "\n".join(similar_sents[:5])
            prompt = f"""너는 신용정보 검증 시스템이야. 아래 JSON 항목의 값을 일반신용정보관리규약 기준에 따라 오류 여부를 판단해.

[규약 내용]
{context}

[검증할 내용]
{query}

- 위 항목이 규약에 위배되지 않으면 '✅ 검증 결과: 이상 없음'이라고 출력해.
- 위배된 항목이 있다면 '- 항목명: 위반 사유' 형식으로 설명해."""

            result = query_llm(prompt)

            if any(term in result for term in ["예시", "출력 형식", "※", "결과입니다", "다음은"]):
                continue

            output_lines.append(result.strip())

        st.subheader("검증 결과")
        st.text("\n".join(output_lines))

    except json.JSONDecodeError:
        st.error("JSON 형식이 잘못되었습니다. 올바른 형식으로 입력해주세요.")
