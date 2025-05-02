# AI 기반 신용도판단정보 JSON 검증 시스템

📌 일반신용정보관리규약을 기반으로 JSON 데이터를 심층 검증하는 Streamlit 기반 애플리케이션입니다.  
로컬 LLM (Ollama) + 문서 벡터 검색(RAG)을 이용해 규약 위반 여부를 분석합니다.

## 기능
- PDF 기반 규약 문서 자동 임베딩
- JSON 데이터 자동 분석 → 유사 규약 문장 추출
- LLM을 통한 규약 위반 판별
- 결과는 마크다운 형식으로 시각화

## Powershell 에서 오류시
PSSecurityException
    + FullyQualifiedErrorId : UnauthorizedAccess
-----------------
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass


## 실행 방법

```bash
# 가상환경 실행
venv\\Scripts\\activate     # Windows
source venv/bin/activate    # macOS/Linux

# 의존성 설치
pip install -r requirements.txt

# 실행
streamlit run Rag_Verification_Server.py
