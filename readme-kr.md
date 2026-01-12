# ColPali 기반 문서 검색 프로젝트

이 프로젝트는 기업 내부의 대규모 문서 검색을 효율적으로 수행하기 위해 최신 Vision Language Model(VLM)인 ColPali를 기반으로 한 검색 시스템을 구현합니다. 이 README는 프로젝트의 배경, 기술적 선택 이유, 데이터셋 구성, 최적화 기법, RAG(Retrieval-Augmented Generation) 구현 방법, 그리고 실행 방법에 대해 상세히 설명합니다.

## 1. 기업 내부 문서 검색과 ColPali의 대두 배경 및 장점

기업 환경에서는 매일 수많은 내부 문서(보고서, 계약서, 인보이스, 이메일 등)가 생성되고 저장됩니다. 이러한 문서를 효율적으로 검색하고 필요한 정보를 빠르게 추출하는 것은 업무 생산성에 직접적인 영향을 미칩니다. 전통적으로, 문서 검색은 OCR(Optical Character Recognition) 기술을 기반으로 이루어졌습니다. OCR은 단순히 텍스트를 추출할 뿐, 표, 다이어그램, 섹션 제목 등 문서의 시각적 구조를 이해하지 못합니다.
실제 기업 문서는 종종 흐릿하거나 왜곡된 스캔 이미지로 제공되며, OCR은 이러한 저품질 데이터에서 정확도가 떨어집니다. OCR은 텍스트를 개별 단위로 처리하기 때문에, 문서 내에서 텍스트와 시각적 요소 간의 관계를 반영하지 못합니다.

이러한 한계를 극복하기 위해 **ColPali**라는 새로운 접근법이 대두되었습니다. ColPali는 Vision Language Model(VLM)을 기반으로 한 혁신적인 문서 검색 프레임워크로, ColBERT의 'late interaction' 메커니즘과 PaLiGemma와 같은 VLM을 결합하여 문서의 텍스트와 시각적 요소를 통합적으로 이해합니다. ColPali의 주요 장점은 다음과 같습니다:

- **텍스트와 시각적 요소의 통합 이해**: 문서의 레이아웃, 표, 이미지, 텍스트 간의 관계를 학습하여 더 풍부한 컨텍스트를 제공합니다.
- **OCR 의존성 감소**: OCR 없이도 문서 이미지를 직접 처리할 수 있어, OCR 오류로 인한 정보 손실을 방지합니다.
- **효율적인 검색**: 다중 벡터(multi-vector) 표현을 통해 쿼리와 문서 간의 세밀한 매칭을 가능하게 하며, 대규모 문서 컬렉션에서도 빠른 검색을 지원합니다.

## 2. ColPali 대신 TomoroAI/tomoro-colqwen3-embed-4b 선택 이유와 장점

이 프로젝트에서는 원래의 ColPali 모델 대신 `TomoroAI/tomoro-colqwen3-embed-4b` 모델을 선택하여 사용했습니다. 이 모델은 Qwen3-VL-4B-Instruct를 기반으로 한 ColBERT 스타일의 다중 벡터 표현을 생성하는 확장판으로, 다음과 같은 이유와 장점으로 선택되었습니다:

- **다국어 지원**: `tomoro-colqwen3-embed-4b `은 영어와 독일어를 포함한 여러 언어를 지원하는 데이터셋으로 학습되었습니다. 이는 글로벌 기업 환경에서 다양한 언어로 작성된 문서를 처리해야 하는 요구사항에 적합합니다.
- **효율적인 아키텍처**: Qwen2.5-VL-3B 기반으로 설계된 이 모델은 3B 파라미터 규모로, 원본 ColPali보다 경량화되어 더 적은 리소스로도 높은 성능을 발휘합니다. 특히, 8xH100 GPU 환경에서 `bfloat16` 형식으로 학습된 이 모델은 메모리 효율성이 뛰어납니다.
- **동적 이미지 해상도 처리**: 이 모델은 고정된 해상도가 아닌 동적 해상도를 지원하며, 이미지 패치 수를 최대 768개로 제한하여 메모리 사용량과 검색 품질 간의 균형을 맞춥니다. 이는 다양한 크기와 품질의 문서 이미지를 처리하는 데 유리합니다.
- **커뮤니티 검증**: Hugging Face에서 41,000회 이상 다운로드된 이 모델은 이미 많은 사용자와 연구자들에 의해 검증되었으며, 안정적인 성능을 보장합니다.

## 3. 데이터셋을 RVL-CDIP로 변경한 이유와 장점

원본 ColPali 프로젝트에서는 자체 샘플 데이터셋을 사용했으나, 이 프로젝트에서는 **RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing)** 데이터셋으로 변경하여 사용했습니다. 변경 이유와 장점은 다음과 같습니다:

- **대규모 실제 문서 데이터**: RVL-CDIP는 400,000장의 흑백 스캔 문서 이미지로 구성된 대규모 데이터셋으로, IIT-CDIP Test Collection의 서브셋입니다. 이 데이터셋은 저해상도(100dpi), 노이즈, 품질 저하 등 실제 기업 환경에서 접하는 문서의 특성을 반영하여, 실세계 적용 가능성을 높입니다.
- **다양한 문서 카테고리**: RVL-CDIP는 16개 카테고리(Letter, Form, Email, Handwritten, Advertisement, Scientific Report, Scientific Publication, Specification, File Folder, News Article, Budget, Invoice, Presentation, Questionnaire, Resume, Memo)로 구성되어 있어, 기업 내부 문서의 다양한 유형을 포괄합니다. 각 카테고리당 25,000장의 이미지가 포함되어 있어 균형 잡힌 학습과 테스트가 가능합니다.
- **샘플링 및 데이터 준비**: 이 프로젝트에서는 RVL-CDIP 데이터셋에서 각 카테고리별로 대표 샘플을 추출하여 총 1,600장의 이미지를 사용했습니다. 이 샘플들은 이미 `samples/` 디렉터리에 준비되어 있어, 별도의 데이터 전처리 없이 바로 데모를 실행할 수 있습니다. 샘플링을 통해 대규모 데이터셋의 부담을 줄이면서도 다양한 문서 유형을 대표할 수 있도록 설계되었습니다.
- **연구 및 산업 표준**: RVL-CDIP는 문서 이미지 분류 및 검색 연구에서 널리 사용되는 표준 데이터셋으로, LayoutLMv3와 같은 최신 문서 AI 모델의 벤치마크로 활용됩니다. 이를 통해 프로젝트 결과의 신뢰성과 비교 가능성을 높일 수 있습니다.

## 4. ColPali의 한계와 이를 극복하기 위한 최적화 구현

ColPali는 혁신적인 문서 검색 프레임워크이지만, 다음과 같은 한계가 존재합니다:

- **대규모 문서 처리의 계산 비용**: ColPali는 다중 벡터 표현을 사용하기 때문에, 수백만 장의 문서를 인덱싱하고 검색하는 데 상당한 계산 리소스가 필요합니다. 특히, 각 문서의 모든 토큰/패치에 대해 벡터를 생성하고 저장하는 것은 메모리와 스토리지 부담을 가중시킵니다.
- **검색 지연 시간**: 다중 벡터 간의 'late interaction' 매칭은 정확도를 높이지만, 대규모 데이터셋에서 쿼리 응답 시간을 늘릴 수 있습니다.
- **PDF 중심의 초점**: ColPali는 주로 PDF 타입의 문서와 고자원 언어(영어 등)에 초점을 맞추어 학습되었기 때문에, 다른 형식의 문서나 저자원 언어에서는 성능이 제한될 수 있습니다.

이러한 한계를 극복하기 위해, 이 프로젝트에서는 원본 `elasticsearch-labs`의 최적화 기법을 차용하여 다음과 같은 세 가지 최적화 전략을 구현했습니다:

- **Average Vector (평균 벡터)**: 대규모 검색의 속도를 높이기 위해, 문서당 다중 벡터를 하나의 평균 벡터(`dense_vector`)로 압축하여 1차 검색(`knn`)을 수행합니다. 이는 계산 비용과 검색 지연 시간을 크게 줄여줍니다. (Part 2 - Step 6)
- **BBQ (Better Binary Quantization)**: 저장 공간과 메모리 사용량을 극적으로 줄이기 위해, `rank_vectors` 필드에 `element_type: 'bit'` 설정을 적용하여 32비트 float 벡터를 1비트 바이너리 형식으로 양자화합니다. 이는 저장 공간을 약 95% 절약하며, 리스코어링 단계의 비용 효율성을 높입니다. (Part 2 - Step 5)
- **Rescore (리스코어링)**: 평균 벡터로 1차 검색한 후보군에 대해, BBQ가 적용된 `rank_vectors`를 사용하여 2차적으로 정밀한 점수를 다시 계산합니다. 이를 통해 1차 검색의 속도와 2차 검색의 정확도를 모두 확보하는 2단계 아키텍처를 완성합니다. (Part 2 - Step 7)

이 세 가지 최적화는 속도, 정확도, 비용이라는 세 가지 목표를 동시에 달성하는 고도의 검색 아키텍처를 구현하며, ColPali의 한계를 효과적으로 극복합니다.

## 5. RAG with ColPali using Inference API (Part 3)

이 섹션에서는 Elastic의 Inference API를 활용하여 ColPali 기반 문서 검색 시스템에 RAG(Retrieval-Augmented Generation) 기능을 통합하는 방법을 설명합니다. 특히 Amazon Bedrock과 연동된 LLM을 사용하여 자연어 쿼리에 대한 응답을 생성하는 데 초점을 맞춥니다.

ColPali를 통해 RVL-CDIP 데이터셋에서 관련 문서를 검색(Retrieval)하고, Elastic Inference API를 통해 Amazon Bedrock(Claude 3.5 Sonnet 모델)의 LLM이 검색된 문서를 기반으로 자연어 응답을 생성(Generation)하는 프로세스를 보여줍니다. 이는 LLM 호출을 Elastic 생태계 내에서 직접 처리하여 외부 의존성을 최소화하고 Elastic의 기능을 최대한 활용하는 방식입니다. `03_rag_with_inference_api.ipynb` 노트북을 통해 Endpoint 생성 및 RAG 데모를 구현할 수 있으며, Streamlit 앱을 통해 시각적인 데모도 제공합니다.

## 6. RAG with ColPali using MCP Integration (Part 4)

이 섹션에서는 Model Context Protocol (MCP) 통합을 통해 ColPali 기반 문서 검색 시스템에 RAG 기능을 구현하는 방법을 설명합니다. 이 방식은 AI 에이전트(예: Claude Desktop)가 MCP 서버를 통해 Elasticsearch 데이터와 자연어로 상호작용하고, LLM이 응답을 생성하는 데 초점을 맞춥니다.

ColPali 기반 검색 시스템은 단일 모델로도 강력한 성능을 발휘하지만, 여전히 특정 문서 유형이나 쿼리에서 최적의 결과를 보장하기 어렵습니다. 이를 보완하기 위해, 이 프로젝트는 Elastic에서 제공하는 오픈소스 MCP 서버(`mcp-server-elasticsearch`)를 활용하여 다중 모델 합의 검색을 위한 프로토콜 통합을 구현합니다.

- **이유**: 단일 모델(ColPali 또는 ColQwen)은 특정 문서 유형(예: 텍스트 중심 vs. 시각적 요소 중심)이나 언어에서 성능 편차를 보일 수 있습니다. 다중 모델 합의는 ColPali 외에 다른 보완적인 모델(예: 텍스트 중심의 BERT, OCR 기반 모델 등)을 결합하여 다양한 문서와 쿼리에 대해 더 균형 잡힌 결과를 제공합니다.
- **기능**: 각 모델로부터 독립적인 검색 결과를 수집한 뒤, 사전 정의된 합의 알고리즘(예: 가중 평균, 상위 K 교집합 등)을 통해 최종 순위를 결정합니다. 이를 통해 각 모델의 강점을 최대한 활용하고 약점을 상쇄합니다.
- **장점**:
  - **검색 정확도 향상**: 다양한 모델의 결과를 종합함으로써, 단일 모델이 놓칠 수 있는 관련 문서를 포착할 가능성이 높아집니다.
  - **적응성**: 문서 유형이나 쿼리 특성에 따라 모델별 가중치를 동적으로 조정할 수 있어, 특정 상황에 최적화된 검색이 가능합니다.
  - **견고성**: 단일 모델의 실패(예: 특정 문서에서 임베딩 생성 실패)가 전체 검색 결과에 미치는 영향을 최소화합니다.

Elastic의 MCP 통합을 통해 다음을 수행할 수 있습니다:
- Claude 또는 다른 호환 가능한 MCP 모델이 자연어 요청을 기반으로 문서를 탐색할 수 있도록 합니다.
- 복잡한 코드 작성 없이 Elasticsearch에서 지식을 검색합니다.
- AI 모델에 조직 정보에 대한 직접적인 액세스를 제공합니다.
- 자체 데이터를 활용하여 더 정확하고 상황에 맞는 응답을 활성화합니다.

이 구현은 `04_rag_with_mcp_claude.ipynb` 노트북을 통해 안내됩니다.

## 7. 실행 안내: Jupyter Notebook과 Streamlit 앱

이 프로젝트는 Jupyter Notebook을 통해 인덱싱 및 검색 로직을 실행한 뒤, 생성된 인덱스를 Streamlit 앱으로 시각화하여 결과를 확인할 수 있도록 설계되었습니다. 실행 단계는 다음과 같습니다:

- **Jupyter Notebook 실행**:
  - **Part 1 (`01_colqwen.ipynb`)**: ColPali 기반의 기본 `rank_vectors` 검색을 구현하며, `colqwen3-rvlcdip-demo-part1` 인덱스를 생성합니다.
  - **Part 2 (`02_avg_colqwen.ipynb`)**: 평균 벡터와 BBQ, 리스코어링 최적화를 구현하며, `colqwen3-rvlcdip-demo-part2` 인덱스를 생성합니다.
  - **Part 3 (`03_rag_with_inference_api.ipynb`)**: Elastic Inference API를 사용한 RAG 데모를 구현하여 Amazon Bedrock과 연동된 LLM 응답 생성을 보여줍니다.
  - **Part 4 (`04_rag_with_mcp_claude.ipynb`)**: Elastic MCP 서버와 Claude Desktop을 활용한 RAG 데모를 구현하여 자연어 기반의 에이전트 상호작용을 보여줍니다.
  - 각 노트북을 순차적으로 실행하면, RVL-CDIP 샘플 데이터를 기반으로 Elasticsearch에 인덱스가 생성됩니다.

- **Streamlit App 실행**:
  - Jupyter Notebook 실행 후 생성된 인덱스를 대상으로, `app_integrated.py` 파일 또는 `colpali_rag_demo.py` 파일을 실행하여 검색 결과를 시각적으로 확인할 수 있습니다.
  - `app_integrated.py`는 ColPali 검색, Average 검색, Rescore 검색 모드를 제공합니다.
  - `colpali_rag_demo.py`는 Part 3의 RAG 데모를 시각적으로 보여주는 Streamlit 앱입니다.
  - 실행 명령어:
    ```
    streamlit run app_integrated.py
    # 또는
    streamlit run colpali_rag_demo.py
    ```
  - 앱 실행 후, 웹 브라우저에서 검색창과 예제 쿼리 버튼을 통해 다양한 쿼리를 테스트하고, 각 모드별 검색 결과를 비교할 수 있습니다.

## 8. 설정 지침

프로젝트를 로컬 환경에 설정하고 실행하려면 다음 단계를 따르십시오:

- **단계 1: 저장소 복제**:
  다음 명령어를 사용하여 프로젝트 저장소를 로컬 머신에 복제합니다:
```
git clone https://github.com/ByungjooChoi/colpali
cd colpali
```
- **단계 2: 의존성 설치**:
터미널 또는 명령 프롬프트에서 다음 명령어를 실행하여 필요한 Python 패키지를 설치합니다. Python과 pip이 설치되어 있는지 확인하십시오:
```
pip install -r requirements.txt
```
- **단계 3: Elasticsearch 자격 증명 구성**:
프로젝트 루트 디렉토리(또는 스크립트와 동일한 디렉토리)에 `elastic.env`라는 파일을 생성하고 Elasticsearch 연결 세부 정보를 추가하십시오:
```
ES_URL=<your-elasticsearch-url-or-cloud-id>
ES_API_KEY=<your-elasticsearch-api-key>
```
`<your-elasticsearch-host-or-cloud-id>` 및 `<your-elasticsearch-api-key>`를 실제 Elasticsearch 자격 증명으로 바꾸십시오. 보안상의 이유로 이 파일을 버전 관리 시스템에 커밋하지 않도록 주의하십시오.

- **단계 4: 데이터셋 준비**:
프로젝트는 RVL-CDIP 데이터셋의 샘플링된 하위 집합을 사용하며, 이는 `samples/` 디렉토리에 이미 포함되어 있습니다. 다른 데이터셋을 사용하려는 경우가 아니라면 추가적인 데이터 준비는 필요하지 않습니다.

- **단계 5: 인덱싱을 위한 Jupyter Notebook 실행**:
다음 순서대로 Jupyter Notebook을 실행하여 Elasticsearch 인덱스를 생성하십시오:
- `01_colqwen.ipynb`를 열고 모든 셀을 실행하여 Part 1 인덱스를 생성합니다.
- `02_avg_colqwen.ipynb`를 열고 모든 셀을 실행하여 Part 2 인덱스를 생성합니다.
- `03_rag_with_inference_api.ipynb`를 열고 모든 셀을 실행하여 Inference API를 사용한 RAG 데모를 설정합니다.
- `04_rag_with_mcp_claude.ipynb`를 열고 모든 셀을 실행하여 MCP 통합을 사용한 RAG 데모를 설정합니다.
JupyterLab 또는 Jupyter 확장 기능이 설치된 VS Code와 같은 도구를 사용하여 Jupyter Notebook을 실행할 수 있습니다. 다음 명령어를 사용하여 JupyterLab을 시작하십시오:
```
jupyter lab
```

- **단계 6: Streamlit 앱 실행**:
인덱싱 후, `app_integrated.py` 파일 또는 `colpali_rag_demo.py` 파일을 실행하여 검색 결과를 시각적으로 확인하십시오:
```
streamlit run app_integrated.py
또는
streamlit run colpali_rag_demo.py
```
이렇게 하면 기본 브라우저에 웹 인터페이스가 열리며, 다양한 모드에서 검색 기능을 테스트할 수 있습니다.

- **단계 7: 하드웨어 요구사항**:
최적의 성능을 위해, 특히 ColQwen 모델을 실행할 때는 GPU(NVIDIA CUDA 호환)가 권장됩니다. GPU를 사용할 수 없는 경우 시스템은 CPU로 대체되며, 이로 인해 처리 시간이 길어질 수 있습니다.

- **문제 해결**:
- Elasticsearch 인스턴스가 실행 중이며 제공된 자격 증명으로 접근 가능한지 확인하십시오.
- 의존성 충돌이 발생하는 경우, 가상 환경 사용을 고려하십시오:
  ```
  python -m venv venv
  source venv/bin/activate  # Windows: venv\Scripts\activate
  pip install -r requirements.txt
  ```
- 추가 지원을 받으려면 개별 라이브러리의 문서를 참조하거나 프로젝트 저장소에 이슈를 제기하십시오.

이 프로젝트는 기업 내부 문서 검색의 효율성과 정확도를 극대화하기 위해 최신 기술과 최적화 기법을 통합한 결과물입니다. 지속적인 피드백과 개선을 통해 더 나은 검색 시스템을 구축할 수 있도록 노력하겠습니다.
