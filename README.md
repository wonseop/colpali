# ColPali-based Document Search Project

This project implements an advanced document search system based on the state-of-the-art Vision Language Model (VLM), ColPali, to efficiently handle large-scale enterprise document retrieval. This README provides a detailed explanation of the project's background, technology choices, dataset configuration, optimization techniques, RAG (Retrieval-Augmented Generation) implementation, and execution instructions.

## 1. Background: Enterprise Document Search and the Rise of ColPali

In enterprise environments, a vast number of internal documents (reports, contracts, invoices, emails, etc.) are created and stored daily. Efficiently retrieving and extracting relevant information from these documents has a direct impact on productivity. Traditional document search has relied on Optical Character Recognition (OCR), which merely extracts text but cannot understand visual structures like tables, diagrams, or section headings.

Real-world enterprise documents are often low-quality scanned images that are blurry or distorted, leading to reduced OCR accuracy. Moreover, OCR processes text as isolated units, ignoring the relationships between text and visual elements within documents.

To overcome these limitations, **ColPali** has emerged as a novel approach. ColPali is a cutting-edge document search framework that combines the 'late interaction' mechanism of ColBERT with VLMs like PaLiGemma, enabling integrated understanding of both textual and visual components. Key advantages of ColPali include:

- **Integrated understanding of text and visual elements**: Learns relationships between layout, tables, images, and text, providing richer context.
- **Reduced dependency on OCR**: Processes document images directly, minimizing information loss due to OCR errors.
- **Efficient search**: Multi-vector representations allow fine-grained matching between queries and documents, supporting fast retrieval at scale.

## 2. Why We Chose `tsystems/colqwen2.5-3b-multilingual-v1.0` Instead of ColPali

This project uses the `tsystems/colqwen2.5-3b-multilingual-v1.0` model instead of the original ColPali. Based on Qwen2.5-VL-3B, it extends ColBERT-style multi-vector representation with the following benefits:

- **Multilingual support**: Trained on datasets covering multiple languages including English and German, making it suitable for global enterprises.
- **Efficient architecture**: With only 3B parameters, it's lighter than ColPali and delivers strong performance with fewer resources. It is trained on 8xH100 GPUs using `bfloat16`, optimizing memory usage.
- **Dynamic image resolution handling**: Supports dynamic input resolution and limits the number of image patches to 768 for a balance of quality and efficiency.
- **Community-proven**: Downloaded over 41,000 times from Hugging Face, the model is validated by a broad user base.

## 3. Why We Switched to RVL-CDIP Dataset

Instead of the original ColPali demo data, this project uses the **RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing)** dataset, with key benefits:

- **Large-scale real-world document data**: RVL-CDIP includes 400,000 grayscale scanned document images (100dpi), representing noisy, low-resolution, real-world documents.
- **Diverse document categories**: Contains 16 categories (e.g., Letter, Form, Email, Handwritten, Invoice, Report, Resume, etc.) with 25,000 images per class.
- **Representative sample set**: We selected 1,600 images (100 from each category), located under `samples/`, ready for demo without preprocessing.
- **Industry benchmark**: RVL-CDIP is widely used in document AI research (e.g., LayoutLMv3), improving reliability and comparability.

## 4. ColPali’s Limitations and Optimization Techniques

While ColPali is powerful, it has known limitations:

- **Computational cost**: Multi-vector indexing requires significant memory and storage.
- **Latency**: Late interaction matching improves accuracy but increases query response time.
- **PDF-centric bias**: Primarily trained on PDFs and high-resource languages like English.

To address this, we apply optimizations inspired by the original `elasticsearch-labs` project:

- **Average Vector**: Uses a single `dense_vector` per document for first-pass `knn` search, reducing search cost. (Part 2 - Step 6)
- **BBQ (Better Binary Quantization)**: Compresses vectors in `rank_vectors` using `element_type: 'bit'`, saving \~95% storage. (Part 2 - Step 5)
- **Rescore**: Re-ranks top candidates using the BBQ vectors, combining speed and precision in a two-stage architecture. (Part 2 - Step 7)

These strategies achieve an optimal balance between speed, accuracy, and cost.

## 5. RAG with ColPali using Inference API (Part 3)

This section explains how to integrate RAG into ColPali search using Elastic's Inference API, specifically leveraging Amazon Bedrock's Claude 3.5 Sonnet LLM. The process:

- Retrieve relevant documents from RVL-CDIP using ColPali
- Use Elastic Inference API to call Claude LLM (via Amazon Bedrock)
- Claude generates natural language answers based on the retrieved content

This is demonstrated in `03_rag_with_inference_api.ipynb` and visualized via Streamlit apps.

## 6. RAG with ColPali using MCP Integration (Part 4)

This section covers RAG via **Model Context Protocol (MCP)**, enabling LLM agents (like Claude Desktop) to interact with Elasticsearch through natural language.

- **Why MCP?**: Single-model systems (ColPali or ColQwen) may underperform for specific document types or languages. A multi-model consensus approach increases robustness and relevance.
- **How?**: Each model contributes results, and a defined consensus (e.g., weighted average, top-K intersection) determines final ranking.

### Advantages:

- **Improved accuracy** via result aggregation
- **Adaptability** by adjusting model weights based on document/query characteristics
- **Resilience** against individual model failures

The setup is demonstrated in `04_rag_with_mcp_claude.ipynb`.

## 7. Running the Project: Jupyter Notebooks & Streamlit

Use Jupyter Notebooks for indexing and search logic, and Streamlit apps for UI-based exploration.

### Jupyter Notebooks:

- **Part 1 (**\`\`**)**: Builds ColPali index using `rank_vectors`
- **Part 2 (**\`\`**)**: Adds average vector, BBQ, and rescore optimizations
- **Part 3 (**\`\`**)**: Sets up RAG using Elastic Inference API
- **Part 4 (**\`\`**)**: RAG with Claude via MCP server

### Streamlit Apps:

- \`\`: Compares ColPali, Avg, and Rescore modes
- \`\`: Visual RAG demo (Part 3)

```bash
streamlit run app_integrated.py
# or
streamlit run colpali_rag_demo.py
```

## 8. Setup Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/ByungjooChoi/colpali
cd colpali
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure Elasticsearch Credentials

Create an `elastic.env` file with the following content:

```bash
ELASTIC_HOST=<your-elasticsearch-host-or-cloud-id>
ELASTIC_API_KEY=<your-elasticsearch-api-key>
```

### Step 4: Prepare Dataset

The `samples/` folder includes 1,600 sample images. No extra setup required.

### Step 5: Run Jupyter Notebooks in Order

```bash
jupyter lab
```

Execute:

- `01_colqwen.ipynb`
- `02_avg_colqwen.ipynb`
- `03_rag_with_inference_api.ipynb`
- `04_rag_with_mcp_claude.ipynb`

### Step 6: Run Streamlit Apps

```bash
streamlit run app_integrated.py
# or
streamlit run colpali_rag_demo.py
```

### Step 7: Hardware Requirements

GPU (NVIDIA CUDA) is recommended. CPU fallback is available but slower.

### Troubleshooting

- Verify Elasticsearch access with provided credentials
- Use a virtual environment if dependency conflicts occur:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

This project integrates state-of-the-art search and optimization techniques to build a high-performance enterprise document retrieval system. Continuous improvement and feedback are welcomed.


