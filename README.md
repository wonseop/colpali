# ColPali-based Multimodal Document Search Project

This project implements an advanced document search system based on the state-of-the-art Vision Language Model (VLM), ColPali, to efficiently handle large-scale enterprise document retrieval. This document provides a detailed explanation of the project's background, technology choices, dataset configuration, optimization techniques, and the step-by-step Retrieval-Augmented Generation (RAG) implementation architecture.

### 1. Background: Enterprise Document Search and the Rise of ColPali

In enterprise environments, a vast number of internal documents (reports, contracts, invoices, etc.) are created and stored daily. Efficiently retrieving and extracting relevant information from these documents directly impacts productivity. Traditional document search has relied on Optical Character Recognition (OCR), which merely extracts text but cannot understand visual structures like tables, diagrams, or contextual layouts.

ColPali has emerged as a novel approach to overcome these limitations. It combines the 'late interaction' mechanism of ColBERT with a VLM, enabling an integrated understanding of both textual and visual components. By processing document images directly, it minimizes information loss from OCR errors.

### 2. Why We Chose `tsystems/colqwen2.5-3b-multilingual-v1.0`

While building on the core ideas of ColPali, this project uses the `tsystems/colqwen2.5-3b-multilingual-v1.0` model. Based on Qwen2.5-VL-3B, it implements a ColBERT-style multi-vector representation and offers the following advantages:

* **State-of-the-Art Performance**: Achieved the #1 rank on both the original Vidore benchmark and its successor, Vidore v2, for visual document retrieval, proving its top-tier capabilities in understanding and matching complex documents.
* **Multilingual Support**: Trained on datasets covering multiple languages, including English and German, making it suitable for global enterprises.
* **Efficient Architecture**: With 3B parameters, it is lighter than the original ColPali, delivering strong performance with fewer resources.
* **Community-Proven**: Widely used on Hugging Face, its performance and stability have been validated by a broad user base.

### 3. Why We Use the RVL-CDIP Dataset

Instead of the original demo data, this project uses the **RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing)** dataset, which better represents real-world scenarios.

* **Large-Scale Real-World Data**: Contains 400,000 low-resolution scanned document images, reflecting the noisy conditions of a real enterprise environment.
* **Diverse Document Categories**: Includes 16 categories, such as letters, emails, invoices, and handwritten notes.
* **Representative Sample Set**: The `samples/` folder contains 1,600 images (100 from each of the 16 categories), ready for use without any preprocessing.

### 4. ColPali’s Limitations and Optimization Techniques

While powerful, ColPali has trade-offs, such as high memory usage for multi-vector indexing and increased latency due to its 'late interaction' mechanism. To address this, we apply the following optimization techniques:

* **Average Vector**: A single `dense_vector` is generated for each document to be used in a first-pass `knn` search, significantly reducing search costs.
* **BBQ (Better Binary Quantization)**: The multi-vectors in the rank_vectors field are compressed using the element_type: 'bit' option, reducing storage by around 95%, and also reducing memory usage and improving search speed.
* **Rescore**: After the initial `knn` search, the top candidates are re-ranked using the original, BBQ-compressed multi-vectors. This two-stage architecture achieves both speed and accuracy.

### 5. Multimodal RAG: Direct Integration with Amazon Bedrock (Part 3)

One of the core features of this project is a RAG pipeline that directly combines Colpali's sophisticated visual search with Amazon Bedrock's multimodal LLMs. This is implemented in the `03_colpali_rag.ipynb` notebook and the `colpali_rag_demo.py` Streamlit app.

### 6. Advanced RAG: MCP and Claude Desktop Integration (Part 4)

The next phase of the project involves building a system where an LLM agent, such as Claude Desktop, can interact with our Elasticsearch-based document store in natural language via the **MCP (Model Context Protocol)**.

* **Architecture**: The `colpali-mcp-server` runs on a remote server (e.g., EC2) as a `streamable-http` server built with the official MCP Python SDK. On the user's local machine, Claude Desktop runs the official `mcp-proxy`, which connects to the remote server. This allows Claude to use the custom `search` tool to retrieve document images as context for its own internal RAG process.
* **Advantages**: This **agent-based interaction** enables users to perform complex, sequential tasks in natural language, such as "Find documents related to financial reports, then search within them for the Q3 earnings."

The setup process for this is detailed in the `04_rag_with_colpali_mcp.ipynb` notebook.

### 7. How to Run the Project: Jupyter Notebooks & Streamlit

The project consists of Jupyter Notebooks for data indexing and logic validation, and Streamlit apps for interactive demos.

* **Jupyter Notebooks**:
    * `01_colqwen.ipynb`: Builds the basic Colpali index.
    * `02_avg_colqwen.ipynb`: Creates an optimized index.
    * `03_colpali_rag.ipynb`: Tests the multimodal RAG architecture using `boto3`.
    * `04_rag_with_colpali_mcp.ipynb`: Guides through setting up the MCP server and Claude Desktop integration.

* **Streamlit Apps**:
    * `app_part1.py`, `app_integrated.py`, `colpali_rag_demo.py`: Various demo applications.

### 8. Setup and Execution Guide

**Step 1: Clone the Repository**
```bash
git clone https://github.com/ByungjooChoi/colpali
cd colpali
```

**Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 3: Configure Credentials**
Create `elastic.env` and `aws.env` files in the project root directory with your credentials.

**Step 4: Run Jupyter Notebooks & MCP Server**
1.  Run `jupyter lab` and execute notebooks `01` through `03` to index data.
2.  On your **remote server (e.g., EC2)**, build and run the MCP server:
    ```bash
    # Build the Docker image
    docker build -t colpali-mcp-server ./colpali_mcp_server

    # Run the container with your environment variables
    docker run -d --rm --name colpali-mcp-server -p 8000:8000 --gpus all --env-file ./elastic.env colpali-mcp-server
    ```
3.  On your **local machine**, install the official MCP proxy:
    ```bash
    pip install mcp-proxy
    ```
4.  Run the `04_rag_with_colpali_mcp.ipynb` notebook locally to generate your Claude Desktop configuration.

**Step 5: Run Streamlit Demo Apps**
Run one of the `streamlit run <app_name>.py` commands.

### 9. Troubleshooting

* **Credential Errors**: Ensure your `.env` files are correct and the keys have necessary permissions.
* **Dependency Conflicts**: Use a Python virtual environment.

---
This project demonstrates how to build a next-generation enterprise document search system by integrating state-of-the-art visual search and multimodal generative AI.
