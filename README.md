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

One of the core features of this project is a RAG pipeline that directly combines Colpali's sophisticated visual search with Amazon Bedrock's multimodal LLMs.

**How it works:**

1.  **Retrieval**: A user's text query is converted into a vector by the Colpali model to find the most visually similar documents in Elasticsearch.
2.  **Image Context Preparation**: The path of the top-ranked image from the search results is retrieved and the image is encoded into a Base64 string.
3.  **Generation**: The AWS SDK (`boto3`) is used to call the Amazon Bedrock API directly. A **multimodal prompt**, containing both the user's original text query and the Base64-encoded image, is sent to the model.
4.  **Answer**: A multimodal LLM like Claude 3 **directly "sees" and analyzes** the image content to generate a specific and accurate answer to the user's question.

This approach maximizes the quality of the RAG system's responses by allowing the LLM to analyze visual information directly. This is implemented in the `03_rag_interactive_boto3.ipynb` notebook and the `colpali_rag_demo.py` Streamlit app.

### 6. Advanced RAG: MCP and Claude Desktop Integration (Part 4)

The next phase of the project involves building a system where an LLM agent, such as Claude Desktop, can interact with Elasticsearch in natural language via the **MCP (Model Context Protocol)**.

* **What is MCP?**: A protocol designed to allow LLM agents to interact with external tools and data sources. Elastic provides an MCP server for Elasticsearch, enabling an LLM like Claude to directly query indices, inspect mappings, and perform searches.
* **How it works**: A Node.js-based `mcp-server-elasticsearch` runs in the terminal, acting as a bridge between Claude Desktop and Elasticsearch. When a user asks a question in Claude Desktop, Claude sends a request to the MCP server, which then queries Elasticsearch and returns the results.
* **Advantages**: This **agent-based interaction** goes beyond simple Q&A, allowing users to perform complex, sequential tasks in natural language, such as "List all available indices" or "Show me the mapping for this index."

The setup process for this is detailed in the `04_rag_with_mcp_claude.ipynb` notebook.

### 7. How to Run the Project: Jupyter Notebooks & Streamlit

The project consists of Jupyter Notebooks for data indexing and logic validation, and Streamlit apps for interactive demos.

* **Jupyter Notebooks**:
    * `01_colqwen.ipynb`: Builds the basic Colpali index using `rank_vectors`.
    * `02_avg_colqwen.ipynb`: Creates an optimized index with average vectors, BBQ, and rescoring.
    * `03_rag_interactive_boto3.ipynb`: Tests the multimodal RAG architecture using `boto3`.
    * `04_rag_with_mcp_claude.ipynb`: Guides through setting up the MCP server and Claude Desktop integration.

* **Streamlit Apps**:
    * `app_part1.py`: A basic demo for the Part 1 `rank_vectors` search.
    * `app_integrated.py`: An integrated demo comparing the Part 1 search with the Part 2 optimization techniques (average vector, rescore).
    * `colpali_rag_demo.py`: The final demo application showcasing the multimodal RAG pipeline from Part 3.

### 8. Setup and Execution Guide

**Step 1: Clone the Repository**
```bash
git clone [https://github.com/ByungjooChoi/colpali](https://github.com/ByungjooChoi/colpali)
cd colpali
```

**Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```
*(Note: Ensure `boto3` is included in your `requirements.txt` file.)*

**Step 3: Configure Credentials**
Create the following two `.env` files in the project root directory.

* `elastic.env`:
    ```
    ELASTIC_HOST=<your-elasticsearch-host-or-cloud-id>
    ELASTIC_API_KEY=<your-elasticsearch-api-key>
    ```
* `aws.env`:
    ```
    AWS_ACCESS_KEY=<your-aws-access-key>
    AWS_SECRET_KEY=<your-aws-secret-key>
    AWS_REGION=<your-aws-region>
    ```

**Step 4: Run Jupyter Notebooks**
Run `jupyter lab` in your terminal, then execute the notebooks in the following order to index the data and verify each step:
1.  `01_colqwen.ipynb`
2.  `02_avg_colqwen.ipynb`
3.  `03_rag_interactive_boto3.ipynb`
4.  `04_rag_with_mcp_claude.ipynb`

**Step 5: Run Streamlit Demo Apps**
Run one of the following commands in your terminal:
```bash
# Basic search demo for Part 1
streamlit run app_part1.py

# Integrated comparison demo for Part 1 & 2
streamlit run app_integrated.py

# Final multimodal RAG demo for Part 3
streamlit run colpali_rag_demo.py
```

### 9. Troubleshooting

* **Credential Errors**: Verify that the information in your `.env` files is correct and that the keys have the necessary permissions. For Bedrock, ensure that you have enabled model access for the desired model (e.g., Claude 3 Haiku) in the AWS Console.
* **Dependency Conflicts**: If you encounter library version conflicts, it is highly recommended to use a Python virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

---
This project demonstrates how to build a next-generation enterprise document search system that overcomes the limitations of traditional methods by integrating state-of-the-art visual search and multimodal generative AI.
