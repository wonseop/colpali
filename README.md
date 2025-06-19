# ColPali-Based Document Search Project

This project implements a search system based on the cutting-edge Vision Language Model (VLM) ColPali to efficiently handle large-scale document retrieval within an enterprise environment. This README provides a detailed explanation of the project's background, technical choices, dataset composition, optimization techniques, MCP integration, and execution instructions.

## 1. Background and Advantages of ColPali for Enterprise Internal Document Search

In corporate environments, countless internal documents (reports, contracts, invoices, emails, etc.) are generated and stored daily. Efficiently searching these documents and quickly extracting relevant information directly impacts business productivity. Traditionally, document search has relied on OCR (Optical Character Recognition) technology, which converts scanned document images into text for keyword-based searching. However, OCR has several limitations:

- **Lack of Understanding Complex Layouts**: OCR merely extracts text and fails to comprehend the visual structure of documents, such as tables, diagrams, or section headings.
- **Quality Issues with Real Documents**: Enterprise documents are often provided as blurry or distorted scanned images, where OCR accuracy significantly drops.
- **Loss of Context**: OCR processes text as individual units, neglecting the relationships between text and visual elements within a document.

To overcome these limitations, a new approach called **ColPali** has emerged. ColPali is an innovative document search framework based on Vision Language Models (VLMs), combining ColBERT's 'late interaction' mechanism with models like PaLiGemma to holistically understand both textual and visual elements of documents. The key advantages of ColPali include:

- **Integrated Understanding of Text and Visuals**: It learns the relationships between document layouts, tables, images, and text, providing richer context.
- **Reduced Dependency on OCR**: Capable of processing document images directly without OCR, preventing information loss due to OCR errors.
- **Efficient Search**: Utilizes multi-vector representations to enable fine-grained matching between queries and documents, supporting fast retrieval even in large document collections.

## 2. Reasons and Advantages of Choosing tsystems/colqwen2.5-3b-multilingual-v1.0 Over ColPali

In this project, instead of the original ColPali model, we opted for the `tsystems/colqwen2.5-3b-multilingual-v1.0` model. This model is an extension based on Qwen2.5-VL-3B, generating multi-vector representations in the ColBERT style, and was chosen for the following reasons and advantages:

- **Multilingual Support**: `colqwen2.5-3b-multilingual-v1.0` was trained on datasets supporting multiple languages, including English and German. This aligns well with the needs of global enterprises handling documents in various languages.
- **Efficient Architecture**: Built on Qwen2.5-VL-3B with a 3B parameter scale, this model is lighter than the original ColPali, delivering high performance with fewer resources. Trained in an 8xH100 GPU environment with `bfloat16` format, it offers excellent memory efficiency.
- **Dynamic Image Resolution Handling**: Unlike fixed-resolution models, it supports dynamic resolution, limiting image patches to a maximum of 768 to balance memory usage and search quality. This is advantageous for processing documents of varying sizes and qualities.
- **Community Validation**: With over 41,000 downloads on Hugging Face, this model has been widely validated by users and researchers, ensuring reliable performance.

## 3. Reasons and Advantages of Switching to RVL-CDIP Dataset

While the original ColPali project used a custom sample dataset, this project switched to the **RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing)** dataset. The reasons and advantages for this change are as follows:

- **Large-Scale Real Document Data**: RVL-CDIP is a massive dataset comprising 400,000 grayscale scanned document images, a subset of the IIT-CDIP Test Collection. It reflects real-world enterprise document characteristics such as low resolution (100dpi), noise, and quality degradation, enhancing real-world applicability.
- **Diverse Document Categories**: RVL-CDIP consists of 16 categories (Letter, Form, Email, Handwritten, Advertisement, Scientific Report, Scientific Publication, Specification, File Folder, News Article, Budget, Invoice, Presentation, Questionnaire, Resume, Memo), covering a wide range of enterprise document types. With 25,000 images per category, it enables balanced training and testing.
- **Sampling and Data Preparation**: In this project, representative samples were extracted from RVL-CDIP, totaling 1,600 images across all categories. These samples are already prepared in the `samples/` directory, allowing immediate demo execution without additional data preprocessing. Sampling reduces the burden of handling a massive dataset while still representing diverse document types.
- **Research and Industry Standard**: RVL-CDIP is a widely used standard dataset in document image classification and retrieval research, serving as a benchmark for modern document AI models like LayoutLMv3. This enhances the reliability and comparability of project results.

## 4. Limitations of ColPali and Optimization Implementations to Overcome Them

ColPali is an innovative document search framework, but it comes with the following limitations:

- **Computational Cost for Large-Scale Document Processing**: Due to its use of multi-vector representations, indexing and searching millions of documents require significant computational resources. Generating and storing vectors for every token/patch in a document increases memory and storage demands.
- **Search Latency**: The 'late interaction' matching between multi-vectors enhances accuracy but can increase query response times in large datasets.
- **Focus on PDF and High-Resource Languages**: ColPali was primarily trained on PDF-type documents and high-resource languages (e.g., English), potentially limiting performance on other document formats or low-resource languages.

To address these limitations, this project adopts optimization techniques from the original `elasticsearch-labs` repository, implementing the following three strategies:

- **Average Vector**: To speed up large-scale searches, multi-vectors per document are compressed into a single average vector (`dense_vector`) for primary search (`knn`). This significantly reduces computational cost and search latency. (Part 2 - Step 6)
- **BBQ (Better Binary Quantization)**: To drastically reduce storage and memory usage, the `rank_vectors` field is set with `element_type: 'bit'`, quantizing 32-bit float vectors into 1-bit binary format. This saves approximately 95% of storage space and enhances cost efficiency in the rescoring phase. (Part 2 - Step 5)
- **Rescore**: For candidates retrieved via the average vector in the primary search, a secondary precise scoring is performed using BBQ-applied `rank_vectors`. This two-stage architecture ensures both the speed of the primary search and the accuracy of the secondary search. (Part 2 - Step 7)

These three optimizations achieve the goals of speed, accuracy, and cost simultaneously, effectively overcoming ColPali's limitations through a sophisticated search architecture.

## 5. Reasons, Features, and Advantages of MCP Integration

While the ColPali-based search system delivers strong performance with a single model, it still struggles to guarantee optimal results for certain document types or queries. To address this, Part 3 implements a protocol integration for multi-model consensus search. This integration leverages the open-source MCP server (`mcp-server-elasticsearch`) provided by Elastic, enabling natural language-based interaction between AI agents and Elasticsearch data. The reasons and advantages of this integration are as follows:

- **Reason**: A single model (ColPali or ColQwen) may exhibit performance variations across specific document types (e.g., text-centric vs. visually dominant) or languages. Multi-model consensus combines ColPali with complementary models (e.g., text-focused BERT, OCR-based models) to deliver more balanced results across diverse documents and queries.
- **Features**: It collects independent search results from each model and determines the final ranking through predefined consensus algorithms (e.g., weighted average, top-K intersection). This maximizes the strengths of each model while mitigating their weaknesses.
- **Advantages**:
  - **Improved Search Accuracy**: By aggregating results from multiple models, the likelihood of capturing relevant documents that a single model might miss increases.
  - **Adaptability**: Weights for each model can be dynamically adjusted based on document type or query characteristics, enabling search optimization for specific scenarios.
  - **Robustness**: The impact of a single model's failure (e.g., inability to generate embeddings for a specific document) on the overall search results is minimized.

Through Elastic’s MCP integration, you can:
- Allow Claude or other compatible MCP models to explore your documents using natural language requests.
- Access insights from your Elasticsearch data without needing to develop intricate code.
- Provide AI models with seamless access to your company's knowledge base.
- Facilitate more precise and contextually relevant answers by leveraging your proprietary data.

## 6. Execution Guide: Jupyter Notebook and Streamlit App

This project is designed to execute indexing and search logic via Jupyter Notebooks, followed by visualizing the results through a Streamlit app targeting the created indices. The execution steps are as follows:

- **Jupyter Notebook Execution**:
  - **Part 1 (`01_colqwen.ipynb`)**: Implements basic `rank_vectors` search based on ColPali, creating the `colqwen-rvlcdip-demo-part1` index.
  - **Part 2 (`02_avg_colqwen.ipynb`)**: Implements average vector, BBQ, and rescoring optimizations, creating the `colqwen-rvlcdip-demo-part2-original` index.
  - **Part 3 (`03_mcp_integration.ipynb`)**: Implements multi-model consensus search to provide optimal search results by aggregating outcomes from multiple models.
  - Running each notebook sequentially creates Elasticsearch indices based on the RVL-CDIP sample data.

- **Streamlit App Execution**:
  - After executing the Jupyter Notebooks, run the `app_integrated.py` file to visually inspect search results on the created indices.
  - The app offers three search modes (A. Colpali Search, B. Average Search, C. Rescore Search), selectable via radio buttons to perform searches.
  - Execution command:
    ```
    streamlit run app_integrated.py
    ```
  - Once the app is running, use the search bar and example query buttons in the web browser to test various queries and compare search results across modes.

## 7. Setup Instructions

To set up and run this project on your local environment, follow these steps:

- **Step 1: Clone the Repository**:
  Clone the project repository to your local machine using the following command:
```
git clone https://github.com/ByungjooChoi/colpali
cd colpali
```
- **Step 2: Install Dependencies**:
Install the required Python packages by running the following command in your terminal or command prompt. Ensure you have Python and pip installed:
```
pip install -r requirements.txt
```
- **Step 3: Configure Elasticsearch Credentials**:
Create a file named `elastic.env` in the project root directory (or in the same directory as your scripts) and add your Elasticsearch connection details:
```
ELASTIC_HOST=<your-elasticsearch-host-or-cloud-id>
ELASTIC_API_KEY=<your-elasticsearch-api-key>
```
Replace `<your-elasticsearch-host-or-cloud-id>` and `<your-elasticsearch-api-key>` with your actual Elasticsearch credentials. Ensure this file is not committed to version control for security reasons.

- **Step 4: Prepare the Dataset**:
The project uses a sampled subset of the RVL-CDIP dataset, which is already included in the `samples/` directory. No additional data preparation is required unless you wish to use a different dataset.

- **Step 5: Run Jupyter Notebooks for Indexing**:
Execute the Jupyter Notebooks in sequence to create the Elasticsearch indices:
- Open `01_colqwen.ipynb` and run all cells to create the Part 1 index.
- Open `02_avg_colqwen.ipynb` and run all cells to create the Part 2 index.
- Open `03_mcp_integration.ipynb` and run all cells to set up the MCP integration.
You can run Jupyter Notebooks using a tool like JupyterLab or VS Code with the Jupyter extension installed. Start JupyterLab with:
```
jupyter lab
```

- **Step 6: Run the Streamlit App**:
After indexing, launch the Streamlit app to interact with the search system:
```
streamlit run app_integrated.py
```
This will open a web interface in your default browser where you can test the search functionalities across different modes.

- **Step 7: Hardware Requirements**:
For optimal performance, especially when running the ColQwen model, a GPU (NVIDIA CUDA-compatible) is recommended. If a GPU is not available, the system will fall back to CPU, which may result in slower processing times.

- **Troubleshooting**:
- Ensure your Elasticsearch instance is running and accessible with the provided credentials.
- If you encounter dependency conflicts, consider using a virtual environment:
  ```
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip install -r requirements.txt
  ```
- For additional support, refer to the documentation of individual libraries or raise an issue in the project repository.

This project integrates the latest technologies and optimization techniques to maximize the efficiency and accuracy of enterprise internal document search. We strive to build a better search system through continuous feedback and improvement.

