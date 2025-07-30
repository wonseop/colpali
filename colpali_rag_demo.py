import os
# This line is for environments where file watching can cause issues.
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
import torch
import numpy as np
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from PIL import Image
import boto3
import base64
import json
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

# --- 1. Page Configuration and Constants ---
st.set_page_config(
    page_title="Colpali Multimodal RAG Demo",
    page_icon="🖼️",
    layout="wide"
)

# --- Constants for Elasticsearch, Models, and Bedrock ---
INDEX_NAME_PART1 = "colqwen-rvlcdip-demo-part1"
VECTOR_FIELD_NAME_PART1 = "colqwen_vectors"
INDEX_NAME_PART2 = "colqwen-rvlcdip-demo-part2-original"
VECTOR_FIELD_NAME_PART2_MULTI = "colqwen_vectors_binary" 
AVG_VECTOR_FIELD_NAME_PART2 = "colqwen_avg_vector"
MODEL_NAME = "tsystems/colqwen2.5-3b-multilingual-v1.0"
# FIX: Switched to Claude 3 Haiku, which is more likely to be available for On-Demand throughput.
BEDROCK_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0" 

# --- 2. Caching and Helper Functions ---

if 'query_text' not in st.session_state:
    st.session_state.query_text = ""

@st.cache_resource
def load_colqwen_model():
    """Loads and caches the ColQwen model and processor."""
    st.info("Loading ColQwen model... This may take a moment on first run.")
    device_map = "cpu"
    if torch.backends.mps.is_available():
        device_map = "mps"
    elif torch.cuda.is_available():
        device_map = "cuda:0"
    
    model = ColQwen2_5.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if device_map != "cpu" else torch.float32,
        device_map=device_map
    ).eval()
    processor = ColQwen2_5_Processor.from_pretrained(MODEL_NAME, use_fast=True)
    st.success("Model loaded successfully.")
    return model, processor

@st.cache_resource
def connect_to_elasticsearch(elastic_host, api_key):
    """Connects to Elasticsearch and caches the client."""
    if ":" in elastic_host and not elastic_host.startswith("http"):
        es = Elasticsearch(cloud_id=elastic_host, api_key=api_key)
    else:
        es = Elasticsearch(hosts=[elastic_host], api_key=api_key)
    return es

@st.cache_resource
def connect_to_bedrock(aws_access_key, aws_secret_key, region_name):
    """Connects to Bedrock and caches the client."""
    st.info(f"Connecting to Amazon Bedrock in {region_name}...")
    bedrock_client = boto3.client(
        "bedrock-runtime",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region_name,
    )
    st.success("Connected to Bedrock successfully.")
    return bedrock_client

def create_colqwen_query_vectors(query_text, model, processor):
    """Creates multi-vector embeddings for a text query."""
    inputs = processor.process_queries([query_text]).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.cpu().to(torch.float32).numpy().tolist()[0]

def to_avg_vector(vectors):
    """Calculates a single, normalized average vector from multi-vectors."""
    vectors_array = np.array(vectors)
    avg_vector = np.mean(vectors_array, axis=0)
    norm = np.linalg.norm(avg_vector)
    return (avg_vector / norm).tolist() if norm > 0 else avg_vector.tolist()

def display_results(hits):
    """Displays visual search results in columns."""
    cols = st.columns(5)
    for i, hit in enumerate(hits):
        with cols[i]:
            st.markdown(f"**Rank {i+1}**")
            score = hit["_score"]
            path = hit["_source"]["image_path"]
            category = hit["_source"]["category"]
            if os.path.exists(path):
                image = Image.open(path)
                st.image(image, caption=f"Category: {category}", use_container_width=True)
            else:
                st.warning(f"Image not found at: {path}")
            st.metric(label="Score", value=f"{score:.4f}")
            st.caption(f"ID: {hit['_id'][:20]}...")

def get_media_type(image_path):
    """Determines the media type from the file extension."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".jpeg", ".jpg"]:
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    elif ext == ".webp":
        return "image/webp"
    else:
        return "image/jpeg" # Default to JPEG

def generate_llm_answer_with_image(bedrock_client, query_text, hits):
    """Encodes the top image, creates a multimodal prompt, and generates an answer using Bedrock."""
    if not hits:
        return "No documents were found by the search engine, so I cannot generate an answer."

    # Use the top search result's image
    top_hit = hits[0]
    image_path = top_hit["_source"].get("image_path")

    if not image_path or not os.path.exists(image_path):
        return "The top search result did not have a valid image path, so I cannot analyze it."

    st.info(f"Analyzing top image for context: `{image_path}`")

    # Encode the image to Base64
    try:
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        return f"Error reading or encoding the image file: {e}"

    # Construct the multimodal request payload for Claude 3
    bedrock_request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": get_media_type(image_path),
                            "data": image_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"You are an expert document analysis AI. Analyze this document image carefully and answer the following question based only on what you see: '{query_text}'"
                    }
                ],
            }
        ],
    }

    # Call the Bedrock API
    try:
        response = bedrock_client.invoke_model(
            body=json.dumps(bedrock_request_body),
            modelId=BEDROCK_MODEL_ID,
            contentType="application/json",
            accept="application/json",
        )
        response_body = json.loads(response.get("body").read())
        
        # Extract the text content from the response
        result = response_body.get("content", [{}])[0].get("text", "No response generated.")
        return result
    except Exception as e:
        st.error(f"Error calling Bedrock API: {e}")
        return f"An error occurred while communicating with the LLM: {e}"


# --- 3. Main App Logic ---

st.title("Colpali Multimodal RAG Search Demo")
st.markdown(
    "This app integrates **Colpali visual search** with a **multimodal LLM (Claude 3 Haiku)**. "
    "The system first finds relevant images and then sends the top image to the LLM for direct analysis and answering."
)

# --- Initialization and Connection ---
try:
    load_dotenv('elastic.env')
    load_dotenv('aws.env', override=True)
    
    ELASTIC_HOST = os.getenv("ELASTIC_HOST")
    ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY")
    AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY") 
    AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY") 
    AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")

    if not all([ELASTIC_HOST, ELASTIC_API_KEY, AWS_ACCESS_KEY, AWS_SECRET_KEY]):
        st.error("🚨 Credentials for Elasticsearch or AWS are missing. Please check your `.env` files.")
        st.stop()
        
    model, processor = load_colqwen_model()
    es = connect_to_elasticsearch(ELASTIC_HOST, ELASTIC_API_KEY)
    bedrock = connect_to_bedrock(AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION)
except Exception as e:
    st.error(f"Failed to initialize. Error: {e}")
    st.stop()

# --- Search UI ---
st.header("Search Interface")

def set_query_text(text):
    st.session_state.query_text = text

st.subheader("Example Queries")
col1, col2 = st.columns(2)
with col1:
    st.button(
        "Find the return request from the customer", 
        on_click=set_query_text, 
        args=("Find the return request from the customer",),
        use_container_width=True
    )
with col2:
    st.button(
        "고객사에서 보낸 반품요청서를 찾아줘", 
        on_click=set_query_text, 
        args=("고객사에서 보낸 반품요청서를 찾아줘",),
        use_container_width=True
    )

# --- Search Form ---
with st.form(key='search_form'):
    search_query = st.text_input(
        "Enter your search query or click an example above:", 
        value=st.session_state.query_text,
        key='search_input'
    )
    search_mode = st.radio(
        "Select Search Mode:",
        ["A. Colpali(colqwen) RAG search (Part 1)", "B. Average RAG search (Part 2 - KNN Only)", "C. Rescore RAG search (Part 2 - KNN + Rescore)"],
        index=0,
        horizontal=True
    )
    submitted = st.form_submit_button("Search and Generate Answer")

# --- Form Submission Logic ---
if submitted:
    st.session_state.query_text = search_query
    if not st.session_state.query_text:
        st.warning("Please enter a query to search.")
    else:
        es_query_body = None
        index_name = ""
        
        with st.spinner("Step 1: Performing visual search with Colpali..."):
            try:
                query_vectors_float = create_colqwen_query_vectors(st.session_state.query_text, model, processor)
                
                if search_mode.startswith("A."):
                    index_name = INDEX_NAME_PART1
                    es_query_body = {
                        "size": 5,
                        "query": {"script_score": {"query": {"match_all": {}}, "script": {"source": f"maxSimDotProduct(params.query_vector, '{VECTOR_FIELD_NAME_PART1}')", "params": {"query_vector": query_vectors_float}}}},
                        "_source": ["image_path", "category"]
                    }
                
                elif search_mode.startswith("B."):
                    index_name = INDEX_NAME_PART2
                    query_avg_vector = to_avg_vector(query_vectors_float)
                    es_query_body = {
                        "size": 5,
                        "knn": {"field": AVG_VECTOR_FIELD_NAME_PART2, "query_vector": query_avg_vector, "k": 10, "num_candidates": 100},
                        "_source": ["image_path", "category"]
                    }
                
                else: 
                    index_name = INDEX_NAME_PART2
                    query_avg_vector = to_avg_vector(query_vectors_float)
                    knn_query = {"field": AVG_VECTOR_FIELD_NAME_PART2, "query_vector": query_avg_vector, "k": 10, "num_candidates": 100}
                    rescore_definition = {"window_size": 10, "query": {"rescore_query": {"script_score": {"query": {"match_all": {}}, "script": {"source": f"maxSimDotProduct(params.query_vector, '{VECTOR_FIELD_NAME_PART2_MULTI}')", "params": {"query_vector": query_vectors_float}}}}, "query_weight": 0.0, "rescore_query_weight": 1.0}}
                    es_query_body = {"size": 5, "knn": knn_query, "rescore": rescore_definition, "_source": ["image_path", "category"]}

                response = es.search(index=index_name, body=es_query_body)
                hits = response["hits"]["hits"]
                
                st.subheader("Visual Search Results")
                if not hits:
                    st.warning("No matching documents found.")
                else:
                    display_results(hits)
                
                st.subheader("LLM Generated Answer from Top Image")
                with st.spinner("Step 2: Analyzing image and generating answer with Bedrock..."):
                    llm_answer = generate_llm_answer_with_image(bedrock, st.session_state.query_text, hits)
                    st.markdown(llm_answer)

            except Exception as e:
                st.error(f"An error occurred during the process: {e}")

