import os
# This line is for environments where file watching can cause issues.
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
import torch
import numpy as np
import time
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from PIL import Image
import base64
import json
import requests
import boto3
from colpali_engine.models import ColQwen3, ColQwen3Processor
from transformers import AutoConfig

# --- 1. Page Configuration and Constants ---
st.set_page_config(
    page_title="Colpali Multimodal RAG Demo",
    page_icon="🖼️",
    layout="wide"
)

# --- Constants for Elasticsearch, Models, OpenAI and Bedrock ---
INDEX_NAME_PART1 = "colqwen3-rvlcdip-demo-part1"
VECTOR_FIELD_NAME_PART1 = "colqwen_vectors"
INDEX_NAME_PART2 = "colqwen3-rvlcdip-demo-part2" # Corrected for consistency
VECTOR_FIELD_NAME_PART2_MULTI = "colqwen_vectors" # Corrected for consistency
AVG_VECTOR_FIELD_NAME_PART2 = "colqwen_avg_vector"
MODEL_NAME = "TomoroAI/tomoro-colqwen3-embed-4b"
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
    
    # Load config first and fix rope_scaling if None
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if hasattr(config, 'text_config') and config.text_config.rope_scaling is None:
        config.text_config.rope_scaling = {"mrope_section": [24, 20, 20], "type": "default"}

    # Try flash_attention_2 first, fall back to sdpa if not available
    attn_impl = "flash_attention_2" if device_map != "cpu" else "eager"
    try:
        import flash_attn
    except ImportError:
        attn_impl = "sdpa" if device_map != "cpu" else "eager"
        print(f"flash-attn not installed, using '{attn_impl}' attention instead.")

    model = ColQwen3.from_pretrained(
        MODEL_NAME,
        config=config,
        torch_dtype=torch.bfloat16 if device_map != "cpu" else torch.float32,
        device_map=device_map,
        attn_implementation=attn_impl,
        trust_remote_code=True
    ).eval()

    processor = ColQwen3Processor.from_pretrained(MODEL_NAME, use_fast=True)
    st.success("Model loaded successfully.")
    return model, processor

@st.cache_resource
def connect_to_elasticsearch(elastic_host, api_key):
    """Connects to Elasticsearch and caches the client."""
    if ":" in elastic_host and not elastic_host.startswith("http"):
        es = Elasticsearch(cloud_id=elastic_host, api_key=api_key, request_timeout=30)
    else:
        es = Elasticsearch(hosts=[elastic_host], api_key=api_key, request_timeout=30, verify_certs=False, ssl_show_warn=False)
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

SYSTEM_PROMPT = """You are an intelligent document analysis assistant. Your task is to analyze a document image that a search system has retrieved in response to a user's query. Your response should be structured as follows:

1.  **Relevance Assessment**: Start by explaining how relevant the document is to the user's query.
2.  **Summary**: Provide a concise summary of the document's key information.
3.  **Direct Answer**: Directly answer the user's query based on the document's content.
4.  **Contextual Explanation**: If the document is not a perfect match for the query, explain why it was likely the most relevant result found. You can describe the relationship between the user's query terms and the document's content. For instance, you could say, "While this document does not specifically mention [a key term from the query], it discusses [a related topic in the document], making it the most relevant document found."
"""

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
    if not hits:
        st.warning("No matching documents found.")
        return
        
    cols = st.columns(5)
    for i, hit in enumerate(hits):
        with cols[i]:
            st.markdown(f"**Rank {i+1}**")
            score = hit["_score"]
            path = hit["_source"]["image_path"]
            category = hit["_source"]["category"]
            if os.path.exists(path):
                image = Image.open(path)
                st.image(image, caption=f"Category: {category}", width='stretch')
            else:
                st.warning(f"Image not found at: {path}")
            st.metric(label="Score", value=f"{score:.4f}")
            st.caption(f"ID: {hit['_id'][:20]}...")

def build_es_query(search_mode, query_vectors_float, query_avg_vector):
    """Builds the Elasticsearch query body based on the selected search mode."""
    index_name = ""
    es_query_body = {
        "size": 5,
        "_source": ["image_path", "category"]
    }

    if search_mode.startswith("A."):
        index_name = INDEX_NAME_PART1
        es_query_body["query"] = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": f"maxSimDotProduct(params.query_vector, '{VECTOR_FIELD_NAME_PART1}')",
                    "params": {"query_vector": query_vectors_float}
                }
            }
        }
    elif search_mode.startswith("B."):
        index_name = INDEX_NAME_PART2
        es_query_body["knn"] = {
            "field": AVG_VECTOR_FIELD_NAME_PART2,
            "query_vector": query_avg_vector,
            "k": 200,
            "num_candidates": 500
        }
    else: # C. Rescore RAG search
        index_name = INDEX_NAME_PART2
        es_query_body.update({
            "knn": {
                "field": AVG_VECTOR_FIELD_NAME_PART2,
                "query_vector": query_avg_vector,
                "k": 200,
                "num_candidates": 500
            },
            "rescore": {
                "window_size": 50,
                "query": {
                    "rescore_query": {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": f"maxSimDotProduct(params.query_vector, '{VECTOR_FIELD_NAME_PART2_MULTI}')",
                                "params": {"query_vector": query_vectors_float}
                            }
                        }
                    },
                    "query_weight": 0.0,
                    "rescore_query_weight": 1.0
                }
            }
        })
        
    return index_name, es_query_body

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

def generate_with_openai(openai_base_url, openai_api_key, openai_model, query_text, image_base64, image_path):
    """Generate answer using OpenAI Compatible API."""
    media_type = get_media_type(image_path)
    
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }
    
    openai_request = {
        "model": openai_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": f"User Query: '{query_text}'\n\nPlease analyze the provided document image."
                    }
                ]
            }
        ],
        "max_tokens": 2048
    }
    
    response = requests.post(
        f"{openai_base_url}/chat/completions",
        headers=headers,
        json=openai_request,
        timeout=120
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def generate_with_bedrock(bedrock_client, query_text, image_base64, image_path):
    """Generate answer using AWS Bedrock."""
    bedrock_request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "system": SYSTEM_PROMPT,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": get_media_type(image_path), "data": image_base64}},
                {"type": "text", "text": f"User Query: '{query_text}'\n\nPlease analyze the provided document image."}
            ]
        }]
    }
    response = bedrock_client.invoke_model(
        body=json.dumps(bedrock_request_body),
        modelId=BEDROCK_MODEL_ID,
        contentType="application/json",
        accept="application/json"
    )
    response_body = json.loads(response.get("body").read())
    return response_body.get("content", [{}])[0].get("text", "No response generated.")

def generate_llm_answer_with_image(llm_backend, query_text, hits, **backend_config):
    """Encodes the top image and generates an answer using the selected LLM backend."""
    if not hits:
        return "No documents were found by the search engine, so I cannot generate an answer."

    # Use the top search result's image
    top_hit = hits[0]
    image_path = top_hit["_source"].get("image_path")

    if not image_path or not os.path.exists(image_path):
        return "The top search result did not have a valid image path, so I cannot analyze it."

    st.info(f"Analyzing top image with {llm_backend}: `{image_path}`")

    # Encode the image to Base64
    try:
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        return f"Error reading or encoding the image file: {e}"

    # Call the appropriate LLM backend
    try:
        if llm_backend == "OpenAI":
            result = generate_with_openai(
                backend_config['openai_base_url'],
                backend_config['openai_api_key'],
                backend_config['openai_model'],
                query_text,
                image_base64,
                image_path
            )
        elif llm_backend == "AWS Bedrock":
            result = generate_with_bedrock(
                backend_config['bedrock_client'],
                query_text,
                image_base64,
                image_path
            )
        else:
            result = "Unknown LLM backend selected."
        return result
    except Exception as e:
        st.error(f"Error calling {llm_backend}: {e}")
        return f"An error occurred while communicating with the LLM: {e}"


# --- 3. Main App Logic ---

st.title("Colpali Multimodal RAG Search Demo")
st.markdown(
    "This app integrates **Colpali visual search** with **multimodal LLMs (OpenAI Compatible API or AWS Bedrock)**. "
    "The system first finds relevant images and then sends the top image to the LLM for direct analysis and answering."
)

# --- Initialization and Connection ---
try:
    # Load Elasticsearch credentials
    load_dotenv('elastic.env')
    ES_URL = os.getenv("ES_URL")
    ES_API_KEY = os.getenv("ES_API_KEY")
    
    if not all([ES_URL, ES_API_KEY]):
        st.error("🚨 Elasticsearch credentials are missing. Please check your `elastic.env` file.")
        st.stop()
    
    # Connect to Elasticsearch and load model
    model, processor = load_colqwen_model()
    es = connect_to_elasticsearch(ES_URL, ES_API_KEY)
    
    # --- Auto-detect available LLM backends ---
    AVAILABLE_LLM_BACKENDS = []
    
    # Check for OpenAI Compatible API
    OPENAI_BASE_URL = None
    OPENAI_API_KEY = None
    OPENAI_MODEL = None
    if os.path.exists('openai.env'):
        load_dotenv('openai.env', override=True)
        OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        try:
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"} if OPENAI_API_KEY else {}
            openai_test = requests.get(f"{OPENAI_BASE_URL}/models", headers=headers, timeout=10)
            if openai_test.status_code == 200:
                st.success(f"✅ OpenAI Compatible API detected at: {OPENAI_BASE_URL}")
                st.info(f"Using model: {OPENAI_MODEL}")
                AVAILABLE_LLM_BACKENDS.append("OpenAI")
            else:
                # Some providers don't support /models endpoint, try anyway
                st.success(f"✅ OpenAI Compatible API configured at: {OPENAI_BASE_URL}")
                st.info(f"Using model: {OPENAI_MODEL}")
                AVAILABLE_LLM_BACKENDS.append("OpenAI")
        except requests.exceptions.RequestException as e:
            st.warning(f"⚠️ openai.env exists but cannot connect to {OPENAI_BASE_URL}: {e}")
    else:
        st.info("ℹ️ openai.env not found - OpenAI backend disabled")
    
    # Check for AWS Bedrock
    BEDROCK_CLIENT = None
    AWS_REGION = None
    if os.path.exists('aws.env'):
        load_dotenv('aws.env', override=True)
        AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY", "")
        AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY", "")
        AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
        
        if AWS_ACCESS_KEY and AWS_SECRET_KEY and AWS_ACCESS_KEY != "<your-aws-access-key>":
            try:
                BEDROCK_CLIENT = connect_to_bedrock(AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION)
                AVAILABLE_LLM_BACKENDS.append("AWS Bedrock")
            except Exception as e:
                st.warning(f"⚠️ aws.env exists but cannot connect to Bedrock: {e}")
        else:
            st.warning("⚠️ aws.env exists but credentials are not set properly")
    else:
        st.info("ℹ️ aws.env not found - AWS Bedrock backend disabled")
    
    # Check if at least one backend is available
    if not AVAILABLE_LLM_BACKENDS:
        st.error("❌ No LLM backends available! Please create openai.env or aws.env")
        st.stop()
    else:
        st.success(f"🎉 Available LLM backends: {', '.join(AVAILABLE_LLM_BACKENDS)}")
        
except Exception as e:
    st.error(f"Failed to initialize. Error: {e}")
    st.stop()

# --- Search UI ---
st.header("Search Interface")

def set_query_text(text):
    st.session_state.query_text = text

st.subheader("Example Queries")
example_queries = [
    "Do you have a benefits policy change notice from HR?",
    "인사팀에서 보내온 복리후생 정책 변경 안내문이 있나?"
]
cols = st.columns(len(example_queries))
for i, query in enumerate(example_queries):
    with cols[i]:
        st.button(query, on_click=set_query_text, args=(query,), width='stretch')

# --- Search Form ---
with st.form(key='search_form'):
    search_query = st.text_input(
        "Enter your search query or click an example above:", 
        value=st.session_state.query_text,
        key='search_input'
    )
    
    col1, col2 = st.columns(2)
    with col1:
        search_mode = st.radio(
            "Select Search Mode:",
            ["A. Colpali(colqwen) RAG search (Part 1)", "B. Average RAG search (Part 2 - KNN Only)", "C. Rescore RAG search (Part 2 - KNN + Rescore)"],
            index=0
        )
    with col2:
        llm_backend = st.radio(
            "Select LLM Backend:",
            AVAILABLE_LLM_BACKENDS,
            index=0
        )
    
    submitted = st.form_submit_button("Search and Generate Answer")

# --- Form Submission Logic ---
if submitted:
    st.session_state.query_text = search_query
    if not st.session_state.query_text:
        st.warning("Please enter a query to search.")
    else:
        with st.spinner("Step 1: Performing visual search with Colpali..."):
            try:
                query_vectors_float = create_colqwen_query_vectors(st.session_state.query_text, model, processor)
                query_avg_vector = to_avg_vector(query_vectors_float) if not search_mode.startswith("A.") else None

                index_name, es_query_body = build_es_query(search_mode, query_vectors_float, query_avg_vector)

                start_time = time.time()
                response = es.search(index=index_name, body=es_query_body)
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                
                hits = response["hits"]["hits"]
                
                st.subheader("Visual Search Results")
                st.metric(label="🚀 Search Latency", value=f"{latency_ms:.2f} ms")
                display_results(hits)
                
                st.subheader("LLM Generated Answer from Top Image")
                with st.spinner(f"Step 2: Analyzing image and generating answer with {llm_backend}..."):
                    # Prepare backend configuration
                    backend_config = {}
                    if llm_backend == "OpenAI":
                        backend_config = {
                            'openai_base_url': OPENAI_BASE_URL,
                            'openai_api_key': OPENAI_API_KEY,
                            'openai_model': OPENAI_MODEL
                        }
                    elif llm_backend == "AWS Bedrock":
                        backend_config = {
                            'bedrock_client': BEDROCK_CLIENT
                        }
                    
                    llm_answer = generate_llm_answer_with_image(llm_backend, st.session_state.query_text, hits, **backend_config)
                    st.markdown(llm_answer)

            except Exception as e:
                st.error(f"An error occurred during the process: {e}")
