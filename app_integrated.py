# streamlit_apps/app_integrated.py

# This should be the very first line to suppress the known torch/streamlit watcher issue
import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
import torch
import numpy as np
import time
import json
import re
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from PIL import Image
from colpali_engine.models import ColQwen3, ColQwen3Processor
from transformers import AutoConfig

# --- 1. Page Configuration and Constants ---
st.set_page_config(
    page_title="ColQwen Integrated Search Demo",
    page_icon="🔎",
    layout="wide"
)

# Constants for different search modes
INDEX_NAME_PART1 = "colqwen3-rvlcdip-demo-part1"
VECTOR_FIELD_NAME_PART1 = "colqwen_vectors"
INDEX_NAME_PART2 = "colqwen3-rvlcdip-demo-part2" # Corrected index name
# FIX: This variable must point to the multi-vector field in the Part 2 index, which is 'colqwen_vectors'
VECTOR_FIELD_NAME_PART2_MULTI = "colqwen_vectors" 
AVG_VECTOR_FIELD_NAME_PART2 = "colqwen_avg_vector"
MODEL_NAME = "TomoroAI/tomoro-colqwen3-embed-4b"

# Initialize session state
if 'query_text' not in st.session_state:
    st.session_state.query_text = ""

# --- 2. Backend Functions (Cached for Performance) ---

@st.cache_resource
def load_colqwen_model():
    """Loads the ColQwen model and processor."""
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
    """Connects to Elasticsearch with a timeout."""
    st.info("Connecting to Elasticsearch...")
    if ":" in elastic_host and not elastic_host.startswith("http"):
        es = Elasticsearch(
            cloud_id=elastic_host, 
            api_key=api_key, 
            request_timeout=30
        )
    else:
        es = Elasticsearch(
            hosts=[elastic_host], 
            api_key=api_key, 
            request_timeout=30,
            verify_certs=False,
            ssl_show_warn=False
        )
    st.success("Connected to Elasticsearch.")
    return es

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

def remove_query_vector_from_explanation(explanation_obj):
    """Recursively removes the lengthy 'query_vector' from the explanation object."""
    if isinstance(explanation_obj, dict):
        if 'description' in explanation_obj and isinstance(explanation_obj['description'], str):
            explanation_obj['description'] = re.sub(
                r"query_vector=\[\[.*?\]\]", 
                "query_vector=[...vector omitted for brevity...]", 
                explanation_obj['description']
            )
        if 'params' in explanation_obj and 'query_vector' in explanation_obj['params']:
            explanation_obj['params']['query_vector'] = "[...vector omitted for brevity...]"
        for key, value in explanation_obj.items():
            remove_query_vector_from_explanation(value)
    elif isinstance(explanation_obj, list):
        for item in explanation_obj:
            remove_query_vector_from_explanation(item)
    return explanation_obj

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
                st.image(image, caption=f"Category: {category}", use_container_width=True)
            else:
                st.warning(f"Image not found at: {path}")
            st.metric(label="Score", value=f"{score:.4f}")
            st.caption(f"ID: {hit['_id'][:20]}...")

def build_es_query(search_mode, query_multi_vectors, query_avg_vector):
    """Builds the Elasticsearch query body based on the selected search mode."""
    index_name = ""
    es_query_body = {
        "size": 5,
        "_source": ["image_path", "category"],
        "explain": True
    }

    if search_mode.startswith("A."):
        index_name = INDEX_NAME_PART1
        es_query_body["query"] = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": f"maxSimDotProduct(params.query_vector, '{VECTOR_FIELD_NAME_PART1}')",
                    "params": {"query_vector": query_multi_vectors}
                }
            }
        }
    
    elif search_mode.startswith("B."):
        index_name = INDEX_NAME_PART2
        es_query_body["knn"] = {
            "field": AVG_VECTOR_FIELD_NAME_PART2,
            "query_vector": query_avg_vector,
            "k": 200,
            "num_candidates": 300
        }
    
    else: # C. KNN + Rescore
        index_name = INDEX_NAME_PART2
        es_query_body.update({
            "knn": {
                "field": AVG_VECTOR_FIELD_NAME_PART2,
                "query_vector": query_avg_vector,
                "k": 200,
                "num_candidates": 400
            },
            "rescore": {
                "window_size": 50,
                "query": {
                    "rescore_query": {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": f"maxSimDotProduct(params.query_vector, '{VECTOR_FIELD_NAME_PART2_MULTI}')",
                                "params": {"query_vector": query_multi_vectors}
                            }
                        }
                    },
                    "query_weight": 0.0,
                    "rescore_query_weight": 1.0
                }
            }
        })
        
    return index_name, es_query_body

# --- 3. UI and Application Logic ---

st.title("Integrated Visual Document Search Demo (Part 1 & 2)")

try:
    dotenv_path = 'elastic.env'
    load_dotenv(dotenv_path=dotenv_path)
    ES_URL = os.getenv("ES_URL")
    ES_API_KEY = os.getenv("ES_API_KEY")

    if not ES_URL or not ES_API_KEY:
        st.error(f"🚨 Elasticsearch credentials not found. Please create an '{dotenv_path}' file with ES_URL and ES_API_KEY.")
        st.stop()

    model, processor = load_colqwen_model()
    es = connect_to_elasticsearch(ES_URL, ES_API_KEY)

except Exception as e:
    st.error(f"Failed to initialize. Error: {e}")
    st.stop()

st.header("Search Interface")

def set_query_text(text):
    st.session_state.query_text = text

st.subheader("Example Queries")
# Update button text as requested
example_queries = [
    "Do you have a benefits policy change notice from HR?",
    "인사팀에서 보내온 복리후생 정책 변경 안내문이 있나?"
]
cols = st.columns(len(example_queries))
for i, query in enumerate(example_queries):
    with cols[i]:
        st.button(query, on_click=set_query_text, args=(query,), use_container_width=True)

with st.form(key='search_form'):
    search_query = st.text_input(
        "Enter your search query or click an example above:", 
        value=st.session_state.query_text,
        key='search_input'
    )
    search_mode = st.radio(
        "Select Search Mode:",
        ["A. Full Colpali Search (Part 1)", "B. KNN Search Only (Part 2)", "C. KNN + Rescore (Part 2)"],
        index=0,
        horizontal=True
    )
    submitted = st.form_submit_button("Search")

if submitted:
    st.session_state.query_text = search_query
    if not st.session_state.query_text:
        st.warning("Please enter a query to search.")
    else:
        with st.spinner("Performing search..."):
            try:
                query_multi_vectors = create_colqwen_query_vectors(st.session_state.query_text, model, processor)
                query_avg_vector = to_avg_vector(query_multi_vectors) if "B." in search_mode or "C." in search_mode else None
                
                index_name, es_query_body = build_es_query(search_mode, query_multi_vectors, query_avg_vector)

                start_time = time.time()
                response = es.search(index=index_name, body=es_query_body)
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000

                st.subheader(f"Search Results for: {search_mode}")
                st.metric(label="🚀 Search Latency", value=f"{latency_ms:.2f} ms")

                hits = response["hits"]["hits"]
                display_results(hits)
                
                if hits:
                    top_hit_explanation = hits[0].get("_explanation")
                    if top_hit_explanation:
                        with st.expander("Show Explanation for Top Result"):
                            cleaned_explanation = remove_query_vector_from_explanation(top_hit_explanation)
                            st.json(cleaned_explanation)

            except Exception as e:
                st.error(f"An error occurred during search: {e}")
