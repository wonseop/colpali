import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
import torch
import numpy as np
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from PIL import Image
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

# --- 1. Page Configuration and Constants ---
st.set_page_config(
    page_title="ColQwen Integrated Search (Part 1 & 2)",
    page_icon="🔎",
    layout="wide"
)

INDEX_NAME_PART1 = "colqwen-rvlcdip-demo-part1"
VECTOR_FIELD_NAME_PART1 = "colqwen_vectors"
INDEX_NAME_PART2 = "colqwen-rvlcdip-demo-part2-original"
VECTOR_FIELD_NAME_PART2 = "colqwen_vectors_binary"
AVG_VECTOR_FIELD_NAME_PART2 = "colqwen_avg_vector"
MODEL_NAME = "tsystems/colqwen2.5-3b-multilingual-v1.0"

if 'query_text' not in st.session_state:
    st.session_state.query_text = ""

@st.cache_resource
def load_colqwen_model():
    st.write("Loading ColQwen model... This may take a moment on first run.")
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
    st.write("Model loaded successfully.")
    return model, processor

def create_colqwen_query_vectors(query_text, model, processor):
    inputs = processor.process_queries([query_text]).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.cpu().to(torch.float32).numpy().tolist()[0]

def to_avg_vector(vectors):
    vectors_array = np.array(vectors)
    avg_vector = np.mean(vectors_array, axis=0)
    norm = np.linalg.norm(avg_vector)
    if norm > 0:
        normalized_avg_vector = avg_vector / norm
    else:
        normalized_avg_vector = avg_vector
    return normalized_avg_vector.tolist()

def connect_to_elasticsearch(elastic_host, api_key):
    if ":" in elastic_host and not elastic_host.startswith("http"):
        es = Elasticsearch(cloud_id=elastic_host, api_key=api_key)
    else:
        es = Elasticsearch(hosts=[elastic_host], api_key=api_key)
    return es

def display_results(hits):
    cols = st.columns(5)
    for i, hit in enumerate(hits):
        with cols[i]:
            st.markdown(f"**Rank {i+1}**")
            score = hit["_score"]
            path = hit["_source"]["image_path"]
            category = hit["_source"]["category"]
            if os.path.exists(path):
                image = Image.open(path)
                # FIX: use_container_width로 변경
                st.image(image, caption=f"Category: {category}", use_container_width=True)
            else:
                st.warning(f"Image not found at: {path}")
            st.metric(label="Score", value=f"{score:.4f}")
            st.caption(f"ID: {hit['_id'][:20]}...")

st.title("ColQwen Visual Document Search (Part 1 & 2 Integrated Demo)")
st.markdown(
    "This app demonstrates search functionalities from both Part 1 (`rank_vectors`) and Part 2 (`average_vector` with optional rescoring). "
    "Select a search mode below to switch between different search strategies."
)

try:
    dotenv_path = 'elastic.env'
    load_dotenv(dotenv_path=dotenv_path)
    ELASTIC_HOST = os.getenv("ELASTIC_HOST")
    ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY")

    if not ELASTIC_HOST or not ELASTIC_API_KEY:
        st.error(f"🚨 Elasticsearch credentials not found. Please create an '{dotenv_path}' file.")
        st.stop()
    model, processor = load_colqwen_model()
    es = connect_to_elasticsearch(ELASTIC_HOST, ELASTIC_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize. Error: {e}")
    st.stop()

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

with st.form(key='search_form'):
    search_query = st.text_input(
        "Enter your search query or click an example above:", 
        value=st.session_state.query_text,
        key='search_input'
    )
    search_mode = st.radio(
        "Select Search Mode:",
        ["A. Colpali(colqwen) search (Part 1)", "B. Average search (Part 2 - KNN Only)", "C. Rescore search (Part 2 - KNN + Rescore)"],
        index=0,
        horizontal=True,
        help="Choose the search strategy to apply. 'A' uses the original Part 1 index, while 'B' and 'C' use the Part 2 index with different techniques."
    )
    submitted = st.form_submit_button("Search")

if submitted:
    st.session_state.query_text = search_query
    if not st.session_state.query_text:
        st.warning("Please enter a query to search.")
    else:
        with st.spinner("Generating query vectors and searching..."):
            try:
                query_vectors_float = create_colqwen_query_vectors(st.session_state.query_text, model, processor)
                if search_mode == "A. Colpali(colqwen) search (Part 1)":
                    index_name = INDEX_NAME_PART1
                    es_query_body = {
                        "size": 5,
                        "query": {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": f"maxSimDotProduct(params.query_vector, '{VECTOR_FIELD_NAME_PART1}')",
                                    "params": {"query_vector": query_vectors_float},
                                },
                            }
                        },
                        "_source": ["image_path", "category"]
                    }
                elif search_mode == "B. Average search (Part 2 - KNN Only)":
                    index_name = INDEX_NAME_PART2
                    query_avg_vector = to_avg_vector(query_vectors_float)
                    es_query_body = {
                        "size": 5,
                        "knn": {
                            "field": AVG_VECTOR_FIELD_NAME_PART2,
                            "query_vector": query_avg_vector,
                            "k": 10,
                            "num_candidates": 100
                        },
                        "_source": ["image_path", "category"]
                    }
                else:  # C. Rescore search (Part 2 - KNN + Rescore)
                    index_name = INDEX_NAME_PART2
                    query_avg_vector = to_avg_vector(query_vectors_float)
                    knn_query = {
                        "field": AVG_VECTOR_FIELD_NAME_PART2,
                        "query_vector": query_avg_vector,
                        "k": 10,
                        "num_candidates": 100
                    }
                    rescore_definition = {
                        "window_size": 10,
                        "query": {
                            "rescore_query": {
                                "script_score": {
                                    "query": {"match_all": {}},
                                    "script": {
                                        "source": f"maxSimDotProduct(params.query_vector, '{VECTOR_FIELD_NAME_PART2}')",
                                        "params": {"query_vector": query_vectors_float}
                                    }
                                }
                            },
                            "query_weight": 0.0,
                            "rescore_query_weight": 1.0
                        }
                    }
                    es_query_body = {
                        "size": 5,
                        "knn": knn_query,
                        "rescore": rescore_definition,
                        "_source": ["image_path", "category"]
                    }
                response = es.search(index=index_name, body=es_query_body)
                st.subheader("Search Results")
                if not response["hits"]["hits"]:
                    st.warning("No matching documents found.")
                else:
                    display_results(response["hits"]["hits"])
            except Exception as e:
                st.error(f"An error occurred during search: {e}")

