# streamlit_apps/app_part1.py

# This should be the very first line to suppress the known torch/streamlit watcher issue
import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
import torch
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from PIL import Image
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

# --- 1. Page Configuration and Constants ---
st.set_page_config(
    page_title="ColQwen Rank Vectors Search",
    page_icon="🔎",
    layout="wide"
)

INDEX_NAME = "colqwen-rvlcdip-demo-part1"
VECTOR_FIELD_NAME = "colqwen_vectors"
MODEL_NAME = "tsystems/colqwen2.5-3b-multilingual-v1.0"

# FIX 1: Initialize session state to hold the query text
if 'query_text' not in st.session_state:
    st.session_state.query_text = ""

# --- 2. Backend Functions (Cached for Performance) ---

@st.cache_resource
def load_colqwen_model():
    """Loads the ColQwen model and processor only once and caches them."""
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
    """Generates multi-vector embeddings for a given text query."""
    inputs = processor.process_queries([query_text]).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.cpu().to(torch.float32).numpy().tolist()[0]

def connect_to_elasticsearch(elastic_host, api_key):
    """Connects to Elasticsearch using credentials."""
    if ":" in elastic_host and not elastic_host.startswith("http"):
      es = Elasticsearch(cloud_id=elastic_host, api_key=api_key)
    else:
      es = Elasticsearch(hosts=[elastic_host], api_key=api_key)
    return es

# --- 3. Streamlit UI ---
st.title("ColQwen Visual Document Search (Part 1 Demo)")
st.markdown(
    "This app demonstrates a search on the `rank_vectors` field using `maxSimDotProduct`, "
    "as implemented in `01_colqwen.ipynb`."
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

# --- Search UI ---
st.header("Search Interface")

# FIX 2: Define callback functions to update session state
def set_query_text(text):
    st.session_state.query_text = text

# Create clickable example queries
st.subheader("Example Queries")
col1, col2 = st.columns(2)
with col1:
    st.button(
        "Do you have a benefits policy change notice from HR", 
        on_click=set_query_text, 
        args=("Do you have a benefits policy change notice from HR",),
        use_container_width=True
    )
with col2:
    st.button(
        "인사팀에서 보내온 복리후생 정책 변경 안내문이 있나?", 
        on_click=set_query_text, 
        args=("인사팀에서 보내온 복리후생 정책 변경 안내문이 있나?",),
        use_container_width=True
    )


# Use st.form to allow search on Enter key press
with st.form(key='search_form'):
    # FIX 3: Set the value from session_state and remove the default value.
    search_query = st.text_input(
        "Enter your search query or click an example above:", 
        value=st.session_state.query_text,
        key='search_input'
    )
    submitted = st.form_submit_button("Search")

if submitted:
    # Update session state with the latest from the text box upon submission
    st.session_state.query_text = search_query
    
    if not st.session_state.query_text:
        st.warning("Please enter a query to search.")
    else:
        with st.spinner("Generating query vectors and searching..."):
            try:
                query_vectors = create_colqwen_query_vectors(st.session_state.query_text, model, processor)

                es_query_body = {
                    "size": 5,
                    "query": {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": f"maxSimDotProduct(params.query_vector, '{VECTOR_FIELD_NAME}')",
                                "params": {"query_vector": query_vectors},
                            },
                        }
                    },
                    "_source": ["image_path", "category"]
                }
                
                response = es.search(
                    index=INDEX_NAME, 
                    body=es_query_body
                )

                st.subheader("Search Results")
                if not response["hits"]["hits"]:
                    st.warning("No matching documents found.")
                else:
                    cols = st.columns(5)
                    for i, hit in enumerate(response["hits"]["hits"]):
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
            
            except Exception as e:
                st.error(f"An error occurred during search: {e}")

