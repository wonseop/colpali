import os
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from elasticsearch import Elasticsearch
from mcp_router import router as mcp_router
import mcp_router as mcp_router_module

# --- Configuration ---
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', 'elastic.env'))
ES_URL = os.getenv("ES_URL")
ES_API_KEY = os.getenv("ES_API_KEY")
MODEL_NAME = "tsystems/colqwen2.5-3b-multilingual-v1.0"

# --- FastAPI App Initialization ---
app = FastAPI(title="Colpali MCP Server")

# Allow all origins for CORS and WebSocket
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(mcp_router, prefix="/mcp")

# --- Model and Client Loading ---
@app.on_event("startup")
def startup_event():
    """Load models and initialize clients on server startup."""
    device_map = "cpu"
    if torch.backends.mps.is_available():
        device_map = "mps"
    elif torch.cuda.is_available():
        device_map = "cuda:0"
    
    print(f"Loading model '{MODEL_NAME}' on device '{device_map}'...")
    model = ColQwen2_5.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if device_map != "cpu" else torch.float32,
        device_map=device_map
    ).eval()
    processor = ColQwen2_5_Processor.from_pretrained(MODEL_NAME)
    print("Model loaded successfully.")

    # Share the loaded models with the router module
    mcp_router_module.model = model
    mcp_router_module.processor = processor

    print("Connecting to Elasticsearch...")
    es_client = None
    if ES_URL and ':' in ES_URL and not ES_URL.startswith('http'):
        es_client = Elasticsearch(cloud_id=ES_URL, api_key=ES_API_KEY, request_timeout=30)
    elif ES_URL:
        es_client = Elasticsearch(hosts=[ES_URL], api_key=ES_API_KEY, request_timeout=30)
    
    if es_client:
        mcp_router_module.es_client = es_client
        print(f"Connected to Elasticsearch version: {es_client.info()['version']['number']}")
    else:
        print("Error: Elasticsearch client could not be initialized. Check ES_URL.")

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
