import os
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from elasticsearch import Elasticsearch

# --- Configuration ---
# Load environment variables from .env file in the parent directory
dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'elastic.env')
load_dotenv(dotenv_path=dotenv_path)

ES_URL = os.getenv("ES_URL")
ES_API_KEY = os.getenv("ES_API_KEY")
MODEL_NAME = "tsystems/colqwen2.5-3b-multilingual-v1.0"

# --- Global Variables ---
app = FastAPI(title="Colpali MCP Server")
model = None
processor = None
es_client = None

# --- Pydantic Models ---
class SearchRequest(BaseModel):
    query: str
    k: int = 10
    num_candidates: int = 100
    rescore_window: int = 50
    index_name: str = "colqwen-rvlcdip-demo-part2"

class DocumentListRequest(BaseModel):
    limit: int = 100
    offset: int = 0
    index_name: str = "colqwen-rvlcdip-demo-part1"

class DocumentDetailRequest(BaseModel):
    document_id: str
    index_name: str = "colqwen-rvlcdip-demo-part1"


@app.on_event("startup")
def startup_event():
    """
    Load models and initialize clients on server startup.
    """
    global model, processor, es_client
    
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
    
    print("Connecting to Elasticsearch...")
    if ':' in ES_URL and not ES_URL.startswith('http'):
        es_client = Elasticsearch(cloud_id=ES_URL, api_key=ES_API_KEY, request_timeout=30)
    else:
        es_client = Elasticsearch(hosts=[ES_URL], api_key=ES_API_KEY, request_timeout=30)
    print(f"Connected to Elasticsearch version: {es_client.info()['version']['number']}")


# --- MCP Tool Implementations (to be filled in) ---

def calculate_average_vector(vectors):
    """Calculates a single, normalized average vector from multi-vectors."""
    if not vectors or len(vectors) == 0:
        return None
    avg_vec = np.array(vectors).mean(axis=0)
    norm = np.linalg.norm(avg_vec)
    if norm == 0:
        return avg_vec.tolist()
    return (avg_vec / norm).tolist()

@app.post("/tools/search")
async def search_chunks(request: SearchRequest):
    """
    Performs a semantic search using Colpali embeddings with a KNN + Rescore strategy.
    """
    try:
        # 1. Convert query to multi-vector and average vector
        inputs = processor.process_queries([request.query]).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        query_multi_vectors = outputs.cpu().to(torch.float32).numpy().tolist()[0]
        query_avg_vector = calculate_average_vector(query_multi_vectors)

        if not query_avg_vector:
            raise HTTPException(status_code=500, detail="Failed to generate query vectors.")

        # 2. Create Elasticsearch k-NN + Rescore query
        es_query = {
            "knn": {
                "field": "colqwen_avg_vector",
                "query_vector": query_avg_vector,
                "k": request.k,
                "num_candidates": request.num_candidates
            },
            "rescore": {
                "window_size": request.rescore_window,
                "query": {
                    "rescore_query": {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "maxSimDotProduct(params.query_vector, 'colqwen_vectors')",
                                "params": {"query_vector": query_multi_vectors}
                            }
                        }
                    }
                }
            },
            "_source": ["image_path", "category"]
        }

        # 3. Execute the search
        response = es_client.search(
            index=request.index_name,
            body=es_query,
            size=request.k
        )
        return response['hits']['hits']
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/list_documents")
async def list_documents(request: DocumentListRequest):
    """
    Lists documents from a specified index.
    """
    try:
        response = es_client.search(
            index=request.index_name,
            body={
                "size": request.limit,
                "from": request.offset,
                "query": {
                    "match_all": {}
                }
            }
        )
        return response['hits']['hits']
    except Exception as e:
        return {"error": str(e)}

@app.post("/tools/get_document")
async def get_document(request: DocumentDetailRequest):
    """
    Retrieves a single document by its ID.
    """
    try:
        response = es_client.get(
            index=request.index_name,
            id=request.document_id
        )
        return response['_source']
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
