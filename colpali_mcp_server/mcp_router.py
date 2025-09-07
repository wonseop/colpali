# mcp_router.py
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
import torch
import numpy as np

# This is a simplified duplication of models and functions from main.py
# to avoid circular imports. In a larger application, this would be
# handled by a shared dependency/service layer.
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from elasticsearch import Elasticsearch

# --- Pydantic Models (duplicated from main.py) ---
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

# --- Global variables from main.py needed here ---
model: ColQwen2_5 = None
processor: ColQwen2_5_Processor = None
es_client: Elasticsearch = None

router = APIRouter()

# --- Helper Functions (duplicated from main.py) ---
def calculate_average_vector(vectors):
    if not vectors or len(vectors) == 0:
        return None
    avg_vec = np.array(vectors).mean(axis=0)
    norm = np.linalg.norm(avg_vec)
    if norm == 0:
        return avg_vec.tolist()
    return (avg_vec / norm).tolist()

# --- Tool Implementations as HTTP Endpoints ---
@router.post("/tools/search")
async def search_chunks(request: SearchRequest):
    global model, processor, es_client
    try:
        inputs = processor.process_queries([request.query]).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        query_multi_vectors = outputs.cpu().to(torch.float32).numpy().tolist()[0]
        query_avg_vector = calculate_average_vector(query_multi_vectors)

        if not query_avg_vector:
            raise HTTPException(status_code=500, detail="Failed to generate query vectors.")

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
        response = es_client.search(index=request.index_name, body=es_query, size=request.k)
        return response['hits']['hits']
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tools/list_documents")
async def list_documents(request: DocumentListRequest):
    global es_client
    try:
        response = es_client.search(
            index=request.index_name,
            body={"size": request.limit, "from": request.offset, "query": {"match_all": {}}}
        )
        return response['hits']['hits']
    except Exception as e:
        return {"error": str(e)}

@router.post("/tools/get_document")
async def get_document(request: DocumentDetailRequest):
    global es_client
    try:
        response = es_client.get(index=request.index_name, id=request.document_id)
        return response['_source']
    except Exception as e:
        return {"error": str(e)}

# --- WebSocket Endpoint for MCP ---
@router.websocket("/v1")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("MCP Client connected.")
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("method") == "tool/execute":
                params = data.get("params", {})
                tool_name = params.get("name")
                tool_input = params.get("input", {})
                
                result = None
                error = None

                try:
                    if tool_name == "search":
                        search_request = SearchRequest(**tool_input)
                        result = await search_chunks(search_request)
                    elif tool_name == "list_documents":
                        list_request = DocumentListRequest(**tool_input)
                        result = await list_documents(list_request)
                    elif tool_name == "get_document":
                        detail_request = DocumentDetailRequest(**tool_input)
                        result = await get_document(detail_request)
                    else:
                        error = {"code": -32601, "message": "Method not found"}
                except Exception as e:
                    error = {"code": -32603, "message": f"Internal error: {str(e)}"}

                if error:
                    response = {"jsonrpc": "2.0", "id": data.get("id"), "error": error}
                else:
                    response = {"jsonrpc": "2.0", "id": data.get("id"), "result": result}
                
                await websocket.send_json(response)

            elif data.get("method") == "initialize":
                response = {
                    "jsonrpc": "2.0", 
                    "id": data.get("id"), 
                    "result": {"message": "Server initialized successfully."}
                }
                await websocket.send_json(response)

    except WebSocketDisconnect:
        print("MCP Client disconnected.")
