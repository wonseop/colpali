import os
import asyncio
import torch
import sys
from dotenv import load_dotenv
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from elasticsearch import Elasticsearch
from document_service import DocumentService
from mcp.server.fastmcp import FastMCP

# --- Configuration ---
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', 'elastic.env'))
ES_URL = os.getenv("ES_URL")
ES_API_KEY = os.getenv("ES_API_KEY")
MODEL_NAME = "tsystems/colqwen2.5-3b-multilingual-v1.0"

# --- Globals ---
document_service: DocumentService = None
mcp_server = FastMCP("colpali-mcp-server")

# --- Tool Definitions ---
@mcp_server.tool()
async def search(
    query: str,
    k: int = 10,
    num_candidates: int = 100,
    rescore_window: int = 50,
    index_name: str = "colqwen-rvlcdip-demo-part2",
) -> list:
    """Performs a semantic search using Colpali embeddings."""
    if not document_service:
        raise Exception("DocumentService is not initialized.")
    return await document_service.search(query, k, num_candidates, rescore_window, index_name)

@mcp_server.tool()
async def list_documents(
    limit: int = 100,
    offset: int = 0,
    index_name: str = "colqwen-rvlcdip-demo-part1",
) -> list:
    """Lists documents from a specified index."""
    if not document_service:
        raise Exception("DocumentService is not initialized.")
    return await document_service.list_documents(limit, offset, index_name)

@mcp_server.tool()
async def get_document(
    document_id: str,
    index_name: str = "colqwen-rvlcdip-demo-part1",
) -> dict:
    """Retrieves a single document by its ID."""
    if not document_service:
        raise Exception("DocumentService is not initialized.")
    return await document_service.get_document(document_id, index_name)

# --- Server Lifecycle ---
def main():
    """Load models, initialize clients, and start the MCP server."""
    global document_service
    
    device_map = "cpu"
    if torch.cuda.is_available():
        device_map = "cuda:0"
    
    print(f"Loading model '{MODEL_NAME}' on device '{device_map}'...", file=sys.stderr)
    model = ColQwen2_5.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if device_map != "cpu" else torch.float32,
        device_map=device_map
    ).eval()
    processor = ColQwen2_5_Processor.from_pretrained(MODEL_NAME)
    print("Model loaded successfully.", file=sys.stderr)

    print("Connecting to Elasticsearch...", file=sys.stderr)
    es_client = None
    if ES_URL and ':' in ES_URL and not ES_URL.startswith('http'):
        es_client = Elasticsearch(cloud_id=ES_URL, api_key=ES_API_KEY, request_timeout=30)
    elif ES_URL:
        es_client = Elasticsearch(hosts=[ES_URL], api_key=ES_API_KEY, request_timeout=30)
    
    if es_client and model and processor:
        document_service = DocumentService(es_client=es_client, model=model, processor=processor)
        print(f"Connected to Elasticsearch version: {es_client.info()['version']['number']}", file=sys.stderr)

        print("Starting MCP server with Streamable HTTP transport...", file=sys.stderr)
        mcp_server.run(transport="streamable-http", host="0.0.0.0", port=8000)

    else:
        print("Error: Could not initialize DocumentService due to missing dependencies.", file=sys.stderr)
        exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Server stopped by user.", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred during server execution: {e}", file=sys.stderr)
        sys.exit(1)
