import os
import asyncio
import torch
import sys
from dotenv import load_dotenv
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from elasticsearch import Elasticsearch
from document_service import DocumentService
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent
from mcp.types import ImageContent, TextContent

# --- Configuration ---
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', 'elastic.env'))
ES_URL = os.getenv("ES_URL")
ES_API_KEY = os.getenv("ES_API_KEY")
MODEL_NAME = "tsystems/colqwen2.5-3b-multilingual-v1.0"

# --- Globals ---
document_service: DocumentService = None
mcp_server = FastMCP("colpali-mcp-server")

from mcp.types import ImageContent, TextContent

# --- Tool Definitions ---
@mcp_server.tool(
    description="Searches and retrieves information from a knowledge base of internal enterprise documents, including letters, reports, invoices, and emails. Use this tool to find specific documents or answer questions based on their visual and textual content."
)
async def search(
    query: str,
    k: int = 3,
    num_candidates: int = 100,
    rescore_window: int = 50,
    index_name: str = "colqwen-rvlcdip-demo-part2",
) -> list:
    """Performs a semantic search and returns images as context."""
    if not document_service:
        raise Exception("DocumentService is not initialized.")
    
    search_results = await document_service.search(query, k, num_candidates, rescore_window, index_name)
    
    content_items = []
    if not search_results:
        content_items.append(TextContent(type="text", text="No relevant documents found."))
    else:
        content_items.append(TextContent(type="text", text=f"Found {len(search_results)} relevant images:"))
        for result in search_results:
            content_items.append(ImageContent(
                type="image",
                data=result["image_data"],
                mimeType=result["mime_type"]
            ))
            content_items.append(TextContent(
                type="text",
                text=f"Image: {result['image_path']}, Category: {result['category']}, Score: {result['score']:.4f}"
            ))
            
    return content_items

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
from contextlib import asynccontextmanager
import uvicorn

@asynccontextmanager
async def lifespan(app: FastMCP):
    """Load models and initialize clients on server startup."""
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
    else:
        print("Error: Could not initialize DocumentService due to missing dependencies.", file=sys.stderr)
        exit(1)
    
    yield
    
    print("Server is shutting down.", file=sys.stderr)

mcp_server.lifespan = lifespan

if __name__ == "__main__":
    print("Starting Uvicorn server...", file=sys.stderr)
    asgi_app = mcp_server.streamable_http_app()
    uvicorn.run(asgi_app, host="0.0.0.0", port=8000)
