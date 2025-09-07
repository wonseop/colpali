import os
import asyncio
import torch
import sys
from dotenv import load_dotenv
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from elasticsearch import Elasticsearch
from document_service import DocumentService, ToolError

# Import MCP SDK components
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import mcp.types as types

# --- Configuration ---
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', 'elastic.env'))
ES_URL = os.getenv("ES_URL")
ES_API_KEY = os.getenv("ES_API_KEY")
MODEL_NAME = "tsystems/colqwen2.5-3b-multilingual-v1.0"

# --- Globals ---
# This is a simple way to share the service; for more complex apps, consider dependency injection
document_service: DocumentService = None
app = Server("colpali-mcp-server")

# --- Tool Definitions ---
ALL_TOOLS = [
    Tool(
        name="search",
        description="Performs a semantic search using Colpali embeddings with a KNN + Rescore strategy.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query text."},
                "k": {"type": "number", "description": "Number of results to return.", "default": 10},
                "num_candidates": {"type": "number", "description": "Number of candidates for KNN search.", "default": 100},
                "rescore_window": {"type": "number", "description": "Window size for rescoring.", "default": 50},
                "index_name": {"type": "string", "description": "The Elasticsearch index name.", "default": "colqwen-rvlcdip-demo-part2"}
            },
            "required": ["query"]
        }
    ),
    Tool(
        name="list_documents",
        description="Lists documents from a specified index.",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {"type": "number", "description": "Maximum number of documents to return.", "default": 100},
                "offset": {"type": "number", "description": "Number of documents to skip.", "default": 0},
                "index_name": {"type": "string", "description": "The Elasticsearch index name.", "default": "colqwen-rvlcdip-demo-part1"}
            }
        }
    ),
    Tool(
        name="get_document",
        description="Retrieves a single document by its ID.",
        inputSchema={
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "The ID of the document to retrieve."},
                "index_name": {"type": "string", "description": "The Elasticsearch index name.", "default": "colqwen-rvlcdip-demo-part1"}
            },
            "required": ["document_id"]
        }
    )
]

@app.list_tools()
async def list_tools() -> list[Tool]:
    """Returns the list of available tools."""
    return ALL_TOOLS

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Executes a tool based on its name."""
    if not document_service:
        raise Exception("DocumentService is not initialized.")

    try:
        if name == "search":
            result = await document_service.search(**arguments)
        elif name == "list_documents":
            result = await document_service.list_documents(**arguments)
        elif name == "get_document":
            result = await document_service.get_document(**arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
        
        # The SDK expects a list of content objects, so we format the JSON result.
        return [TextContent(type="text", text=str(result))]

    except Exception as e:
        print(f"Error calling tool {name}: {e}", file=sys.stderr)
        # Re-raise the exception to let the SDK handle JSON-RPC error formatting
        raise

async def main():
    """Load models, initialize clients, and start the MCP server."""
    global document_service
    
    device_map = "cpu"
    if torch.backends.mps.is_available():
        device_map = "mps"
    elif torch.cuda.is_available():
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

        print("Starting MCP server with StdioTransport...", file=sys.stderr)
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options()
            )
        print("MCP server has stopped.", file=sys.stderr)

    else:
        print("Error: Could not initialize DocumentService due to missing dependencies.", file=sys.stderr)
        exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped by user.", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred during server execution: {e}", file=sys.stderr)
        sys.exit(1)
