import logging
import json
import base64
from typing import Any, Dict, List, Optional
from elasticsearch import Elasticsearch
import torch
import numpy as np
from colpali_engine.models import ColQwen3, ColQwen3Processor

logger = logging.getLogger(__name__)

class ToolError(Exception):
    """Exception raised when a tool execution fails."""
    pass

class DocumentService:
    def __init__(
        self,
        es_client: Elasticsearch,
        model: ColQwen3,
        processor: ColQwen3Processor,
    ):
        if not es_client:
            raise ValueError("Elasticsearch client is not initialized.")
        if not model:
            raise ValueError("Model is not loaded.")
        if not processor:
            raise ValueError("Processor is not loaded.")
            
        self.es_client = es_client
        self.model = model
        self.processor = processor
        logger.info("DocumentService initialized successfully.")

    def _calculate_average_vector(self, vectors: List[List[float]]) -> Optional[List[float]]:
        """Calculates the normalized average vector."""
        if not vectors or len(vectors) == 0:
            return None
        avg_vec = np.array(vectors).mean(axis=0)
        norm = np.linalg.norm(avg_vec)
        if norm == 0:
            return avg_vec.tolist()
        return (avg_vec / norm).tolist()

    def _image_to_base64(self, image_path: str) -> Optional[str]:
        """Reads an image file and returns its raw base64 encoded string."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path} to base64: {e}")
            return None

    async def search(
        self,
        query: str,
        k: int = 3, # Return top 3 images for context
        num_candidates: int = 100,
        rescore_window: int = 50,
        index_name: str = "colqwen3-rvlcdip-demo-part2",
    ) -> List[Dict[str, Any]]:
        """
        Performs a semantic search and returns results with base64 encoded images.
        """
        try:
            logger.info(f"Performing search for query: '{query}'")
            inputs = self.processor.process_queries([query]).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            query_multi_vectors = outputs.cpu().to(torch.float32).numpy().tolist()[0]
            query_avg_vector = self._calculate_average_vector(query_multi_vectors)

            if not query_avg_vector:
                raise ToolError("Failed to generate query vectors.")

            es_query = {
                "knn": {
                    "field": "colqwen_avg_vector",
                    "query_vector": query_avg_vector,
                    "k": k,
                    "num_candidates": num_candidates,
                },
                "rescore": {
                    "window_size": rescore_window,
                    "query": {
                        "rescore_query": {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "maxSimDotProduct(params.query_vector, 'colqwen_vectors')",
                                    "params": {"query_vector": query_multi_vectors},
                                },
                            }
                        }
                    },
                },
                "_source": ["image_path", "category"],
            }
            
            response = self.es_client.search(index=index_name, body=es_query, size=k)
            logger.info(f"Search completed. Found {len(response['hits']['hits'])} results.")
            
            results = []
            for hit in response["hits"]["hits"]:
                source = hit.get("_source", {})
                image_path = source.get("image_path")
                if image_path:
                    image_base64 = self._image_to_base64(image_path)
                    if image_base64:
                        results.append({
                            "score": hit.get("_score"),
                            "image_path": image_path,
                            "category": source.get("category"),
                            "image_data": image_base64,
                            "mime_type": "image/png" # Assuming PNG for simplicity
                        })
            return results
        except Exception as e:
            logger.error(f"Error during search: {e}", exc_info=True)
            raise ToolError(f"Error performing search: {str(e)}")

    async def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        index_name: str = "colqwen3-rvlcdip-demo-part1",
    ) -> List[Dict[str, Any]]:
        try:
            response = self.es_client.search(
                index=index_name,
                body={"size": limit, "from": offset, "query": {"match_all": {}}},
            )
            return response["hits"]["hits"]
        except Exception as e:
            raise ToolError(f"Error listing documents: {str(e)}")

    async def get_document(
        self,
        document_id: str,
        index_name: str = "colqwen3-rvlcdip-demo-part1",
    ) -> Dict[str, Any]:
        try:
            response = self.es_client.get(index=index_name, id=document_id)
            return response["_source"]
        except Exception as e:
            raise ToolError(f"Error getting document: {str(e)}")
