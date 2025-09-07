import logging
from typing import Any, Dict, List, Optional
from elasticsearch import Elasticsearch
import torch
import numpy as np

# Assuming model and processor are loaded and passed from main.py
# This avoids direct dependency on model loading within the service.
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

logger = logging.getLogger(__name__)

class ToolError(Exception):
    """Exception raised when a tool execution fails."""
    pass

class DocumentService:
    def __init__(
        self,
        es_client: Elasticsearch,
        model: ColQwen2_5,
        processor: ColQwen2_5_Processor,
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

    async def search(
        self,
        query: str,
        k: int = 10,
        num_candidates: int = 100,
        rescore_window: int = 50,
        index_name: str = "colqwen-rvlcdip-demo-part2",
    ) -> List[Dict[str, Any]]:
        """
        Performs a semantic search using Colpali embeddings.
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
            return response["hits"]["hits"]
        except Exception as e:
            logger.error(f"Error during search: {e}", exc_info=True)
            raise ToolError(f"Error performing search: {str(e)}")

    async def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        index_name: str = "colqwen-rvlcdip-demo-part1",
    ) -> List[Dict[str, Any]]:
        """Lists documents from a specified index."""
        try:
            logger.info(f"Listing documents from index '{index_name}' with limit {limit} and offset {offset}.")
            response = self.es_client.search(
                index=index_name,
                body={"size": limit, "from": offset, "query": {"match_all": {}}},
            )
            logger.info(f"Found {len(response['hits']['hits'])} documents.")
            return response["hits"]["hits"]
        except Exception as e:
            logger.error(f"Error listing documents: {e}", exc_info=True)
            raise ToolError(f"Error listing documents: {str(e)}")

    async def get_document(
        self,
        document_id: str,
        index_name: str = "colqwen-rvlcdip-demo-part1",
    ) -> Dict[str, Any]:
        """Retrieves a single document by its ID."""
        try:
            logger.info(f"Getting document '{document_id}' from index '{index_name}'.")
            response = self.es_client.get(index=index_name, id=document_id)
            return response["_source"]
        except Exception as e:
            logger.error(f"Error getting document: {e}", exc_info=True)
            raise ToolError(f"Error getting document: {str(e)}")
