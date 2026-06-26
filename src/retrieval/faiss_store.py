import os
import faiss
import pickle
import numpy as np
from typing import List, Dict
from loguru import logger
from src.schemas.chunk import RetrievedChunk

class FAISSStore:
    """
    A FAISS-based vector store using IndexFlatIP for cosine similarity 
    (assuming embeddings are normalized).
    """
    def __init__(self, dimension: int, index_path: str = "models/faiss/index.faiss", metadata_path: str = "models/metadata/metadata.pkl"):
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata: Dict[int, RetrievedChunk] = {}
        self._next_id = 0

    def build_index(self):
        """Initializes a new empty IndexFlatIP index."""
        logger.info(f"Building new FAISS IndexFlatIP with dimension {self.dimension}")
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = {}
        self._next_id = 0

    def add(self, chunks: List[RetrievedChunk]):
        """
        Adds a list of chunks (with their embeddings) to the index.
        """
        if not chunks:
            return
            
        if self.index is None:
            self.build_index()
            
        # Extract embeddings
        embeddings_list = [chunk.embedding for chunk in chunks]
        
        if any(emb is None for emb in embeddings_list):
            raise ValueError("All chunks must have an embedding before being added to FAISS.")
            
        embeddings_np = np.array(embeddings_list, dtype=np.float32)
        
        # Verify dimension
        if embeddings_np.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings_np.shape[1]} does not match index dimension {self.dimension}")
            
        # Verify normalization
        norms = np.linalg.norm(embeddings_np, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("Embeddings must be normalized before adding to IndexFlatIP.")
            
        # Contiguous array
        embeddings_np = np.ascontiguousarray(embeddings_np)
        
        # Add to index
        self.index.add(embeddings_np)
        
        # Store metadata
        for chunk in chunks:
            self.metadata[self._next_id] = chunk
            self._next_id += 1
            
        logger.info(f"Added {len(chunks)} chunks to FAISS. Total index size: {self.index.ntotal}")

    def search(self, query_embedding: List[float], k: int = 5) -> List[RetrievedChunk]:
        """
        Searches the index for the top-k most similar chunks.
        """
        if k <= 0:
            raise ValueError("k must be greater than 0")
            
        query_np = np.array([query_embedding], dtype=np.float32)
        
        if query_np.shape[1] != self.dimension:
            raise ValueError(f"Query dimension {query_np.shape[1]} does not match index dimension {self.dimension}")
            
        # Verify normalization
        norms = np.linalg.norm(query_np, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("Query embedding must be normalized before searching IndexFlatIP.")
            
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Search called on empty or uninitialized index.")
            return []
            
        query_np = np.ascontiguousarray(query_np)
        distances, indices = self.index.search(query_np, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue # FAISS returns -1 if there are fewer than k results
                
            chunk = self.metadata[idx].model_copy(deep=True)
            chunk.score = float(dist)
            results.append(chunk)
            
        return results

    def save(self):
        """Saves the index and metadata to persistent storage."""
        if self.index is None:
            logger.warning("No index to save.")
            return
            
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        
        logger.info(f"Saving FAISS index to {self.index_path}")
        faiss.write_index(self.index, self.index_path)
        
        logger.info(f"Saving metadata to {self.metadata_path}")
        with open(self.metadata_path, "wb") as f:
            pickle.dump({
                "metadata": self.metadata,
                "next_id": self._next_id,
                "dimension": self.dimension
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    def load(self):
        """Loads the index and metadata from persistent storage."""
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Cannot find index or metadata files at {self.index_path}, {self.metadata_path}")
            
        logger.info(f"Loading FAISS index from {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        
        logger.info(f"Loading metadata from {self.metadata_path}")
        with open(self.metadata_path, "rb") as f:
            data = pickle.load(f)
            self.metadata = data["metadata"]
            self._next_id = data["next_id"]
            
            loaded_dim = data.get("dimension", self.dimension)
            if loaded_dim != self.dimension:
                raise ValueError(f"Loaded index dimension ({loaded_dim}) differs from initialized dimension ({self.dimension})")
