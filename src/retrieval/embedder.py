import torch
from typing import List
from sentence_transformers import SentenceTransformer
from loguru import logger
from src.schemas.chunk import RetrievedChunk

MODEL_CACHE = {}

class Embedder:
    """
    Wrapper for the SentenceTransformer model with model-level caching.
    Handles embedding of queries and chunks.
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._initialize_model()

    def _initialize_model(self):
        """Loads the SentenceTransformer model or retrieves it from cache."""
        if self.model_name in MODEL_CACHE:
            self._model = MODEL_CACHE[self.model_name]
            return
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading SentenceTransformer '{self.model_name}' on {device}...")
        self._model = SentenceTransformer(self.model_name, device=device)
        MODEL_CACHE[self.model_name] = self._model
        logger.info(f"Model loaded. Dimension: {self.get_dimension()}")

    def get_dimension(self) -> int:
        """
        Returns the embedding dimension of the loaded model.
        """
        return self._model.get_sentence_embedding_dimension()

    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single query. BGE models require a prefix for queries.
        
        Args:
            query (str): The search query.
            
        Returns:
            List[float]: The normalized embedding vector.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty.")
            
        prefix = "Represent this sentence for searching relevant passages: "
        formatted_query = f"{prefix}{query}"
        logger.debug(f"Embedding query: '{query}'")
        
        # encode returns numpy array. convert to list of floats
        embedding = self._model.encode([formatted_query], normalize_embeddings=True)[0]
        return embedding.tolist()

    def embed_chunks(self, chunks: List[RetrievedChunk], batch_size: int = 32) -> List[List[float]]:
        """
        Embeds a batch of RetrievedChunk objects in-place and returns the embeddings.
        
        Args:
            chunks (List[RetrievedChunk]): The chunks to embed.
            batch_size (int): Batch size for embedding model.
            
        Returns:
            List[List[float]]: A list of normalized embedding vectors.
        """
        if not chunks:
            return []
            
        texts = [chunk.text for chunk in chunks]
        logger.debug(f"Embedding {len(texts)} chunks with batch_size={batch_size}...")
        
        # For BGE models, documents do not need a prefix
        embeddings = self._model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
        
        embeddings_list = embeddings.tolist()
        
        # Assign embeddings back to the chunks
        for chunk, emb in zip(chunks, embeddings_list):
            chunk.embedding = emb
            
        return embeddings_list
