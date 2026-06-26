from typing import List, Dict
from loguru import logger
from src.retrieval.chunker import Chunker
from src.retrieval.embedder import Embedder
from src.retrieval.faiss_store import FAISSStore
from src.schemas.chunk import RetrievedChunk

class Retriever:
    """
    High-level Retriever module combining Chunker, Embedder, and FAISSStore.
    """
    def __init__(self, 
                 model_name: str = "BAAI/bge-small-en-v1.5",
                 chunk_size: int = 5,
                 overlap: int = 1,
                 index_path: str = "models/faiss/index.faiss",
                 metadata_path: str = "models/metadata/metadata.pkl"):
        
        self.chunker = Chunker(chunk_size=chunk_size, overlap=overlap)
        self.embedder = Embedder(model_name=model_name)
        
        dimension = self.embedder.get_dimension()
        self.store = FAISSStore(dimension=dimension, index_path=index_path, metadata_path=metadata_path)

    def load_index(self):
        """Loads an existing FAISS index and metadata from disk."""
        self.store.load()

    def build_from_chunks(self, chunks: List[RetrievedChunk], batch_size: int = 32):
        """
        Embeds chunks and adds them to a new FAISS index.
        """
        logger.info(f"Building index from {len(chunks)} chunks.")
        self.store.build_index()
        self.embedder.embed_chunks(chunks, batch_size=batch_size)
        self.store.add(chunks)

    def build_from_documents(self, documents: List[Dict[str, str]], batch_size: int = 32):
        """
        Chunks raw documents, embeds them, and adds them to a new FAISS index.
        Expected document dict format: {"text": "...", "source": "..."}
        """
        logger.info(f"Building index from {len(documents)} documents.")
        all_chunks = []
        for doc in documents:
            text = doc.get("text", "")
            source = doc.get("source", "unknown")
            chunks = self.chunker.chunk_text(text, source)
            all_chunks.extend(chunks)
            
        self.build_from_chunks(all_chunks, batch_size=batch_size)

    def save(self):
        """Saves the FAISS index and metadata."""
        self.store.save()

    def search(self, query: str, k: int = 5) -> List[RetrievedChunk]:
        """
        Alias for retrieve(). Embeds query and returns top-k retrieved chunks.
        """
        return self.retrieve(query, k)

    def retrieve(self, query: str, k: int = 5) -> List[RetrievedChunk]:
        """
        Embeds a user query and returns the top-k retrieved chunks from the FAISS store.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty.")
            
        if k <= 0:
            raise ValueError("k must be greater than 0")
            
        logger.info(f"Retrieving top {k} chunks for query: '{query}'")
        query_embedding = self.embedder.embed_query(query)
        results = self.store.search(query_embedding, k=k)
        
        logger.info(f"Found {len(results)} chunks.")
        return results
