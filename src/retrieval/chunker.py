import uuid
import spacy
from typing import List
from loguru import logger
from src.schemas.chunk import RetrievedChunk

class Chunker:
    """
    A class for splitting text into overlapping sentence-based chunks using SpaCy.
    """
    def __init__(self, chunk_size: int = 5, overlap: int = 1):
        """
        Initializes the Chunker.
        
        Args:
            chunk_size (int): Number of sentences per chunk.
            overlap (int): Number of overlapping sentences between chunks.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= chunk_size:
            raise ValueError("overlap must be < chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap
        
        logger.info(f"Initializing Chunker with size={chunk_size}, overlap={overlap}")
        
        try:
            # We don't need the full pipeline just for sentence segmentation
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
        except OSError:
            raise RuntimeError("Please run: python -m spacy download en_core_web_sm")
            
        # Ensure we have a sentencizer
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")

    def chunk_text(self, text: str, source: str) -> List[RetrievedChunk]:
        """
        Splits text into smaller overlapping chunks based on sentences.
        
        Args:
            text (str): The raw text to split.
            source (str): The source identifier.
            
        Returns:
            List[RetrievedChunk]: A list of chunk objects with unique IDs.
        """
        logger.debug(f"Chunking text from source: {source}")
        
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        chunks = []
        if not sentences:
            logger.warning(f"No sentences found in text from source: {source}")
            return chunks

        i = 0
        while i < len(sentences):
            chunk_sentences = sentences[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_sentences)
            
            chunk = RetrievedChunk(
                chunk_id=str(uuid.uuid4()),
                text=chunk_text,
                source=source,
                score=0.0  # Default initial score
            )
            chunks.append(chunk)
            
            i += self.chunk_size - self.overlap
            
        logger.info(f"Created {len(chunks)} chunks from {source}")
        return chunks
