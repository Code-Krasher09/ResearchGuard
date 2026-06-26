import spacy
from typing import List, Optional
import json
from src.schemas.claim import Claim
from loguru import logger

class ClaimExtractor:
    """
    Extracts atomic factual claims from text.
    Primary approach is sentence-based segmentation using SpaCy.
    """
    def __init__(self, min_claim_length: int = 10):
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger", "lemmatizer"])
            self.nlp.add_pipe("sentencizer")
        except OSError:
            raise RuntimeError("Please run: python -m spacy download en_core_web_sm")
            
        self.min_claim_length = min_claim_length
        self.conversational_fillers = {
            "i think", "perhaps", "in my opinion", "it seems", "maybe", 
            "based on the evidence", "according to the document", 
            "the text states", "as mentioned", "it is likely"
        }
        self.refusal_patterns = {
            "i don't know",
            "insufficient evidence",
            "not enough information",
            "do not have enough information",
            "cannot determine",
            "document does not mention",
            "not mentioned",
            "unable to answer",
            "unrelated to the provided evidence",
            "cannot provide an answer",
            "cannot provide information"
        }

    def _clean_sentence(self, sentence: str) -> str:
        """
        Removes conversational fillers from the beginning of a sentence.
        """
        cleaned = sentence.strip()
        lower_cleaned = cleaned.lower()
        
        for filler in self.conversational_fillers:
            if lower_cleaned.startswith(filler):
                # Remove filler and any following punctuation/space
                cleaned = cleaned[len(filler):].lstrip(" ,.:;-")
                # Capitalize first letter safely
                if cleaned:
                    cleaned = cleaned[0].upper() + cleaned[1:]
                lower_cleaned = cleaned.lower()
        
        return cleaned.strip()

    def _detect_claim_type(self, claim_text: str) -> str:
        """
        Classifies claim into FACTUAL, REFUSAL, or META.
        """
        lower_text = claim_text.lower()
        for pattern in self.refusal_patterns:
            if pattern in lower_text:
                return "REFUSAL"
        return "FACTUAL"

    def extract_claims(self, answer: str) -> List[Claim]:
        """
        Extracts independent claims from an answer using sentence segmentation.
        """
        if not answer or not answer.strip():
            return []
            
        doc = self.nlp(answer)
        claims = []
        position = 0
        
        for sent in doc.sents:
            cleaned_text = self._clean_sentence(sent.text)
            # Filter out empty or extremely short sentences
            if len(cleaned_text) >= self.min_claim_length:
                claim_type = self._detect_claim_type(cleaned_text)
                claims.append(Claim(
                    id=f"claim_{position:04d}",
                    text=cleaned_text,
                    position=position,
                    claim_type=claim_type
                ))
                position += 1
                
        logger.debug(f"Extracted {len(claims)} claims from answer.")
        return claims

    def extract_json(self, text: str) -> List[Claim]:
        """
        Optional utility to parse LLM-generated JSON claim structures.
        Expects format: [{"id": "...", "text": "..."}]
        """
        try:
            data = json.loads(text)
            claims = []
            for i, item in enumerate(data):
                claims.append(Claim(
                    id=item.get("id", f"claim_{i:04d}"),
                    text=item.get("text", ""),
                    position=i
                ))
            return claims
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON claims: {e}")
            return []
