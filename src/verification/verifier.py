import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List
from src.schemas.claim import Claim
from src.schemas.chunk import RetrievedChunk
from src.schemas.verification import VerificationResult
from loguru import logger

class Verifier:
    """
    Verifies factual claims against retrieved evidence using Natural Language Inference (NLI).
    Utilizes DeBERTa-v3 trained on MNLI, FEVER, and ANLI.
    """
    def __init__(self, model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading NLI Verifier ({model_name}) on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _map_label(self, label_id: int) -> str:
        """
        Dynamically infers label from model config and maps to domain schema.
        """
        raw_label = self.model.config.id2label.get(label_id, "neutral").lower()
        if "entail" in raw_label or "support" in raw_label:
            return "SUPPORTED"
        elif "contradict" in raw_label:
            return "CONTRADICTED"
        else:
            return "NEUTRAL"

    def verify_claim(self, claim: Claim, evidence: List[RetrievedChunk]) -> VerificationResult:
        """
        Verifies a single claim against the provided evidence chunks.
        """
        return self.verify_batch([claim], evidence)[0]

    def verify_batch(self, claims: List[Claim], evidence: List[RetrievedChunk]) -> List[VerificationResult]:
        """
        Verifies a batch of claims against the provided evidence.
        Evaluates each chunk independently and selects the one with the highest confidence.
        """
        if not claims:
            return []
            
        valid_chunks = [c for c in evidence if c.text]
        evidence_chunk_ids = [c.chunk_id for c in valid_chunks]
        
        if not valid_chunks:
            return [
                VerificationResult(
                    claim_id=c.id,
                    claim=c.text,
                    evidence_chunk_ids=[],
                    top_evidence="",
                    label="NEUTRAL",
                    top_confidence=1.0
                )
                for c in claims
            ]
            
        # Create pairs: (claim_idx, chunk_idx, premise, hypothesis)
        pairs = []
        for c_idx, claim in enumerate(claims):
            for e_idx, chunk in enumerate(valid_chunks):
                pairs.append((c_idx, e_idx, chunk.text.strip(), claim.text))
                
        premises = [p[2] for p in pairs]
        hypotheses = [p[3] for p in pairs]
        
        inputs = self.tokenizer(premises, hypotheses, padding=True, truncation=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        probs = torch.softmax(outputs.logits, dim=1)
        predicted_class_ids = probs.argmax(dim=1).tolist()
        confidences = probs.max(dim=1).values.tolist()
        
        # Aggregate best chunk per claim based on label priority
        # Priority: SUPPORTED (3) > CONTRADICTED (2) > NEUTRAL (1)
        priority = {"SUPPORTED": 3, "CONTRADICTED": 2, "NEUTRAL": 1}
        best_results = {}
        
        for i, pair in enumerate(pairs):
            c_idx, e_idx = pair[0], pair[1]
            conf = confidences[i]
            label = self._map_label(predicted_class_ids[i])
            chunk_text = valid_chunks[e_idx].text.strip()
            
            # Update if first time, or if higher priority label, or if same priority but higher confidence
            if c_idx not in best_results:
                best_results[c_idx] = {"label": label, "conf": conf, "top_evidence": chunk_text}
            else:
                curr_priority = priority[best_results[c_idx]["label"]]
                new_priority = priority[label]
                if new_priority > curr_priority:
                    best_results[c_idx] = {"label": label, "conf": conf, "top_evidence": chunk_text}
                elif new_priority == curr_priority and conf > best_results[c_idx]["conf"]:
                    best_results[c_idx] = {"label": label, "conf": conf, "top_evidence": chunk_text}
                
        results = []
        for c_idx, claim in enumerate(claims):
            res = best_results[c_idx]
            results.append(
                VerificationResult(
                    claim_id=claim.id,
                    claim=claim.text,
                    evidence_chunk_ids=evidence_chunk_ids,
                    top_evidence=res["top_evidence"],
                    label=res["label"],
                    top_confidence=res["conf"]
                )
            )
            
        return results
