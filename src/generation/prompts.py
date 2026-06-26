from typing import List, Dict
from src.schemas.chunk import RetrievedChunk

SCIENTIFIC_QA_TEMPLATE = """You are a strict, objective scientific assistant.
Answer the user's question using ONLY the provided evidence.
If the evidence is insufficient to fully answer the question, state explicitly that you do not have enough information.
Do NOT fabricate citations. Be concise and scientifically accurate.

EVIDENCE:
{context}

QUESTION: {query}

ANSWER:"""

QUERY_REWRITE_TEMPLATE = """Rewrite the following user query to be optimized for vector similarity search in a scientific database.
Expand acronyms where obvious, remove conversational filler, and focus on core scientific entities.

USER QUERY: {query}

OPTIMIZED QUERY:"""

CLAIM_EXTRACTION_TEMPLATE = """Extract all independent, scientifically verifiable atomic claims from the following text.
Output MUST be valid JSON in the exact following format:
[
  {{
    "id": "1",
    "text": "Extracted claim text..."
  }}
]
Do not include opinions, connective tissue, or conversational filler.

TEXT: {answer}

JSON CLAIMS:"""

REPAIR_QA_TEMPLATE = """Based strictly on the provided evidence, accurately answer the following question avoiding contradictions and hallucinations.

EVIDENCE:
{context}

QUESTION: {query}

ANSWER:"""

JUDGE_PROMPT_TEMPLATE = """You are a scientific judge evaluating an answer's faithfulness to retrieved evidence.
Review the list of claims made, and the verification results for each claim.
Provide a faithfulness score from 0.0 to 1.0.
State whether repair is needed (Yes/No) and provide a brief reason.

Output your judgment in EXACTLY this format:
FAITHFULNESS_SCORE: [score]
REPAIR_NEEDED: [Yes/No]
REASON: [reason]

CLAIMS MADE:
{claims_str}

VERIFICATION RESULTS:
{results_str}

JUDGMENT:"""


STRICT_EXTRACTION_TEMPLATE = """You are an evidence extraction engine.
Your task is NOT to answer questions.
Your task is ONLY to extract information explicitly present in the provided evidence.

Rules:
1. Never use prior knowledge.
2. Never infer.
3. Never summarize.
4. Never combine information.
5. Never generalize.
6. Never complete partially implied facts.
7. Quote evidence whenever possible.
8. If answer cannot be found explicitly, respond ONLY: INSUFFICIENT EVIDENCE
No additional text. No explanations. No reasoning.

EVIDENCE:
{context}

QUESTION: {query}

ANSWER:"""

GROUNDED_EXTRACTION_TEMPLATE = """You are a grounded scientific assistant.
Answer ONLY using information explicitly present inside the retrieved evidence.

Allowed actions:
• paraphrasing
• combining facts from multiple chunks
• rewording sentences
• short summaries

Forbidden actions:
• external knowledge
• speculation
• assumptions
• unstated implications
• parametric memory

If the answer cannot be constructed solely from the evidence return exactly:
INSUFFICIENT EVIDENCE

Always cite chunk ids if available.

EVIDENCE:
{context}

QUESTION: {query}

ANSWER:"""

class PromptManager:
    """
    Centralizes all prompt engineering templates for the ResearchGuard pipeline.
    Ensures that generated text is heavily grounded in retrieved contexts.
    """
    VERSION = "v6_grounded"

    @classmethod
    def get_version(cls) -> str:
        return cls.VERSION

    def strict_extraction(self, query: str, context: List[RetrievedChunk]) -> str:
        """
        Returns a strict prompt for extracting explicit evidence only.
        """
        context_str = "\n".join([f"[{c.source}] {c.text}" for c in context])
        return STRICT_EXTRACTION_TEMPLATE.format(context=context_str, query=query)
        
    def grounded_extraction(self, query: str, context: List[RetrievedChunk]) -> str:
        """
        Returns a prompt that allows synthesis and paraphrasing strictly bounded by retrieved evidence.
        """
        context_str = "\n".join([f"[Chunk {c.chunk_id}] {c.text}" for c in context])
        return GROUNDED_EXTRACTION_TEMPLATE.format(context=context_str, query=query)
        
    def scientific_qa(self, query: str, context: List[RetrievedChunk]) -> str:
        """
        Returns a prompt for scientific QA grounded in evidence.
        """
        context_str = "\n".join([f"[{c.source}] {c.text}" for c in context])
        return SCIENTIFIC_QA_TEMPLATE.format(context=context_str, query=query)
        
    def query_rewrite(self, query: str) -> str:
        """
        Returns a prompt for rewriting a user query into an optimized semantic search query.
        """
        return QUERY_REWRITE_TEMPLATE.format(query=query)
        
    def repair_qa(self, query: str, context: List[RetrievedChunk]) -> str:
        """
        Returns a strict prompt for rewriting an answer to avoid hallucinations.
        """
        context_str = "\n".join([f"[{c.source}] {c.text}" for c in context])
        return REPAIR_QA_TEMPLATE.format(context=context_str, query=query)
        
    def claim_extraction(self, answer: str) -> str:
        """
        Returns a prompt to extract atomic verifiable claims from a generated answer.
        """
        return CLAIM_EXTRACTION_TEMPLATE.format(answer=answer)
        
    def judge_prompt(self, claims: List[str], verification_results: List[Dict[str, str]]) -> str:
        """
        Returns a prompt to judge if an answer is fully supported by evidence.
        """
        claims_str = "\n".join([f"- {c}" for c in claims])
        results_str = "\n".join([
            f"Claim: {r.get('claim', '')}\nEvidence: {r.get('evidence', '')}\nSupported: {r.get('supported', '')}" 
            for r in verification_results
        ])
        
        return JUDGE_PROMPT_TEMPLATE.format(claims_str=claims_str, results_str=results_str)
