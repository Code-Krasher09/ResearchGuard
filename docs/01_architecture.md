# ResearchGuard Architecture

Version: 1.0

---

# System Overview


ResearchGuard consists of seven major modules.


User Query

↓

Retriever

↓

Generator

↓

Claim Extractor

↓

Verifier

↓

Judge

↓

Repair Module

↓

Final Answer


---

# Architecture Diagram


```text
                     ┌─────────────┐
                     │ User Query  │
                     └──────┬──────┘
                            │
                            ▼
                 ┌──────────────────┐
                 │ Retriever        │
                 │ FAISS + BGE      │
                 └────────┬─────────┘
                          │
                          ▼
              ┌──────────────────────┐
              │ Retrieved Context    │
              └────────┬─────────────┘
                       │
                       ▼
            ┌────────────────────────┐
            │ Generator              │
            │ Groq (Qwen/Llama)      │
            └────────┬───────────────┘
                     │
                     ▼
          ┌──────────────────────────┐
          │ Draft Answer             │
          └────────┬─────────────────┘
                   │
                   ▼
       ┌─────────────────────────────┐
       │ Claim Extractor             │
       └────────┬────────────────────┘
                │
                ▼
      ┌──────────────────────────────┐
      │ Verifier                     │
      │ DeBERTa-MNLI                 │
      └────────┬─────────────────────┘
               │
               ▼
      ┌──────────────────────────────┐
      │ Judge                        │
      │ Faithfulness Assessment      │
      └───────┬─────────────┬────────┘
              │             │
              │             │
              ▼             ▼

         Accepted      Repair Required

                             │

                             ▼

                ┌────────────────────┐
                │ Repair Module      │
                └────────┬───────────┘
                         │
                         ▼
                   Regeneration
```


---

# Components


## Retriever


Responsibilities


Semantic search


Chunk ranking


Similarity computation


Libraries


FAISS


sentence-transformers



Model


BAAI/bge-small-en-v1.5



---

## Generator


Responsibilities


Answer generation


Evidence grounding


Context-aware responses


Provider


Groq



Models


Qwen


Llama


Gemma



---

## Claim Extractor


Responsibilities


Extract factual claims.


Sentence decomposition.


Methods


SpaCy


LLM


Regex



---

## Verifier


Responsibilities


Determine evidence support.


Model


DeBERTa-v3


MNLI



Outputs


Support


Neutral


Contradiction



---

## Judge


Responsibilities


Faithfulness scoring.


Confidence assessment.


Thresholding.


Decision making.


---

## Repair Module


Responsibilities


Diagnose failures.


Improve retrieval.


Regenerate answers.


Strategies


Increase k


Rewrite query


Hybrid retrieval


Re-ranking



---

# Design Decisions


Dense Retrieval


Chosen over BM25


Reason:


Better semantic understanding.


---

Pure Python


Chosen over LangChain.


Reason:


Transparency


Interview explainability


Less abstraction


---

Groq


Chosen because:


Free


Fast


No local inference


---