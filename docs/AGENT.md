# AGENT.md

Project: ResearchGuard

Version: 1.0

Status: Active


---

# Mission


Build a self-correcting scientific Retrieval-Augmented Generation system.


The system should:

Retrieve evidence

Generate answers

Extract claims

Verify claims

Detect hallucinations

Repair failures

Evaluate improvements



---

# Core Principles


Prefer simplicity.


Prefer explicit code.


Prefer modular design.


Avoid abstractions.


Everything should be explainable in interviews.



---

# Design Constraints


Budget

₹0



Development Time

~2 weeks



GPU

RTX 4060 Laptop



Framework

Pure Python



Target Audience

Researchers

Students

Interviewers



---

# Forbidden Technologies


DO NOT USE


LangChain


CrewAI


AutoGen


Haystack


LlamaIndex


DSPy


OpenAI Assistants API



Reason


Excessive abstraction.


Harder to explain.


Interview disadvantage.



---

# Approved Stack


Embeddings


sentence-transformers


BAAI/bge-small-en-v1.5



Retriever


FAISS



Generator


Groq API



Models


llama3


qwen


gemma



Verifier


MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli



Claim Extraction


SpaCy



Evaluation


RAGAS



UI


Streamlit



Logging


loguru



Testing


pytest



Schemas


Pydantic v2



---

# Coding Standards


Use Python 3.11.


Type hints mandatory.


Docstrings mandatory.


No function >100 LOC.


No class >300 LOC.


Prefer composition.


Prefer dataclasses or Pydantic.


Avoid globals.


No magic numbers.


Use constants.


Use logging.



---

# Testing Requirements


Every public function


must have


Unit Tests


Edge Cases


Happy Path Tests



Coverage Goal


85%+



---

# Folder Structure


ResearchGuard/


docs/


src/


retrieval/


generation/


verification/


repair/


evaluation/


schemas/


utils/


tests/


app/


scripts/


data/


logs/


README.md


AGENT.md



---

# Development Order


Phase 1


Retriever



Phase 2


Generator



Phase 3


Claims



Phase 4


Verification



Phase 5


Judge



Phase 6


Repair



Phase 7


Evaluation



Phase 8


UI



Phase 9


Documentation



---

# Evaluation Targets


Faithfulness


>0.85



Support Ratio


>0.90



Recall@5


>0.80



Repair Success


>70%



Latency


<10 sec



---

# Repair Strategies


Allowed


Increase k


Query rewrite


Hybrid retrieval


Re-ranking



Not Allowed


Fine-tuning


RLHF


LoRA


Complex agents



---

# Prompting Rules


Answers should only use retrieved evidence.


Never fabricate citations.


If evidence is insufficient,


say so.



---

# Logging


Log every stage.


retrieval


generation


claims


verification


judge


repair


evaluation



Use JSON logs.



---

# Code Generation Instructions


Prefer maintainable code.


Prefer readability.


Prefer interview explainability.


When uncertain,


choose the simplest implementation.


Never sacrifice explainability for novelty.



---

# Final Goal


ResearchGuard should be implementable, understandable, explainable, and demo-ready within two weeks.


A successful project should allow the developer to confidently explain:


- RAG
- Embeddings
- FAISS
- NLI
- Hallucination Detection
- Faithfulness Metrics
- Adaptive Retrieval
- Scientific NLP
- Evaluation Methodologies
- Self-Correcting AI Systems


without relying on opaque frameworks.


Status: FROZEN
Date: 25 June 2026