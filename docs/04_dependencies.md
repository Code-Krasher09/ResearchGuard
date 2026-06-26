# ResearchGuard Dependency Specification


Version: 1.0


---

# Purpose


Defines approved dependencies.


Goals:


Reproducibility


Lightweight stack


Agent compatibility


Easy installation


---

# Python Version


Recommended


Python 3.11


Tested


3.11.x


Avoid


3.13


---

# Core ML Libraries


## torch


Purpose


Model inference


Version


>=2.7




---

## transformers


Purpose


NLI


DeBERTa


Version


>=4.50




---

## sentence-transformers


Purpose


Embeddings


Version


>=3.0




---

## datasets


Purpose


SciFact


FEVER


Version


latest




---

# Retrieval


## faiss-cpu


Purpose


Vector search



Version


latest



---

Alternative


faiss-gpu



Optional




---

# NLP


## spacy


Purpose


Sentence splitting


Claim extraction



---

Model


en_core_web_sm




Installation


```bash
python -m spacy download en_core_web_sm
```



---

## nltk


Optional



Tokenization




---

# Verification


Uses


transformers



Models


MoritzLaurer/


DeBERTa-v3-base-mnli-fever-anli



---

# LLM


## groq


Purpose


Generation



Installation


```bash
pip install groq
```



---

Environment variable


```bash
GROQ_API_KEY=
```



---

Models


llama-3-8b


qwen-72b


gemma




---

# Evaluation


## ragas


Purpose


Faithfulness



Context precision



Answer relevancy




---

Installation


```bash
pip install ragas
```



---

## deepeval


Optional



Hallucination metrics




---

# Web Interface


## streamlit


Purpose


Demo UI



---

Installation


```bash
pip install streamlit
```



---

Run


```bash
streamlit run app.py
```



---

# API


## pydantic


Purpose


Schemas



Version


>=2




---

## python-dotenv


Purpose


Environment management




---

# Utilities


## tqdm


Progress bars



---

## rich


Logging



Pretty CLI




---

## loguru


Structured logging




---

# Testing


## pytest


Required



---

## pytest-cov


Coverage



---

## hypothesis


Optional




---

# Development


## black


Formatter



---

## isort


Import sorting




---

## mypy


Type checking




---

## pre-commit


Hooks




---

# Recommended requirements.txt


```text

torch

transformers

sentence-transformers

datasets

faiss-cpu

spacy

groq

ragas

streamlit

pydantic

python-dotenv

rich

loguru

pytest

pytest-cov

black

isort

mypy

tqdm

```



---

# Installation


```bash

git clone repo


cd ResearchGuard


python -m venv .venv


source .venv/bin/activate


pip install -r requirements.txt


python -m spacy download en_core_web_sm


```



---

# Frozen Decisions


Retriever


FAISS


Embeddings


BGE-small-v1.5


Generator


Groq


Verifier


DeBERTa-MNLI


Evaluation


RAGAS


UI


Streamlit


Framework


Pure Python



---

# Non-goals


LangChain


CrewAI


AutoGen


Haystack


LlamaIndex



Reason


Prefer explicit implementations


Improve interview explainability


Reduce abstraction layers


---