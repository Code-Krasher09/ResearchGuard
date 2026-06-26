# ResearchGuard Schema Specification

Version: 1.0
Status: Draft
Last Updated: 25 June 2026

---

# Purpose

This document defines the internal data structures used throughout
ResearchGuard.

Goals:

- Standardize communication between modules
- Enable easier testing
- Simplify serialization
- Improve maintainability
- Assist code generation agents


---

# Schema Overview


Pipeline Objects


Query
│
├── RetrievedChunk[]
│
├── GeneratedAnswer
│
├── Claim[]
│
├── VerificationResult[]
│
├── JudgeDecision
│
└── EvaluationResult



---

# Query


Represents user input.


Suggested implementation:


Pydantic BaseModel


Example


```python
class Query(BaseModel):

    id:str

    text:str

    timestamp:str

    metadata:dict = {}
```


---

Fields


id


Unique identifier


UUID preferred



text


User question



timestamp


ISO format



metadata


Optional




Example


```json
{
"id":"q001",

"text":"How does LoRA reduce memory?",

"timestamp":"2026-06-25T19:32:00"
}
```



---

# RetrievedChunk


Retrieved scientific evidence.


---

Schema


```python
class RetrievedChunk(BaseModel):

    chunk_id:str

    text:str

    source:str

    score:float

    embedding:Optional[List[float]]
```



---

Fields


chunk_id


Unique id



text


Chunk content



source


Paper name



score


Similarity



embedding


Optional




---

Example


```json
{

"chunk_id":"c102",


"text":"LoRA freezes pretrained weights...",


"source":"lora_paper",


"score":0.92

}
```



---

# GeneratedAnswer


Represents draft generation.


Schema


```python
class GeneratedAnswer(BaseModel):


answer:str


model:str


latency:float


tokens:int


}
```




---

Example


```json
{

"answer":"LoRA reduces memory...",


"model":"llama3-8b",


"latency":1.2,


"tokens":164

}
```




---

# Claim


Atomic factual statement.


---

Schema


```python
class Claim(BaseModel):


id:str


text:str


position:int

}
```



---

Example


```json
{

"id":"claim_01",


"text":"LoRA freezes pretrained weights.",


"position":0

}
```




---

# VerificationResult


Stores verification outcome.


---

Schema


```python
class VerificationResult(BaseModel):


claim_id:str


claim:str


evidence:str


label:str


confidence:float

}
```



---

Allowed labels


SUPPORTED


NEUTRAL


CONTRADICTED




---

Example


```json
{


"claim":"LoRA introduced in 2022",


"label":"CONTRADICTED",


"confidence":0.91


}
```




---

# JudgeDecision


Judge output.


---

Schema


```python
class JudgeDecision(BaseModel):


faithfulness:float


repair_needed:bool


reason:str

}
```



---

Example


```json
{


"faithfulness":0.74,


"repair_needed":true,


"reason":"Unsupported claims"


}
```




---

# RepairAttempt


Tracks repair iterations.


---

Schema


```python
class RepairAttempt(BaseModel):


attempt:int


strategy:str


success:bool


old_score:float


new_score:float

}
```




---

Example


```json
{


"attempt":1,


"strategy":"increase_k",


"success":True,


"old_score":0.71,


"new_score":0.88


}
```




---

# EvaluationResult


Stores metrics.


---

Schema


```python
class EvaluationResult(BaseModel):


faithfulness:float


context_precision:float


context_recall:float


answer_relevancy:float


retrieval_recall:float

}
```



---

Example


```json
{


"faithfulness":0.91,


"context_precision":0.87,


"answer_relevancy":0.93

}
```




---

# Logging Schema


Suggested format


```json
{


"query_id":"q01",


"stage":"verification",


"timestamp":"...",


"message":"Verification complete"

}
```



---

# Serialization


Preferred


JSON


Pydantic


msgpack optional



---

# Notes


Embedding vectors should NOT be stored permanently.


Only cache locally.


Avoid serializing large tensors.


---