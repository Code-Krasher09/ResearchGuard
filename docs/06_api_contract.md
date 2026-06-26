# ResearchGuard API Contract

Version: 1.0
Status: Frozen

---

# Purpose


Defines interfaces between modules.


Goals


Loose coupling


Easy testing


Agent friendliness


Clear ownership



---

# Pipeline


Query


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

Repair


↓

Response



---

# Retriever API


File


retriever.py



---

Function


```python

retrieve(
    query:str,
    k:int=5
) -> list[RetrievedChunk]

```



---

Inputs



query


User query



k


Number of chunks




---

Output



List[RetrievedChunk]



---

Example



```python

chunks = retrieve(
    "What is LoRA?",
    k=5
)

```



---

# Generator API


File


generator.py



---

Function



```python

generate_answer(

query:str,

context:list[RetrievedChunk]

) -> GeneratedAnswer

```



---

Example



```python

answer = generate_answer(

query,

chunks

)

```



---

Output



GeneratedAnswer



---

# Claim Extractor API


File


claims.py



---

Function



```python

extract_claims(

answer:str

) -> list[Claim]

```



---

Example



```python

claims = extract_claims(answer)

```



---

Output



List[Claim]



---

# Verification API


File


verifier.py



---

Function



```python

verify_claim(

claim:Claim,

evidence:list[RetrievedChunk]

) -> VerificationResult

```



---

Output



VerificationResult



---

Example



```python

result = verify_claim(

claim,

chunks

)

```



---

# Batch Verification


Function



```python

verify_all(

claims:list[Claim],

evidence:list[RetrievedChunk]

)

-> list[VerificationResult]

```



---

# Judge API


File


judge.py



---

Function



```python

judge(

results:list[VerificationResult]

)

-> JudgeDecision

```



---

Example



```python

decision = judge(results)

```



---

Output



JudgeDecision



---

# Repair API


File


repair.py



---

Function



```python

repair(

query:str,

decision:JudgeDecision,

attempt:int

)

-> tuple[str,int]

```



---

Returns



new_query


new_k



---

Strategies



increase_k



query_rewrite



hybrid



rerank



---

# Evaluation API


File


evaluate.py



---

Function



```python

evaluate(

answer:str,

context:list,

ground_truth:str

)

-> EvaluationResult

```



---

Output



EvaluationResult



---

# Pipeline API


File


pipeline.py



---

Main Function



```python

run_pipeline(

query:str,

max_repairs:int=2

)

```



---

Returns



```python

PipelineOutput


```



---

Suggested Schema



```python


class PipelineOutput:


    query:str


    answer:str


    faithfulness:float


    claims_supported:int


    repair_attempts:int


    latency:float


```



---

# Logging API


File


logger.py



---

Function



```python

log_stage(

stage:str,

payload:dict

)

```



---

Stages



retrieval


generation


claims


verification


judge


repair


evaluation



---

# Environment Variables



.env



```bash

GROQ_API_KEY=

MODEL_NAME=

EMBED_MODEL=

FAITHFULNESS_THRESHOLD=0.8

MAX_REPAIRS=2

```



---

# Testing Requirements


Every public function


must have



Unit Tests


Type Hints


Docstrings


Pytest Coverage



Target



Coverage > 85%



---

# Frozen Decisions


Retriever


FAISS



Generator


Groq



Verifier


DeBERTa



Evaluation


RAGAS



Framework


Pure Python



Status


FROZEN


---