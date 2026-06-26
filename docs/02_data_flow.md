# ResearchGuard Data Flow

Version: 1.0

---

# Pipeline Overview


ResearchGuard follows an iterative retrieval-verification-repair loop.


---

# Stage 1

User Query


Input


```json
{
    "query": "How does LoRA reduce memory consumption?"
}
```



---

# Stage 2


Retrieval



Process


Embedding query


Similarity search


Retrieve top-k chunks



Output


```json
{
    "chunks":[

        {
            "id":"001",
            "text":"LoRA freezes pretrained weights...",
            "score":0.92
        },

        {
            "id":"002",
            "text":"Low-rank matrices reduce trainable parameters...",
            "score":0.88
        }

    ]
}
```



---

# Stage 3


Generation



Input


Query


Retrieved context



Output


```json
{
    "draft_answer":

"LoRA reduces memory consumption by training low-rank adaptation matrices while keeping pretrained parameters frozen."

}
```



---

# Stage 4


Claim Extraction



Input


Draft answer



Output


```json
{

"claims":[

"LoRA reduces memory consumption.",

"LoRA freezes pretrained weights.",

"LoRA trains low-rank matrices."

]

}
```



---

# Stage 5


Verification



Input


Claim


Evidence



Process


NLI inference



Output


```json
{

"claim":

"LoRA freezes pretrained weights.",


"status":"supported",


"confidence":0.94


}
```



---

# Stage 6


Faithfulness Assessment



Inputs


Verification results



Example


```json
{

"supported":3,


"contradicted":0,


"neutral":1


}
```



Score


```json
{

"faithfulness":0.75


}
```



---

# Stage 7


Judge



Decision logic



If


Faithfulness > 0.8



Return answer



Else


Repair



---

Output


```json
{

"repair_needed":true

}
```



---

# Stage 8


Repair



Strategies



Increase Retrieval Depth


k = 5


↓

k = 15



---

Query Rewrite


Original


LoRA



Rewritten


Explain low-rank adaptation in transformer fine-tuning.



---

Hybrid Retrieval


Dense


+

BM25



---

# Stage 9


Regeneration



Updated context


New answer



---

# Stage 10


Final Response



Output


```json
{

"answer":"...",


"faithfulness":0.91,


"claims_supported":4,


"repair_attempts":1


}
```



---

# Failure Modes


## Retrieval Failure


Wrong chunks


Missing chunks


Low similarity



---

## Verification Failure


Low confidence


Contradictions



---

## Generation Failure


Hallucination


Unsupported claims



---

# Logging


Each stage should log:


Query


Retrieved chunks


Generated answer


Claims


Verification results


Faithfulness score


Repair attempts


Latency



---