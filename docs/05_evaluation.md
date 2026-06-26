# ResearchGuard Evaluation Specification

Version: 1.0
Status: Frozen
Last Updated: 25 June 2026

---

# Purpose

This document defines the evaluation framework used to assess the
performance of ResearchGuard.

Goals:

• Evaluate retrieval quality
• Measure faithfulness
• Quantify hallucinations
• Assess repair effectiveness
• Benchmark against baseline RAG

---

# Evaluation Philosophy

Traditional RAG evaluation is often limited to answer correctness.

ResearchGuard evaluates four major aspects:

1. Retrieval

2. Verification

3. Generation

4. Repair


---

# Evaluation Datasets


## Primary Dataset


SciFact



Reason:

Scientific claims


Evidence labels


Fact verification


Compact dataset


Suitable for local experiments




---

## Secondary Dataset


FEVER


Optional


General-domain claims



---

## Tertiary Dataset


HotpotQA


Optional


Multi-hop retrieval




---

# Baseline


ResearchGuard should always be compared against a standard RAG pipeline.


Baseline architecture


Query

↓

Retriever

↓

Generator

↓

Answer



No verification


No repair



---

# Retrieval Metrics


## Recall@K


Measures whether relevant evidence appears in top-k retrieved chunks.



Formula


Recall@K


=

Relevant Retrieved


───────────────


Relevant Total




---

Target


Recall@5 > 0.80



---

## Precision@K


Measures proportion of useful chunks.



Target


Precision@5 > 0.70



---

## MRR


Mean Reciprocal Rank



Measures ranking quality.



Formula


MRR


=


Σ 1/rank


────────


N




---

Target


MRR > 0.75



---

# Verification Metrics


Verification is evaluated independently.



---

## Claim Accuracy



Percentage of correctly classified claims.



Formula



Claim Accuracy


=


Correct Predictions


──────────────


Total Claims




---

Target


>90%



---

## Support Ratio



Measures amount of supported claims.



Formula



Supported


─────────


Total Claims




---

Target


>0.85



---

## Contradiction Rate



Formula



Contradicted


────────────


Total Claims




---

Target


<0.05



---

# Generation Metrics


Generated responses are evaluated using RAGAS.


---

## Faithfulness


Measures whether generated answer is supported by retrieved context.



Target


>0.85



---

## Context Precision



Measures relevance of retrieved context.



Target


>0.80



---

## Context Recall



Measures completeness.



Target


>0.80



---

## Answer Relevancy



Measures usefulness.



Target


>0.85



---

# Hallucination Metrics


Custom metric.


---

Hallucination Score



Formula



Supported Claims


────────────────


Total Claims




---

Example



Supported


4



Neutral


1



Contradicted


0



Hallucination score



0.80



---

Target


>0.85



---

# Repair Metrics


ResearchGuard introduces repair mechanisms.



These need separate evaluation.


---

## Repair Success Rate



Formula



Successful Repairs


────────────────


Repairs Attempted




---

Target


>70%



---

## Faithfulness Improvement



Formula



Final


-


Initial




---

Example



Initial


0.72



Final


0.89



Improvement


0.17



---

Target


Positive improvement



---

## Retrieval Improvement



Example



Recall@5


0.72



Recall@15


0.84



---

# Latency Metrics



Average response time.



Measure


Retrieval


Generation


Verification


Repair




---

Target


<10 seconds



---

# Experiments


Experiment 1


Baseline vs ResearchGuard



---

Experiment 2


Adaptive K



K=5


K=10


K=15




---

Experiment 3


Query rewriting



Enabled


Disabled




---

Experiment 4


Different embedding models



MiniLM


BGE


E5




---

Experiment 5


Different NLI models



DeBERTa


RoBERTa


BART




---

# Expected Results


| Metric | Baseline | ResearchGuard |
|-------|---------|---------------|
| Recall@5 | 0.72 | 0.84 |
| Faithfulness | 0.78 | 0.91 |
| Support Ratio | 0.80 | 0.93 |
| Hallucination Score | 0.76 | 0.90 |
| Repair Success | - | 0.74 |


---

# Libraries


RAGAS


DeepEval


Scikit-learn


NumPy



---

# Deliverables


evaluation.ipynb


metrics.json


experiment_logs/


plots/


tables/


---
