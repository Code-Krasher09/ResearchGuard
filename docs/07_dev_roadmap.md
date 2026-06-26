# ResearchGuard Development Roadmap

Version: 1.0
Status: Frozen
Last Updated: 25 June 2026

---

# Purpose

This document defines the implementation plan for ResearchGuard.

Goals:

- Prevent scope creep
- Track progress
- Define milestones
- Enable autonomous coding agents
- Freeze MVP requirements


---

# Development Philosophy

ResearchGuard should be developed incrementally.

Each phase must:

- Pass unit tests
- Be independently executable
- Produce measurable outputs
- Avoid introducing unnecessary abstractions


---

# Milestones


Phase 1

Retriever


Phase 2

Generator


Phase 3

Claim Extraction


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

Polish


---

# Phase 1

Retriever


Estimated Duration

1 Day


Status

Pending


---

Objectives


Build embedding pipeline


Build FAISS index


Implement retrieval


Add tests



---

Tasks


Install dependencies


Load embedding model


Build chunker


Build index


Search top-k


Store metadata


Unit tests



---

Deliverables


retriever.py


chunker.py


index.faiss


tests/test_retriever.py



---

Success Criteria


Top-k retrieval works


Latency < 500 ms


Coverage > 85%


---

# Phase 2

Generator


Estimated Duration

1 Day


---

Objectives


Connect Groq


Generate grounded answers



---

Tasks


API wrapper


Prompt template


Inference


Output parser


Tests



---

Deliverables


generator.py


prompts.py


tests/test_generator.py



---

Success Criteria


Answer generation works


Average latency < 3 sec



---

# Phase 3

Claim Extraction


Estimated Duration

1 Day


---

Objectives


Extract factual claims


Support sentence splitting



---

Tasks


SpaCy pipeline


Sentence extraction


Filtering


Optional LLM mode



---

Deliverables


claims.py


tests/test_claims.py



---

Success Criteria


Claims extracted correctly


Accuracy > 90%


---

# Phase 4

Verification


Estimated Duration

1 Day


---

Objectives


Support NLI verification



---

Tasks


Load model


Batch inference


Label mapping


Confidence scoring


Tests



---

Deliverables


verifier.py


tests/test_verifier.py



---

Success Criteria


Entailment works


Batch verification works


---

# Phase 5

Judge


Estimated Duration

0.5 Day


---

Objectives


Decision making


Faithfulness scoring



---

Tasks


Threshold logic


Metrics


Decision schema



---

Deliverables


judge.py


tests/test_judge.py



---

Success Criteria


Correct repair decisions



---

# Phase 6

Repair


Estimated Duration

1 Day


---

Objectives


Improve low-confidence answers



---

Strategies


Increase k


Rewrite query


Hybrid retrieval



---

Tasks


Adaptive k


Query rewrite


Looping


Stopping criteria



---

Deliverables


repair.py


tests/test_repair.py



---

Success Criteria


Faithfulness improves



---

# Phase 7

Evaluation


Estimated Duration

2 Days


---

Objectives


Evaluate system



---

Tasks


SciFact


RAGAS


Metrics


Plots


Comparison



---

Deliverables


evaluate.py


evaluation.ipynb


metrics.json



---

Success Criteria


Baseline comparison complete



---

# Phase 8

Streamlit UI


Estimated Duration

1 Day


---

Objectives


Interactive demo



---

Features


Question input


Answer


Evidence


Claims


Faithfulness


Repair attempts



---

Deliverables


app.py



---

# Phase 9

Polish


Estimated Duration

1 Day


---

Objectives


Open source readiness



---

Tasks


README


Architecture figure


Screenshots


Badges


Examples


Refactor


Coverage



---

# Frozen Scope


Included


FAISS


Groq


SciFact


RAGAS


DeBERTa



---

Excluded


LangChain


CrewAI


Neo4j


Fine-tuning


Agents


Knowledge Graphs


GraphRAG



---

# MVP Checklist


[ ] Retriever

[ ] Generator

[ ] Claims

[ ] Verification

[ ] Judge

[ ] Repair

[ ] Evaluation

[ ] UI

[ ] README

[ ] Tests

[ ] Documentation


---