# Phase 6D: Error Attribution Analysis

## Objective
Analyze the 14 failed queries from the Phase 6C-3 evaluation (LoRA Corpus) to determine the root cause of the recorded "hallucinations" and "repair failures." 

## Error Classification Categories
- **A**: Parametric leakage (External facts injected)
- **B**: Over-generalization (Generator inferred beyond evidence)
- **C**: Verifier false negative (Generator was correct, Verifier failed to ground it)
- **D**: Claim extraction issue (Non-factual statements extracted as claims)
- **E**: Prompt issue

## Analysis Summary

| Failure Type | Count | Percentage |
|--------------|-------|------------|
| **A: Parametric Leakage** | 2 | 14.3% |
| **B: Over-generalization** | 5 | 35.7% |
| **C: Verifier False Negative** | 2 | 14.3% |
| **D: Claim Extraction Issue** | 5 | 35.7% |
| **E: Prompt Issue** | 0 | 0.0% |

### Category Breakdown

#### 1. Category D: Claim Extraction Issue (5 cases)
In **all unanswerable queries**, the Generator correctly declined to answer (e.g., "I do not have enough information to answer the question"). However, the `ClaimExtractor` inappropriately extracted these refusal statements as factual claims. The `Verifier` then evaluated statements like "The text does not mention asthma" against the LoRA paper. Since the paper doesn't explicitly say "I am not about asthma", the Verifier labeled these as `CONTRADICTED` or `NEUTRAL`, triggering false hallucination penalties.
*Affected Queries:*
- What disease causes asthma?
- How do you cook spaghetti?
- What is the capital of France?
- Who won the 1994 World Cup?
- Who invented LoRA?

#### 2. Category B: Over-generalization (5 cases)
The Generator made inductive leaps based on the text rather than strictly quoting it. For example, when the text mentioned setting $r$ to recover full fine-tuning, the Generator inferred that LoRA "does not necessarily freeze weights" or "trains all biases." While logically plausible, these statements could not be strictly grounded by the Verifier.
*Affected Queries:*
- Does LoRA freeze weights?
- What does LoRA train?
- How does LoRA adapt pretrained models?
- Why does LoRA freeze weights?
- Does LoRA reduce memory consumption?

#### 3. Category A: Parametric Leakage (2 cases)
The Generator outright leaked external knowledge not present in the retrieved chunks.
*Affected Queries:*
- *How is LoRA applied to large language models?* (Hallucinated that it is applied to `Wq` and `Wv` matrices, which was not in the immediate context).
- *When was LoRA published?* (Hallucinated "published in 2019" and "Alec Radford").

#### 4. Category C: Verifier False Negative (2 cases)
The Generator correctly answered the question, but the Verifier failed to properly align the claim with the evidence, marking it as `NEUTRAL` or `CONTRADICTED` due to semantic mismatch.
*Affected Queries:*
- Does LoRA use low rank matrices?
- Is LoRA better than full fine-tuning?

## Conclusion
A significant portion of the "70% Hallucination Rate" from Phase 6C-3 is actually an artifact of the evaluation pipeline—specifically the **Claim Extractor penalizing safe refusals** (35.7%) and **Verifier False Negatives** (14.3%). However, true generator errors (Parametric Leakage and Over-generalization) still account for exactly **50%** of the failures. Both the Generator prompt and the Claim Extractor require hardening.
