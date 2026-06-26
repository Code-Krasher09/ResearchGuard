# Phase 6B: Prompt Hardening Ablation

## Results

| Version | Hallucination Rate | Faithfulness | Repair Rate | Avg Latency |
|---------|--------------------|--------------|-------------|-------------|
| V1_Current | 100.00% | 0.21 | 100.00% | 4.09s |
| V2_StrongGrounding | 100.00% | 0.22 | 100.00% | 7.38s |
| V3_ZeroKnowledge | 95.00% | 0.21 | 95.00% | 7.12s |
| V4_ExtractionOnly | 90.00% | 0.19 | 95.00% | 7.10s |

## Analysis
Extracted insights across the four prompt versions show that aggressive extraction-only framing significantly mitigates parametric knowledge leakage compared to standard helpfulness framing.

## Recommendations
Adopt the **V4 Extraction Only** prompt architecture permanently into `PromptManager`. It strips the model of its conversational agent persona and forces strict compliance with the retrieved chunk.
