# Phase 6A: Experimental Hallucination Analysis

## 1. Corpus
- **paper1**: LoRA freezes pretrained weights.
- **paper2**: LoRA trains low rank matrices.

## 2. Summary Metrics
| Metric | Value |
|--------|-------|
| Hallucination Rate | 95.00% |
| Repair Rate | 95.00% |
| Average Faithfulness | 0.47 |
| Average Latency | 11.18s |
| Average Claims per Response | 4.75 |

## 3. Results Table
| Query | Faithfulness | Claims (S/N/C) | Repaired | Hallucination |
|-------|--------------|----------------|----------|---------------|
| What is LoRA? | 0.50 | 2 (0/2/0) | Yes | Yes |
| Does LoRA freeze weights? | 0.17 | 3 (0/1/2) | Yes | Yes |
| What does LoRA train? | 1.00 | 1 (1/0/0) | No | No |
| How does LoRA adapt pretrained models? | 0.50 | 2 (0/2/0) | Yes | Yes |
| What matrices are used in LoRA? | 0.50 | 2 (0/2/0) | Yes | Yes |
| Are pretrained weights frozen in LoRA? | 0.75 | 2 (1/1/0) | Yes | Yes |
| Does LoRA use low rank matrices? | 0.50 | 2 (1/0/1) | Yes | Yes |
| Why does LoRA freeze weights? | 0.50 | 2 (0/2/0) | Yes | Yes |
| Does LoRA reduce memory consumption? | 0.50 | 3 (0/3/0) | Yes | Yes |
| Does LoRA accelerate training? | 0.25 | 2 (0/1/1) | Yes | Yes |
| Is LoRA better than full fine-tuning? | 0.75 | 2 (1/1/0) | Yes | Yes |
| What are the benefits of low rank matrices in LoRA? | 0.45 | 29 (0/26/3) | Yes | Yes |
| How is LoRA applied to large language models? | 0.50 | 2 (0/2/0) | Yes | Yes |
| Does LoRA improve inference latency? | 0.43 | 28 (0/24/4) | Yes | Yes |
| Who invented LoRA? | 0.50 | 2 (0/2/0) | Yes | Yes |
| When was LoRA published? | 0.25 | 2 (0/1/1) | Yes | Yes |
| What disease causes asthma? | 0.33 | 3 (0/2/1) | Yes | Yes |
| How do you cook spaghetti? | 0.25 | 2 (0/1/1) | Yes | Yes |
| What is the capital of France? | 0.50 | 2 (0/2/0) | Yes | Yes |
| Who won the 1994 World Cup? | 0.25 | 2 (0/1/1) | Yes | Yes |

## 4. Prompt Investigation
**Why do unsupported claims survive?**
1. **Generator Overconfidence**: The generator (Qwen via Groq) possesses vast latent knowledge about LoRA (e.g., that it reduces memory consumption). Despite strict prompting, it injects parametric knowledge when the retrieved context is too sparse.
2. **Neutral Label Ambiguity**: DeBERTa classifies these claims as NEUTRAL because they don't explicitly contradict the corpus, they just aren't explicitly supported by the two tiny sentences.
3. **Lenient Repair Trigger**: The current pipeline might tolerate some level of neutral claims if the threshold allows it, failing to aggressively trigger regeneration on parametric knowledge leakage.

**Recommendations for Prompt Changes:**
1. Add strict grounding anchors to the Generator prompt: `DO NOT use your internal knowledge. If the context does not contain the answer, explicitly state 'I don't know'.`
2. Add a hallucination warning block: `WARNING: Any statement made about benefits, applications, or history not explicitly detailed in the chunk will be heavily penalized.`
3. Modify the Repair Planner prompt to aggressively target 'NEUTRAL' labels as completely unacceptable failures when strict adherence is required.
