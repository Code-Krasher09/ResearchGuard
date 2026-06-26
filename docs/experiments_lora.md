# Phase 6C-3: Experimental Hallucination Analysis on Rich Corpus

## 1. Corpus Comparison
| Metric | Synthetic Corpus (Phase 6A) | LoRA Corpus (Phase 6C-3) | Delta |
|--------|-----------------------------|--------------------------|-------|
| Hallucination Rate | 95.00% | 10.00% | -85.00% |
| Repair Rate | 95.00% | 25.00% | -70.00% |
| Avg Faithfulness | 0.28 | 0.97 | +0.69 |
| Avg Latency | 4.45s | 8.59s | +4.14s |
| Avg Claims | 2.35 | 1.30 | -1.05 |

## 2. Results Table
| Query | Faithfulness | Claims (S/N/C) | Repaired | Hallucination |
|-------|--------------|----------------|----------|---------------|
| What is LoRA? | 1.00 | 2 (2/0/0) | No | No |
| Does LoRA freeze weights? | 1.00 | 1 (0/0/0) | No | No |
| What does LoRA train? | 1.00 | 1 (1/0/0) | No | No |
| How does LoRA adapt pretrained models? | 0.50 | 1 (0/1/0) | Yes | Yes |
| What matrices are used in LoRA? | 1.00 | 1 (1/0/0) | No | No |
| Are pretrained weights frozen in LoRA? | 1.00 | 1 (0/0/0) | Yes | No |
| Does LoRA use low rank matrices? | 0.83 | 3 (2/1/0) | Yes | Yes |
| Why does LoRA freeze weights? | 1.00 | 1 (0/0/0) | No | No |
| Does LoRA reduce memory consumption? | 1.00 | 1 (0/0/0) | No | No |
| Does LoRA accelerate training? | 1.00 | 3 (3/0/0) | Yes | No |
| Is LoRA better than full fine-tuning? | 1.00 | 1 (0/0/0) | No | No |
| What are the benefits of low rank matrices in LoRA? | 1.00 | 2 (2/0/0) | No | No |
| How is LoRA applied to large language models? | 1.00 | 1 (1/0/0) | Yes | No |
| Does LoRA improve inference latency? | 1.00 | 1 (1/0/0) | No | No |
| Who invented LoRA? | 1.00 | 1 (0/0/0) | No | No |
| When was LoRA published? | 1.00 | 1 (0/0/0) | No | No |
| What disease causes asthma? | 1.00 | 1 (0/0/0) | No | No |
| How do you cook spaghetti? | 1.00 | 1 (0/0/0) | No | No |
| What is the capital of France? | 1.00 | 1 (0/0/0) | No | No |
| Who won the 1994 World Cup? | 1.00 | 1 (0/0/0) | No | No |
