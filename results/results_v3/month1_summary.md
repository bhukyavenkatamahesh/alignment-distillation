# Month 1 Summary: Alignment Distillation Prototype

## Completed Scope

- Set up teacher-student response distillation pipeline.
- Compared Base TinyLlama, Direct SFT, LoRA SFT, and Response Distillation.
- Prepared English held-out evaluation and multilingual QA artifacts.
- Implemented ROUGE-L, factuality proxy, hallucination proxy, instruction-following, explicit factual accuracy, and explicit hallucination tests.
- Added leakage checks for SFT data and teacher-response distillation data.

## Key Held-Out Results

| Approach | ROUGE-L | Factuality Proxy | Hallucination Proxy | Instruction Following |
|---|---:|---:|---:|---:|
| Base Student | 0.392 | 0.761 | 0.200 | 0.935 |
| Direct SFT | 0.371 | 0.556 | 0.155 | 0.733 |
| LoRA SFT | 0.444 | 0.722 | 0.150 | 0.881 |
| Response Distillation | 0.350 | 0.865 | 0.223 | 0.962 |

## Explicit Benchmark Results

| Approach | Factual Accuracy | Hallucination Rate |
|---|---:|---:|
| Base Student | 0.500 | 1.000 |
| Direct SFT | 0.500 | 0.667 |
| LoRA SFT | 0.375 | 1.000 |
| Response Distillation | 0.625 | 0.333 |

## Interpretation

The Month 1 prototype shows partial trust transfer. Response distillation changes held-out factuality proxy from 0.761 to 0.865, while hallucination proxy changes from 0.200 to 0.223. On explicit tests, factual accuracy changes from 0.500 to 0.625, and explicit hallucination rate changes from 1.000 to 0.333.

Conclusion: the framework is complete for Month 1, but alignment transfer is not uniformly reliable. Month 2 should focus on stronger hallucination benchmarks, multilingual robustness, retrieval grounding, response consistency, and efficiency-quality trade-off analysis.
