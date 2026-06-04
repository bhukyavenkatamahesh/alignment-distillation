# Alignment Distillation and Trustworthiness Evaluation in Small Language Models

**Student:** Venkata Mahesh &nbsp;|&nbsp; **Supervisor:** Dr. Kanica &nbsp;|&nbsp; **Status:** Month 1 Complete

> Investigating whether trustworthiness, factuality, and safety behaviours can be transferred from a large teacher model to a compact 1.1B student model using supervised fine-tuning and distillation techniques.

---

## Table of Contents
1. [Project Objective](#1-project-objective)
2. [Architecture Overview](#2-architecture-overview)
3. [Models](#3-models)
4. [Methods Implemented](#4-methods-implemented)
5. [Dataset](#5-dataset)
6. [Evaluation Framework](#6-evaluation-framework)
7. [Results](#7-results)
8. [Model Efficiency Profile](#8-model-efficiency-profile)
9. [Key Findings](#9-key-findings)
10. [Current Limitations](#10-current-limitations)
11. [Next Steps — Month 2](#11-next-steps--month-2)
12. [Project Structure](#12-project-structure)
13. [Setup](#13-setup)

---

## 1. Project Objective

Small Language Models (≤ 1.1B parameters) are attractive for on-device and low-resource inference. However, they lack the alignment and factual robustness of large frontier models. This project rigorously evaluates four fine-tuning strategies to transfer **alignment** from Llama 3.1 8B to TinyLlama 1.1B, measuring the impact on:

- **Factuality** (TruthfulQA MC benchmark + explicit factuality probes)
- **Hallucination resistance** (unanswerable-question probes)
- **Free-text quality** (ROUGE-L, entropy/repetition proxy)
- **Instruction-following** (keyword overlap proxy)
- **Response consistency** (stochastic pairwise ROUGE-L)
- **Multilingual transfer** (EN, HI, DE, FR, ZH subsets)

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                     DATA PREPARATION                             │
│  Alpaca (300) + TruthfulQA (100) + Anchor Samples (20)          │
│       ↓ Human SFT Dataset          ↓ Teacher Inference           │
│                           Llama 3.1 8B (Groq API)               │
│                               ↓ Teacher responses               │
└──────────┬───────────────────────────────────────────┬──────────┘
           │                                           │
           ▼                                           ▼
  ┌─────────────────┐                      ┌────────────────────┐
  │  Human SFT Data │                      │ Teacher-Gen Data   │
  └────────┬────────┘                      └─────────┬──────────┘
           │                                         │
           ▼                                         ▼
┌──────────────────────────────────────────────────────────────────┐
│               STUDENT MODEL: TinyLlama 1.1B Chat                 │
│                                                                  │
│  EXP001: Base Baseline  (no training)                            │
│  EXP002: LoRA SFT       (human data, PEFT adapters)             │
│  EXP003: Response Distil (teacher-generated text, PEFT)         │
│  EXP004: Direct SFT     (human data, full parameters)           │
│  EXP005: KL Distillation (logit-level, OpenLlama-3B teacher)    │
└──────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    EVALUATION PIPELINE                           │
│                                                                  │
│  ① TruthfulQA MC1 & MC2  (standard benchmark, 200 questions)    │
│  ② Explicit Factuality    (12-item factual accuracy probe)       │
│  ③ Explicit Hallucination (10 unanswerable question probe)       │
│  ④ Proxy: ROUGE-L, Factuality, Hallucination, Instr-Following   │
│  ⑤ Response Consistency   (pairwise ROUGE-L, 3 runs)            │
│  ⑥ Multilingual           (HI, DE, FR, ZH sample subsets)       │
│  ⑦ Model Efficiency       (params, GPU memory, tokens/sec)       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Models

| Role | Model | Parameters | Notes |
|---|---|---|---|
| **Student** | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1B | Primary model being aligned |
| **Teacher (Response)** | Llama 3.1 8B via **Groq API** | 8B | Generates aligned training responses |
| **Teacher (KL)** | openlm-research/open_llama_3b | 3B | Same LlamaTokenizer (32K vocab), logit-level KD |

---

## 4. Methods Implemented

| ID | Method | Description |
|---|---|---|
| **EXP001** | **Baseline** | Unmodified TinyLlama-1.1B-Chat, no training |
| **EXP002** | **LoRA Fine-tuning** | PEFT adapter (r=16) trained on Alpaca + TruthfulQA human data |
| **EXP003** | **Response Distillation** | LoRA adapter trained on Llama 3.1 8B teacher-generated answers |
| **EXP004** | **Direct SFT** | Full-parameter tuning on human data (Adafactor optimizer) |
| **EXP005** | **KL Distillation** | Logit-level distillation using OpenLlama-3B; loss = 0.7×KL + 0.3×CE, T=3 |

---

## 5. Dataset

| Source | Type | Count |
|---|---|---|
| tatsu-lab/alpaca | Instruction following | 300 samples |
| truthfulqa/truthful_qa (generation) | Hallucination avoidance | 100 samples |
| Manual anchor samples | High-priority factual anchors | 20 samples |
| **Total SFT training samples** | Mixed | **~420** |
| **Scaled distillation run** | Alpaca + TruthfulQA + anchors | **~4,900+** |

**Evaluation sets:**
- 10 training questions (seen during fine-tuning, sanity check)
- **50 held-out questions** (never seen — true generalisation test)
- **200 TruthfulQA MC** questions (standard benchmark)
- 12 explicit factuality probes + 10 unanswerable hallucination probes
- 8 multilingual QA samples (HI, DE, FR, ZH)

---

## 6. Evaluation Framework

| Metric | Type | Notes |
|---|---|---|
| **TruthfulQA MC1** | ✅ Standard benchmark | Accuracy selecting the single correct answer via log-prob |
| **TruthfulQA MC2** | ✅ Standard benchmark | Normalised probability mass on all correct answers |
| **Explicit Factuality Acc** | ✅ Targeted probe | Must-have + forbidden term exact-match on 12 factual questions |
| **Explicit Hallucination Rate** | ✅ Targeted probe | Rate of confident answers on 10 unanswerable/fictional questions |
| **ROUGE-L Proxy** | ⚠️ Proxy (heuristic) | LCS-based overlap with reference answers |
| **Factuality Proxy** | ⚠️ Proxy (heuristic) | Token-set overlap between response and reference |
| **Hallucination Proxy** | ⚠️ Proxy (heuristic) | Repetition + entropy-based text degeneration score |
| **Instruction-Following Proxy** | ⚠️ Proxy (heuristic) | Keyword overlap + response length heuristic |
| **Consistency** | Qualitative | Pairwise ROUGE-L across 3 stochastic runs per question |

> **Note:** All evaluation decoding is deterministic (`do_sample=False`). Consistency evaluation uses stochastic sampling (temperature=0.7). All aggregate scores include 95% bootstrap confidence intervals (1000 resamples).

---

## 7. Results

All numbers are from the final executed run in [`notebooks/month1_final_v4.ipynb`](notebooks/month1_final_v4.ipynb) on **held-out questions never seen during training**.

### 7.1 Standard Benchmarks (Primary Evidence)

| Method | TruthfulQA MC1 ↑ | 95% CI | TruthfulQA MC2 ↑ | 95% CI |
|:---|:---:|:---:|:---:|:---:|
| EXP001: Base Student | 0.335 | (0.270, 0.405) | 0.4535 | (0.422, 0.485) |
| EXP004: Direct SFT | **0.465** | (0.400, 0.530) | **0.4916** | (0.460, 0.525) |
| EXP002: LoRA SFT | 0.395 | (0.325, 0.460) | 0.4682 | (0.437, 0.501) |
| EXP003: Response Distill | 0.355 | (0.290, 0.420) | 0.4638 | (0.432, 0.495) |

### 7.2 Explicit Factuality & Hallucination Probes

| Method | Explicit Fact Acc ↑ | Explicit Hall Rate ↓ |
|:---|:---:|:---:|
| EXP001: Base Student | 0.500 | 0.900 |
| EXP004: Direct SFT | **0.750** | **0.700** |
| EXP002: LoRA SFT | **0.750** | 0.800 |
| EXP003: Response Distill | 0.583 | **0.700** |

### 7.3 Proxy Metrics — Held-Out (Qualitative Signal)

| Method | ROUGE-L ↑ | Factuality Proxy ↑ | Hallucination Proxy ↓ | Instr-Following ↑ |
|:---|:---:|:---:|:---:|:---:|
| EXP001: Base Student | 0.370 | 0.844 | 0.195 | 0.937 |
| EXP004: Direct SFT | 0.124 | 0.842 | 0.430 | 0.953 |
| EXP002: LoRA SFT | 0.151 | 0.858 | 0.344 | 0.947 |
| EXP003: Response Distill | **0.341** | 0.831 | **0.221** | 0.938 |

### 7.4 Response Consistency (Held-Out, Stochastic 3-run)

| Method | Avg Consistency (ROUGE-L pairwise) ↑ |
|:---|:---:|
| EXP001: Base Student | 0.585 |
| EXP002: LoRA SFT | 0.312 |
| EXP003: Response Distill | 0.360 |

---

## 8. Model Efficiency Profile

| Model | Total Params | Trainable | Peak GPU Mem | Tokens/sec |
|:---|:---:|:---:|:---:|:---:|
| Base Student | 1,100M | 1,100M | 2.21 GB | **187.0** |
| Direct SFT | 1,100M | 1,100M | 2.23 GB | 29.0 |
| LoRA SFT | 1,104.6M | **4.5M** | 2.28 GB | 8.9 |

> LoRA trains only **0.4% of parameters** (4.5M / 1,104.6M) while achieving comparable factual accuracy to full-parameter SFT with far less text degradation.

---

## 9. Key Findings

### Finding 1 — The SFT Fluency-Factuality Drift
Full-parameter Direct SFT achieves the **highest TruthfulQA MC1 (0.465)** and explicit factuality accuracy (0.750), but at a severe cost: free-text ROUGE-L collapses to **0.124** and the hallucination proxy (repetition/entropy) spikes to **0.430**. The model selects correct MCQ answers but generates repetitive or incoherent free-form text — a classic alignment–fluency trade-off.

### Finding 2 — Response Distillation Offers the Best Balance
Training on Llama 3.1 8B teacher outputs via Response Distillation (EXP003) achieves:
- ROUGE-L of **0.341** (close to the 0.370 baseline — fluency preserved)
- Hallucination proxy of **0.221** (lowest after baseline)
- Explicit hallucination rate reduced to **70%** (from 90% baseline)
- TruthfulQA MC1 of **0.355** (+2pp over baseline)

This confirms that distilling structured, aligned teacher outputs provides a smoother training signal for small models than raw human-written SFT data.

### Finding 3 — LoRA is the Best Parameter-Efficient Strategy
LoRA SFT trains only **4.5M parameters** (~0.4%) and matches Direct SFT on factual accuracy (0.750) with significantly lower text degeneration (hallucination proxy 0.344 vs 0.430). It is the recommended fine-tuning strategy for deployment-constrained settings.

### Finding 4 — TruthfulQA MC vs. Proxy Metrics Diverge
Standard benchmark and proxy metric rankings do **not agree**:
- Direct SFT ranks **1st** on TruthfulQA MC1 but **last** on ROUGE-L and hallucination proxy.
- Response Distillation ranks **last** on MC1 but **best** on free-text quality metrics.

This shows that MCQ-style benchmarks and open-ended generation quality measure fundamentally different capabilities. **Both must be reported together.**

---

## 10. Current Limitations

- **Capacity ceiling:** At 1.1B parameters, TinyLlama cannot fully absorb the 8B teacher's factual depth. Explicit hallucination rates remain high (≥70%) across all approaches.
- **Multilingual gap:** TinyLlama is English-primary; multilingual ROUGE-L is near-zero for some language pairs.
- **Optimizer confound:** Direct SFT uses Adafactor (memory limit), LoRA uses AdamW — a documented experimental confound.
- **Small evaluation set:** The held-out set is 50 questions; results carry uncertainty (shown via CIs).
- **Response distillation ≠ logit-level KD:** Teacher logits are unavailable via API; distillation is purely response-level (imitation learning), not true soft-label KD.

---

## 11. Next Steps — Month 2

| Priority | Task |
|---|---|
| 🔴 High | Integrate HaluEval or SelfCheckGPT for robust hallucination benchmarking |
| 🔴 High | Scale multilingual evaluation using prepared HI/DE/FR/ZH subsets |
| 🟡 Medium | Implement Retrieval-Augmented Generation (RAG) to ground factual responses |
| 🟡 Medium | Evaluate 4-bit and 8-bit quantization impact on quality and speed |
| 🟢 Low | Investigate self-consistency decoding (majority voting) for alignment |

---

## 12. Project Structure

```
alignment-distillation/
├── README.md                       ← This file
├── PROJECT_STATUS.md               ← Task tracking & roadmap
├── WEEKLY_REPORT_TEMPLATE.md       ← Friday supervisor report template
├── requirements.txt
├── .gitignore
├── docs/
│   ├── month1_report.md            ← Month 1 deliverables & conclusions
│   ├── experiment_log.md           ← EXP001–EXP004 hyperparameters & results
│   ├── meeting_notes.md            ← Running supervisor meeting log
│   ├── weekly_reports/             ← Submitted weekly reports
│   └── Teacher — Student idea Geoffrey Hinton.pdf
├── notebooks/
│   └── month1_final_v4.ipynb      ← Final executed notebook (all 5 experiments)
├── src/
│   ├── dataset_preparation.py      ← Dataset loading & formatting utilities
│   ├── distillation_framework.py   ← Teacher-student distillation with KL+CE loss
│   └── evaluation_pipeline.py      ← Evaluation metrics implementation
├── results/
│   ├── final_results.json          ← Serialised per-approach metric scores
│   ├── month1_results.csv          ← CSV summary across all splits
│   ├── month1_final_report.json    ← Full structured JSON report
│   ├── month1_summary.md           ← Auto-generated Markdown summary
│   ├── month1_results.png          ← Proxy metric bar charts
│   ├── month1_consistency.png      ← Consistency evaluation chart
│   ├── month1_multilingual.png     ← Multilingual evaluation chart
│   └── month1_artifacts.zip        ← Full artifact archive
├── configs/
│   └── experiment_config.yaml      ← Hyperparameters for all experiments
└── datasets/
    └── processed/                  ← Ignored by git (large files)
```

---

## 13. Setup

```bash
# Clone the repository
git clone <repo-url>
cd alignment-distillation

# Install dependencies
pip install -r requirements.txt
```

To reproduce the full Month 1 experiment, open and run:
```
notebooks/month1_final_v4.ipynb
```
in a GPU environment (NVIDIA T4 or better, ≥16 GB VRAM recommended).

> **API key required:** Set `GROQ_API_KEY` in your environment for teacher response generation (Response Distillation). HuggingFace login required for TinyLlama and OpenLlama model access.
