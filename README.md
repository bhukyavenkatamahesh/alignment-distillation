# Alignment Distillation and Trustworthiness Evaluation in Small Language Models


## Objective
Study whether trustworthy behavior and alignment from larger language models can be transferred to smaller models using distillation and adapter-based fine-tuning techniques.

## Project Structure
```
src/
├── distillation_framework.py   # Teacher-Student distillation with KL + CE loss
├── dataset_preparation.py      # Multilingual dataset loading and formatting
└── evaluation_pipeline.py      # 7 trustworthiness metrics (hallucination, factuality, etc.)
configs/
└── experiment_config.yaml      # All hyperparameters
requirements.txt                # Dependencies
```

## Models
- **Teacher:** Meta-Llama-3-8B-Instruct (aligned, 8B parameters)
- **Student:** TinyLlama-1.1B-Chat (compact, 1.1B parameters)

## Approaches Compared
1. Direct Fine-tuning (baseline)
2. LoRA Adapter-based Distillation
3. Response Distillation

## Evaluation Metrics
- Hallucination Rate
- Factuality Score
- ROUGE-L / BLEU-1
- Instruction-Following Score
- Consistency Score
- Toxicity Rate

## Datasets
- [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) — Instruction following
- [TruthfulQA](https://huggingface.co/datasets/truthful_qa) — Hallucination evaluation
- [XQuAD](https://huggingface.co/datasets/google/xquad) — Multilingual QA
- Languages: EN, HI, DE, FR, ZH

## Setup
```bash
pip install -r requirements.txt
```

## Run Evaluation Pipeline
```bash
python src/evaluation_pipeline.py
```

## Run Dataset Preparation
```bash
python src/dataset_preparation.py
```
