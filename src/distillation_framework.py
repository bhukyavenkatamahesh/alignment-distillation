"""
Alignment Distillation Framework
Teacher-Student Knowledge Distillation for Aligned Language Models

Project: "Alignment Distillation and Trustworthiness Evaluation in Small Language Models"
Author: Venkata (Internship Project under Dr. Kanica)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from dataclasses import dataclass, field
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Configuration Dataclasses
# ──────────────────────────────────────────────

@dataclass
class TeacherConfig:
    """Configuration for the Teacher (large aligned) model."""
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"   # Default teacher
    load_in_4bit: bool = True                                   # Quantise to save VRAM
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class StudentConfig:
    """Configuration for the Student (small) model."""
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # Default student
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 512


@dataclass
class LoRAConfig:
    """LoRA/Adapter fine-tuning configuration."""
    r: int = 16                         # LoRA rank
    lora_alpha: int = 32
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class DistillationConfig:
    """Controls the distillation training process."""
    # Loss weighting
    alpha: float = 0.5          # Weight for KL-divergence (soft) loss
    beta: float = 0.5           # Weight for cross-entropy (hard) loss
    temperature: float = 4.0    # Softmax temperature for soft targets

    # Training hyperparameters
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0

    # Approach flags
    use_lora: bool = True
    use_response_distillation: bool = True  # Use teacher-generated responses
    use_direct_finetuning: bool = False      # Baseline: train without teacher

    output_dir: str = "outputs/distilled_student"


# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────

class TeacherModel:
    """
    Wraps a large, aligned teacher model.
    Used to generate soft targets and reference responses.
    """

    def __init__(self, config: TeacherConfig):
        self.config = config
        logger.info(f"Loading teacher model: {config.model_name}")

        bnb_cfg = None
        if config.load_in_4bit and torch.cuda.is_available():
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_cfg,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info("Teacher model loaded successfully.")

    @torch.no_grad()
    def generate_response(self, prompt: str) -> str:
        """Generate a reference aligned response (for response distillation)."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @torch.no_grad()
    def get_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get raw logits from teacher (for soft-target KL distillation)."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


class StudentModel:
    """
    Wraps a small student model, optionally with LoRA adapters.
    """

    def __init__(self, config: StudentConfig, lora_config: Optional[LoRAConfig] = None):
        self.config = config
        logger.info(f"Loading student model: {config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        # Attach LoRA adapters if requested
        if lora_config is not None:
            peft_cfg = LoraConfig(
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                target_modules=lora_config.target_modules,
                lora_dropout=lora_config.lora_dropout,
                bias=lora_config.bias,
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, peft_cfg)
            self.model.print_trainable_parameters()
            logger.info("LoRA adapters attached to student model.")

        logger.info("Student model loaded successfully.")


# ──────────────────────────────────────────────
# Distillation Loss Functions
# ──────────────────────────────────────────────

class DistillationLoss(nn.Module):
    """
    Combined distillation loss:
      L = alpha * L_KL(soft) + beta * L_CE(hard)

    - L_KL  : KL divergence between teacher and student soft distributions
    - L_CE  : Standard cross-entropy against ground-truth labels
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, temperature: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.T = temperature

    def forward(
        self,
        student_logits: torch.Tensor,   # (B, L, V)
        teacher_logits: torch.Tensor,   # (B, L, V)
        labels: torch.Tensor,           # (B, L)
    ) -> dict:

        # ── Soft target loss (KL divergence) ──
        s_log_probs = F.log_softmax(student_logits / self.T, dim=-1)
        t_probs     = F.softmax(teacher_logits    / self.T, dim=-1)
        kl_loss = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (self.T ** 2)

        # ── Hard target loss (cross-entropy) ──
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        total = self.alpha * kl_loss + self.beta * ce_loss
        return {"total": total, "kl": kl_loss, "ce": ce_loss}


# ──────────────────────────────────────────────
# Distillation Trainer
# ──────────────────────────────────────────────

class AlignmentDistillationTrainer:
    """
    Orchestrates the full teacher → student distillation loop.

    Supported approaches (can be combined):
      1. Response distillation  – student learns from teacher-generated responses
      2. KL distillation        – student matches teacher token distribution
      3. Direct fine-tuning     – baseline, no teacher signal
    """

    def __init__(
        self,
        teacher: TeacherModel,
        student: StudentModel,
        distill_config: DistillationConfig,
    ):
        self.teacher = teacher
        self.student = student
        self.cfg = distill_config
        self.loss_fn = DistillationLoss(
            alpha=distill_config.alpha,
            beta=distill_config.beta,
            temperature=distill_config.temperature,
        )
        self.optimizer = torch.optim.AdamW(
            self.student.model.parameters(),
            lr=distill_config.learning_rate,
        )
        self.history: list[dict] = []

    def _encode(self, texts: list[str]) -> dict:
        """Tokenise a batch of strings using the student tokeniser."""
        return self.student.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.student.config.max_length,
        )

    def train_step(self, batch: dict) -> dict:
        """
        Single training step.
        batch = {"input_ids": Tensor, "attention_mask": Tensor, "labels": Tensor}
        """
        self.student.model.train()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # Student forward pass
        student_out = self.student.model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = student_out.logits

        if self.cfg.use_response_distillation or not self.cfg.use_direct_finetuning:
            # Teacher soft targets
            teacher_logits = self.teacher.get_logits(input_ids, attention_mask)
            losses = self.loss_fn(student_logits, teacher_logits, labels)
        else:
            # Baseline: pure cross-entropy
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels  = labels[..., 1:].contiguous()
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            losses = {"total": ce, "kl": torch.tensor(0.0), "ce": ce}

        # Backward
        (losses["total"] / self.cfg.gradient_accumulation_steps).backward()
        torch.nn.utils.clip_grad_norm_(self.student.model.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {k: v.item() for k, v in losses.items()}

    def save(self, path: Optional[str] = None):
        """Save the fine-tuned student model (LoRA adapters or full weights)."""
        save_path = path or self.cfg.output_dir
        self.student.model.save_pretrained(save_path)
        self.student.tokenizer.save_pretrained(save_path)
        logger.info(f"Student model saved to {save_path}")


# ──────────────────────────────────────────────
# Quick sanity-check (no GPU needed)
# ──────────────────────────────────────────────

def _mock_smoke_test():
    """Smoke-test the loss function and config objects without loading real models."""
    logger.info("Running smoke test (CPU, mock tensors)…")

    cfg = DistillationConfig()
    loss_fn = DistillationLoss(alpha=cfg.alpha, beta=cfg.beta, temperature=cfg.temperature)

    B, L, V = 2, 16, 1000  # batch, seq_len, vocab_size
    s_logits = torch.randn(B, L, V)
    t_logits = torch.randn(B, L, V)
    labels   = torch.randint(0, V, (B, L))
    labels[labels == 0] = -100   # simulate padding

    losses = loss_fn(s_logits, t_logits, labels)
    for k, v in losses.items():
        logger.info(f"  {k}: {v:.4f}")

    assert all(not torch.isnan(v) for v in losses.values()), "NaN detected in losses!"
    logger.info("Smoke test PASSED ✓")


if __name__ == "__main__":
    _mock_smoke_test()
