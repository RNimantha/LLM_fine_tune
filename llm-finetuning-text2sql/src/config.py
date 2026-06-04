"""Typed configuration loading.

Why this exists: passing dozens of loose kwargs around is how training runs become
irreproducible. We parse the YAML once into validated Pydantic models so a bad value
fails fast (at load time) instead of three hours into a run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class ModelCfg(BaseModel):
    base_model: str
    trust_remote_code: bool = False


class DataCfg(BaseModel):
    dataset_name: str
    test_size: float = 0.05
    seed: int = 42
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None


class LoraCfg(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


class QuantCfg(BaseModel):
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"


class TrainingCfg(BaseModel):
    output_dir: str
    num_train_epochs: float = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_length: int = 1024
    packing: bool = False
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    bf16: bool = True
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 2
    report_to: str = "tensorboard"
    seed: int = 42


class HubCfg(BaseModel):
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None


class Config(BaseModel):
    model: ModelCfg
    data: DataCfg
    lora: LoraCfg
    quant: QuantCfg
    training: TrainingCfg
    hub: HubCfg = Field(default_factory=HubCfg)


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config into a typed Config object."""
    raw = yaml.safe_load(Path(path).read_text())
    return Config.model_validate(raw)
