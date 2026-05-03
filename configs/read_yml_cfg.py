from dataclasses import dataclass, field
from typing import Any, Dict, List

import yaml


def _to_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


@dataclass
class ExperimentConfig:
    exp_name: str = ""
    version: int = 0
    save_dir: str = ""


@dataclass
class DatasetConfig:
    data_dir: str = ""
    annot_dir: str = ""
    train_split: Dict[str, List[int]] = field(default_factory=dict)
    val_split: Dict[str, List[int]] = field(default_factory=dict)
    test_split: Dict[str, List[int]] = field(default_factory=dict)
    num_classes_person: int = 0
    num_classes_group: int = 0
    seq: bool = False


@dataclass
class TrainingConfig:
    seed: int = 0
    batch_size: int = 0
    num_epochs: int = 0
    learning_rate: float = 0.0
    weight_decay: float = 0.0
    grad_clip: float = 0.0
    grad_accum_steps: int = 0
    optimizer: str = ""
    resume_path: str = ""


@dataclass
class config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_yaml(cls, config_path: str) -> "config":
        with open(config_path, "r", encoding="utf-8") as file_handle:
            raw_config = yaml.safe_load(file_handle) or {}

        experiment_config = raw_config.get("experiment", {})
        dataset_config = raw_config.get("dataset", {})
        training_config = raw_config.get("training", {})

        # Normalize numeric-like YAML strings (e.g. "1e-4") before dataclass construction.
        experiment_config = {
            **experiment_config,
            "version": _to_int(experiment_config.get("version", 0)),
        }
        dataset_config = {
            **dataset_config,
            "num_classes_person": _to_int(dataset_config.get("num_classes_person", 0)),
            "num_classes_group": _to_int(dataset_config.get("num_classes_group", 0)),
        }
        training_config = {
            **training_config,
            "seed": _to_int(training_config.get("seed", 0)),
            "batch_size": _to_int(training_config.get("batch_size", 0)),
            "num_epochs": _to_int(training_config.get("num_epochs", 0)),
            "learning_rate": _to_float(training_config.get("learning_rate", 0.0)),
            "weight_decay": _to_float(training_config.get("weight_decay", 0.0)),
            "grad_clip": _to_float(training_config.get("grad_clip", 0.0)),
            "grad_accum_steps": _to_int(training_config.get("grad_accum_steps", 0)),
            "optimizer": str(training_config.get("optimizer", "")),
            "resume_path": str(training_config.get("resume_path", "")),
        }

        return cls(
            experiment=ExperimentConfig(**experiment_config),
            dataset=DatasetConfig(**dataset_config),
            training=TrainingConfig(**training_config),
            raw=raw_config,
        )


def get_config(config_path: str) -> config:
    return config.from_yaml(config_path)