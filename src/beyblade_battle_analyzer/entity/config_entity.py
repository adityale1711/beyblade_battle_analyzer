from pathlib import Path
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    """
    Configuration for data ingestion.
    """
    root_dir: Path
    source_url: str
    roboflow_api_key: str
    dataset_version: str


@dataclass
class ModelTrainingConfig:
    """
    Configuration for the training pipeline.
    """
    root_dir: Path
    weight_models_dir: Path
    yaml_path: str
    weight_model: str
    epochs: int
    patience: int
    image_size: int
    project_name: str
