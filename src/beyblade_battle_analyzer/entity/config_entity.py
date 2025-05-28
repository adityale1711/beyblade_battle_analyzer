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
