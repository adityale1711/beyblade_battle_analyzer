from typing import List, Tuple, Optional
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
    dataset_name: str


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
    device: str  # Added device configuration

@dataclass
class ArenaBoundsSelectorConfig:
    """
    Configuration for arena bounds selection.
    """
    root_dir: Path
    input_video_path: str
    window_name: str


@dataclass
class VideoProcessorConfig:
    """
    Configuration for video processing.
    """
    root_dir: Path
    input_video_path: str
    output_video_path: Path
    arena_bounds: Tuple[int, int, int, int]  # Explicitly define as tuple of 4 integers
    visualization: bool


@dataclass
class BattleAnalyzerConfig:
    """
    Configuration for battle analysis.
    """
    root_dir: Path
    movement_threshold: float


@dataclass
class BeybladeDetectorConfig:
    """
    Configuration for Beyblade detection.
    """
    root_dir: Path
    model_path: str
    image_size: int
    confidence_threshold: float
    device: str  # Added device configuration


@dataclass
class BeybladeTrackerConfig:
    """
    Configuration for Beyblade tracking.
    """
    id: int
    position: List[Tuple[int, int]]
    velocities: List[float]
    is_spinning: bool
    last_seen_frame: int
    stopped_frame: Optional[int]
    exit_frame: Optional[int]
    spin_confidence: float


@dataclass
class BattleSummaryConfig:
    """
    Configuration for battle summary.
    """
    root_dir: Path
