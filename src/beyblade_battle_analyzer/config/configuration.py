from src.beyblade_battle_analyzer.constants import *
from src.beyblade_battle_analyzer.utils.common import read_yaml, create_directories
from src.beyblade_battle_analyzer.entity.config_entity import (DataIngestionConfig, ModelTrainingConfig,
                                                               AnalyzeVideoConfig)


class ConfigurationManager:
    """
    A class to manage configuration settings for the Beyblade Battle Analyzer.
    """

    def __init__(self, config_file_path=CONFIG_FILE_PATH):
        """
        Initializes the ConfigurationManager with a specified configuration file path.

        :param config_file_path: Path to the configuration file.
        """
        self.config = read_yaml(str(config_file_path))
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Retrieves the data ingestion configuration.

        :return: DataIngestionConfig object containing the configuration settings.
        """
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            roboflow_api_key=config.roboflow_api_key,
            dataset_version=config.dataset_version,
            dataset_name=config.dataset_name
        )

        return data_ingestion_config

    def get_model_training_config(self):
        """
        Retrieves the training pipeline configuration.

        :return: TrainingPipelineConfig object containing the configuration settings.
        """
        config = self.config.model_training
        create_directories([config.root_dir])

        training_pipeline_config = ModelTrainingConfig(
            root_dir=config.root_dir,
            weight_models_dir=config.weight_models_dir,
            yaml_path=config.yaml_path,
            weight_model=config.weight_model,
            epochs=config.epochs,
            patience=config.patience,
            image_size=config.image_size,
            project_name=config.project_name
        )

        return training_pipeline_config

    def get_analyze_video_config(self) -> AnalyzeVideoConfig:
        """
        Retrieves the video analysis configuration.

        :return: AnalyzeVideoConfig object containing the configuration settings.
        """
        config = self.config.analyze_video
        create_directories([config.root_dir])

        analyze_video_config = AnalyzeVideoConfig(
            root_dir=config.root_dir,
            video_path=config.video_path,
            model_path=config.model_path,
            image_size=config.image_size
        )

        return analyze_video_config
