from src.beyblade_battle_analyzer.constants import CONFIG_FILE_PATH
from src.beyblade_battle_analyzer.utils.common import read_yaml, create_directories
from src.beyblade_battle_analyzer.entity.config_entity import (DataIngestionConfig, ModelTrainingConfig,
                                                               ArenaBoundsSelectorConfig, VideoProcessorConfig,
                                                               BattleAnalyzerConfig, BeybladeDetectorConfig,
                                                               BattleSummaryConfig)


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
            project_name=config.project_name,
            device=config.device
        )

        return training_pipeline_config

    def get_arena_bounds_selector_config(self) -> ArenaBoundsSelectorConfig:
        """
        Retrieves the arena bounds selector configuration.

        :return: ArenaBoundsSelectorConfig object containing the configuration settings.
        """
        config = self.config.arena_bounds_selector
        create_directories([config.root_dir])

        arena_bounds_selector_config = ArenaBoundsSelectorConfig(
            root_dir=config.root_dir,
            input_video_path=config.input_video_path,
            window_name=config.window_name
        )

        return arena_bounds_selector_config

    def get_video_processor_config(self):
        """
        Retrieves the video processor configuration.

        :return: VideoProcessorConfig object containing the configuration settings.
        """
        config = self.config.video_processor
        create_directories([config.root_dir])

        video_processor_config = VideoProcessorConfig(
            root_dir=config.root_dir,
            input_video_path=config.input_video_path,
            output_video_path=config.output_video_path,
            arena_bounds=config.arena_bounds,
            visualization=config.visualization
        )

        return video_processor_config

    def get_battle_analyzer_config(self) -> BattleAnalyzerConfig:
        """
        Retrieves the battle analyzer configuration.

        :return: BattleAnalyzerConfig object containing the configuration settings.
        """
        config = self.config.battle_analyzer
        create_directories([config.root_dir])

        battle_analyzer_config = BattleAnalyzerConfig(
            root_dir=config.root_dir,
            movement_threshold=config.movement_threshold
        )

        return battle_analyzer_config

    def get_beyblade_detector_config(self) -> BeybladeDetectorConfig:
        """
        Retrieves the Beyblade detector configuration.

        :return: BeybladeDetectorConfig object containing the configuration settings.
        """
        config = self.config.beyblade_detector
        create_directories([config.root_dir])

        beyblade_detector_config = BeybladeDetectorConfig(
            root_dir=config.root_dir,
            model_path=config.model_path,
            image_size=config.image_size,
            confidence_threshold=config.confidence_threshold,
            device=config.device
        )

        return beyblade_detector_config

    def get_battle_summary_config(self) -> BattleSummaryConfig:
        """
        Retrieves the battle summary configuration.

        :return: BattleSummaryConfig object containing the configuration settings.
        """
        config = self.config.battle_summary
        create_directories([config.root_dir])

        battle_summary_config = BattleSummaryConfig(
            root_dir=config.root_dir
        )

        return battle_summary_config
