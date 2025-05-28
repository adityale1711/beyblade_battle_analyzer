from src.beyblade_battle_analyzer.constants import *
from src.beyblade_battle_analyzer.utils.common import read_yaml, create_directories
from src.beyblade_battle_analyzer.entity.config_entity import DataIngestionConfig


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
            dataset_version=config.dataset_version
        )

        return data_ingestion_config
