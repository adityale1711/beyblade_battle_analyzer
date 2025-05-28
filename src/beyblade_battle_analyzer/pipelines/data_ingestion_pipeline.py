import shutil

from src.beyblade_battle_analyzer import logger
from src.beyblade_battle_analyzer.config.configuration import ConfigurationManager
from src.beyblade_battle_analyzer.components.data_ingestion import DataIngestion


STAGE_NAME = "Data Ingestion Stage"
class DataIngestionPipeline:
    def __init__(self):
        pass

    def initiate_data_ingestion(self):
        """
        Initiates the data ingestion process by reading the configuration and downloading the dataset.
        """
        try:
            # Initialize the configuration manager to read the configuration settings
            config_manager = ConfigurationManager()
            data_ingestion_config = config_manager.get_data_ingestion_config()

            # Log the data ingestion configuration
            data_ingestion = DataIngestion(config=data_ingestion_config)

            # Download and configure the dataset
            dataset = data_ingestion.config_dataset()

            return dataset
        except Exception as e:
            logger.exception(f'Error occurred in {STAGE_NAME}: {e}')
            raise e

    def run(self):
        """
        Runs the data ingestion pipeline.
        """
        try:
            term_width = shutil.get_terminal_size().columns
            separator_length = (term_width - len(STAGE_NAME) - 2) // 2
            print(f"{'=' * separator_length} {STAGE_NAME} {'=' * separator_length}")
            dataset = self.initiate_data_ingestion()
            print(f"{'=' * separator_length} {STAGE_NAME} {'=' * separator_length}")

            return dataset
        except Exception as e:
            logger.exception(f'Error occurred while running the {STAGE_NAME}: {e}')
            raise e
