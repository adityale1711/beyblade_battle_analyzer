import shutil

from src.beyblade_battle_analyzer import logger
from src.beyblade_battle_analyzer.config.configuration import ConfigurationManager
from src.beyblade_battle_analyzer.components.model_training import ModelTraining


STAGE_NAME = "Model Training Stage"
class ModelTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_training(self):
        """
        Initiates the model training process.
        """
        try:
            # Initialize the configuration manager to read the configuration settings
            config_manager = ConfigurationManager()
            model_training_config = config_manager.get_model_training_config()

            # Log the model training configuration
            model_training = ModelTraining(config=model_training_config)
            model_training.train_model()

            return model_training
        except Exception as e:
            logger.exception(f'Error occurred in {STAGE_NAME}: {e}')
            raise e

    def run(self):
        """
        Runs the model training pipeline.
        """
        try:
            term_width = shutil.get_terminal_size().columns
            separator_length = (term_width - len(STAGE_NAME) - 2) // 2
            print(f"{'=' * separator_length} {STAGE_NAME} {'=' * separator_length}")
            trained_model = self.initiate_model_training()
            print(f"{'=' * separator_length} {STAGE_NAME} {'=' * separator_length}")

            return trained_model
        except Exception as e:
            logger.exception(f'Error occurred while running the {STAGE_NAME}: {e}')
            raise e