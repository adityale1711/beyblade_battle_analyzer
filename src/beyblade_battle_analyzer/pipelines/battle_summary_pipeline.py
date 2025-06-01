import shutil

from src.beyblade_battle_analyzer import logger
from src.beyblade_battle_analyzer.components.battle_summary import BattleSummary
from src.beyblade_battle_analyzer.config.configuration import ConfigurationManager


STAGE_NAME = 'Battle Summary Pipeline'
class BattleSummaryPipeline:
    """
    Pipeline to generate a summary of the Beyblade battle.
    """

    def __init__(self, result):
        """
        Initializes the BattleSummaryPipeline with the result of the video processing.

        :param result: Result from the video processing pipeline.
        """
        self.result = result

    def initiate_battle_summary_pipeline(self):
        """
        Initiates the battle summary pipeline.

        :return: Result containing the battle summary.
        """
        try:
            config = ConfigurationManager()
            config = config.get_battle_summary_config()

            # Log the video analyzer configuration
            battle_summary = BattleSummary(config, self.result)
            battle_summary = battle_summary.generate_summary()

            return battle_summary

        except Exception as e:
            logger.exception(f'Error occurred in {STAGE_NAME}: {e}')
            raise e

    def run(self):
        """
        Runs the battle summary pipeline.

        :return: Result containing the battle summary.
        """
        try:
            term_width = shutil.get_terminal_size().columns
            separator_length = (term_width - len(STAGE_NAME) - 2) // 2
            print(f"{'=' * separator_length} {STAGE_NAME} {'=' * separator_length}")
            battle_summary = self.initiate_battle_summary_pipeline()
            print(f"{'=' * separator_length} {STAGE_NAME} {'=' * separator_length}")

            return battle_summary
        except Exception as e:
            logger.exception(f'Error occurred while running the {STAGE_NAME}: {e}')
            raise e