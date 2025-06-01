import shutil

from src.beyblade_battle_analyzer import logger
from src.beyblade_battle_analyzer.config.configuration import ConfigurationManager
from src.beyblade_battle_analyzer.components.arena_bounds_selector import ArenaBoundsSelector

STAGE_NAME = 'Select Arena Bounds Stage'
class SelectArenaBoundsPipeline:
    def __init__(self):
        pass

    def initiate_arena_bounds_selector_pipeline(self):
        """
        Initiates the arena bounds selection pipeline.

        :return: None
        """
        try:
            config = ConfigurationManager()
            arena_bounds_selector_config = config.get_arena_bounds_selector_config()

            # Log the arena selector configuration
            arena_bounds = ArenaBoundsSelector(arena_bounds_selector_config)
            arena_bounds = arena_bounds.select_bounds()

            return arena_bounds

        except Exception as e:
            logger.exception(f'Error occurred in {STAGE_NAME}: {e}')
            raise e

    def run(self):
        """
        Runs the video analyzer pipeline.
        """
        try:
            term_width = shutil.get_terminal_size().columns
            separator_length = (term_width - len(STAGE_NAME) - 2) // 2
            print(f"{'=' * separator_length} {STAGE_NAME} {'=' * separator_length}")
            arena_bounds = self.initiate_arena_bounds_selector_pipeline()
            print(f"{'=' * separator_length} {STAGE_NAME} {'=' * separator_length}")

            return arena_bounds
        except Exception as e:
            logger.exception(f'Error occurred while running the {STAGE_NAME}: {e}')
            raise e