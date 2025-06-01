import shutil

from src.beyblade_battle_analyzer import logger
from src.beyblade_battle_analyzer.config.configuration import ConfigurationManager
from src.beyblade_battle_analyzer.components.video_processor import VideoProcessor

STAGE_NAME = 'Video Processing Stage'
class VideoProcessingPipeline:
    def __init__(self, arena_bounds):
        self.arena_bounds = arena_bounds

    def initiate_video_processing_pipeline(self):
        """
        Initiates the video analysis pipeline.

        :return: None
        """
        try:
            config = ConfigurationManager()
            video_processor_config = config.get_video_processor_config()
            battle_analyzer_config = config.get_battle_analyzer_config()
            beyblade_detector_config = config.get_beyblade_detector_config()

            # Log the video analyzer configuration
            video_processing = VideoProcessor(
                video_processor_config, battle_analyzer_config, beyblade_detector_config, self.arena_bounds
            )
            video_processing = video_processing.process()

            return video_processing

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
            video_processing = self.initiate_video_processing_pipeline()
            print(f"{'=' * separator_length} {STAGE_NAME} {'=' * separator_length}")

            return video_processing
        except Exception as e:
            logger.exception(f'Error occurred while running the {STAGE_NAME}: {e}')
            raise e