import shutil

from src.beyblade_battle_analyzer import logger
from src.beyblade_battle_analyzer.config.configuration import ConfigurationManager
from src.beyblade_battle_analyzer.components.video_processor import VideoProcessor

STAGE_NAME = 'Video Processing Stage'
class VideoProcessingPipeline:
    def __init__(self, arena_bounds, device_override: str = None):
        """
        Initialize the VideoProcessingPipeline.
        
        :param arena_bounds: Arena bounds coordinates
        :param device_override: Optional device override (e.g., 'cpu', 'cuda', 'auto')
        """
        self.arena_bounds = arena_bounds
        self.device_override = device_override

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
            
            # Override device if specified
            if self.device_override:
                logger.info(f"Overriding detector device setting from '{beyblade_detector_config.device}' to '{self.device_override}'")
                beyblade_detector_config.device = self.device_override

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