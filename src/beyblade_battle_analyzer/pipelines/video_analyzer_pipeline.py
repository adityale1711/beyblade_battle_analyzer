import shutil

from src.beyblade_battle_analyzer import logger
from src.beyblade_battle_analyzer.config.configuration import ConfigurationManager
from src.beyblade_battle_analyzer.components.analyze_video import VideoAnalyzer

STAGE_NAME = 'Analyze Video Stage'
class VideoAnalyzerPipeline:
    def __init__(self):
        pass

    def initiate_video_analyzer_pipeline(self):
        """
        Initiates the video analysis pipeline.

        :return: None
        """
        try:
            config = ConfigurationManager()
            analyze_video_config = config.get_analyze_video_config()

            # Log the video analyzer configuration
            video_analyzer = VideoAnalyzer(config=analyze_video_config)
            video_analyzer.analyze()

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
            self.initiate_video_analyzer_pipeline()
            print(f"{'=' * separator_length} {STAGE_NAME} {'=' * separator_length}")
        except Exception as e:
            logger.exception(f'Error occurred while running the {STAGE_NAME}: {e}')
            raise e