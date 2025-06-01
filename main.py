import sys
import argparse

from src.beyblade_battle_analyzer import logger
from src.beyblade_battle_analyzer.pipelines.data_ingestion_pipeline import DataIngestionPipeline
from src.beyblade_battle_analyzer.pipelines.model_training_pipeline import ModelTrainingPipeline
from src.beyblade_battle_analyzer.pipelines.battle_summary_pipeline import BattleSummaryPipeline
from src.beyblade_battle_analyzer.pipelines.video_processing_pipeline import VideoProcessingPipeline
from src.beyblade_battle_analyzer.pipelines.select_arena_bounds_pipeline import SelectArenaBoundsPipeline


def parse_args():
    """
    Parse command line arguments.

    :return: Parsed arguments.
    """

    parser = argparse.ArgumentParser(description="Beyblade Battle Analyzer")
    parser.add_argument('--pipeline', type=str, choices=['training', 'video_analyzer'], default='video_analyzer',
                        help='Specify the pipeline to run: training or video_analyzer.')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for inference/training. Options: auto, cpu, cuda, cuda:0, cuda:1, etc. Overrides config setting.')

    args, _ = parser.parse_known_args()

    return args

def main():
    args = parse_args()

    if args.pipeline == 'training':
        try:
            data_ingestion = DataIngestionPipeline()
            data_ingestion.run()

            model_training = ModelTrainingPipeline(device_override=args.device)
            model_training.run()
        except Exception as e:
            logger.exception(f'Error occurred: {e}')
            raise e
    elif args.pipeline == 'video_analyzer':
        try:
            arena_bounds = SelectArenaBoundsPipeline()
            arena_bounds = arena_bounds.run()

            logger.info('Arena bounds selection completed successfully.')
            logger.info(f'Arena bounds selected: {arena_bounds}')

            video_processor = VideoProcessingPipeline(arena_bounds, device_override=args.device)
            result = video_processor.run()

            logger.info('Video processing completed successfully.')

            battle_summary = BattleSummaryPipeline(result)
            battle_summary = battle_summary.run()
            logger.info('Battle summary generation completed successfully.')
        except Exception as e:
            logger.exception(f'Error occurred: {e}')
            raise e
    else:
        logger.error(f'Invalid pipeline specified: {args.pipeline}')
        raise ValueError(f'Invalid pipeline specified: {args.pipeline}')


if __name__ == "__main__":
    sys.exit(main())