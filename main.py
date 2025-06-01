import sys
import argparse

from src.beyblade_battle_analyzer import logger
from src.beyblade_battle_analyzer.pipelines.data_ingestion_pipeline import DataIngestionPipeline
from src.beyblade_battle_analyzer.pipelines.model_training_pipeline import ModelTrainingPipeline
from src.beyblade_battle_analyzer.pipelines.video_processing_pipeline import VideoProcessingPipeline
from src.beyblade_battle_analyzer.pipelines.select_arena_bounds_pipeline import SelectArenaBoundsPipeline


def parse_args():
    """
    Parse command line arguments.

    :return: Parsed arguments.
    """

    parser = argparse.ArgumentParser(description="Data Ingestion Script")
    parser.add_argument('--pipeline', type=str, choices=['training', 'video_analyzer'], default='training',
                        help='Specify the pipeline to run: training or inference.')

    args, _ = parser.parse_known_args()

    return args

def main():
    args = parse_args()

    if args.pipeline == 'training':
        try:
            data_ingestion = DataIngestionPipeline()
            data_ingestion.run()

            model_training = ModelTrainingPipeline()
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

            video_processor = VideoProcessingPipeline(arena_bounds)
            result = video_processor.run()

            logger.info('Video processing completed successfully.')
            logger.info(f"Frame processed: {result['frame_processed']}, ")

            battle_summary = result['battle_summary']
            if battle_summary.get('battle_duration'):
                logger.info(f"Battle Duration: {battle_summary['battle_duration']:.2f} seconds")
            if battle_summary.get('winner_id') is not None:
                logger.info(f"Winner: Beyblade {battle_summary['winner_id']}")

            logger.info('Exported Results:')
            for file_type, file_path in result['export_results'].items():
                logger.info(f"{file_type}: {file_path}")
        except Exception as e:
            logger.exception(f'Error occurred: {e}')
            raise e
    else:
        logger.error(f'Invalid pipeline specified: {args.pipeline}')
        raise ValueError(f'Invalid pipeline specified: {args.pipeline}')


if __name__ == "__main__":
    sys.exit(main())