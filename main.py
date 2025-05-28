import argparse

from src.beyblade_battle_analyzer import logger
from src.beyblade_battle_analyzer.pipelines.data_ingestion_pipeline import DataIngestionPipeline
from src.beyblade_battle_analyzer.pipelines.model_training_pipeline import ModelTrainingPipeline
from src.beyblade_battle_analyzer.pipelines.video_analyzer_pipeline import VideoAnalyzerPipeline


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
            video_analyzer = VideoAnalyzerPipeline()
            video_analyzer.run()
        except Exception as e:
            logger.exception(f'Error occurred: {e}')
            raise e
    else:
        logger.error(f'Invalid pipeline specified: {args.pipeline}')
        raise ValueError(f'Invalid pipeline specified: {args.pipeline}')


if __name__ == "__main__":
    main()