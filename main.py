import argparse

from src.beyblade_battle_analyzer import logger
from src.beyblade_battle_analyzer.pipelines.data_ingestion_pipeline import DataIngestionPipeline
from src.beyblade_battle_analyzer.pipelines.model_training_pipeline import ModelTrainingPipeline


def parse_args():
    """
    Parse command line arguments.

    :return: Parsed arguments.
    """

    parser = argparse.ArgumentParser(description="Data Ingestion Script")
    parser.add_argument('--pipeline', type=str, choices=['training', 'inference'], default='training',
                        help='Specify the pipeline to run: training or inference.')

    args, _ = parser.parse_known_args()

    return args

def main():
    args = parse_args()

    try:
        data_ingestion = DataIngestionPipeline()
        data_ingestion.run()

        model_training = ModelTrainingPipeline()
        model_training.run()
    except Exception as e:
        logger.exception(f'Error occurred: {e}')
        raise e


if __name__ == "__main__":
    main()