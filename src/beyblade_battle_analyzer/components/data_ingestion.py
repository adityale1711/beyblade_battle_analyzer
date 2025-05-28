import os
import yaml
import shutil

from pathlib import Path
from roboflow import Roboflow
from urllib.parse import urlparse
from src.beyblade_battle_analyzer import logger
from src.beyblade_battle_analyzer.entity.config_entity import DataIngestionConfig


class DataIngestion:
    """
    Class for handling data ingestion from roboflow datasets.
    """

    def __init__(self, config: DataIngestionConfig):
        """
        Initializes the DataIngestion class with the provided configuration.

        :param config: DataIngestionConfig object containing the configuration settings.
        """
        self.config = config
        self.rf_project_version = None

    def download_dataset(self):
        """
        Downloads the dataset from the specified roboflow URL and saves it to the root artifacts.
        Skips download if the dataset already exists in the root directory.
        """
        # Parse the source URL to extract the path components
        parsed_url = urlparse(self.config.source_url)

        # Extract the path components from the URL
        path_parts = parsed_url.path.strip('/').split('/')

        # Ensure that the path has at least two components (workspace and project)
        rf_workspace = path_parts[0]
        rf_project = path_parts[1]

        # Check if the URL is from app.roboflow.com or universe.roboflow.com
        if parsed_url.netloc == 'app.roboflow.com':
            self.rf_project_version = path_parts[2]
        elif parsed_url.netloc == 'universe.roboflow.com':
            self.rf_project_version = path_parts[3]
        else:
            logger.error('Invalid Roboflow URL format. Please provide a valid URL.')

        # Construct the destination path for the dataset
        destination_path = os.path.join(self.config.root_dir, self.config.dataset_name)

        # Check if the dataset already exists
        if os.path.exists(destination_path):
            logger.info(f"Dataset already exists at {destination_path}. Skipping download.")
            yaml_data = os.path.join(destination_path, 'data.yaml')
            return yaml_data

        # If dataset doesn't exist, proceed with download
        logger.info(f"Downloading dataset to {destination_path}")

        # Initialize Roboflow with the provided API key
        rf = Roboflow(api_key=self.config.roboflow_api_key)

        # Get the project and version
        project = rf.workspace(rf_workspace).project(rf_project)
        version = project.version(self.rf_project_version)

        # Download the dataset using the specified version and save to root_dir
        dataset = version.download(self.config.dataset_version)

        # Move the dataset to the root directory specified in the configuration
        dataset_dir = Path(dataset.location)
        dataset_dir = shutil.move(dataset_dir, self.config.root_dir)

        # Construct the path to the YAML file
        yaml_data = os.path.join(str(dataset_dir), 'data.yaml')

        return yaml_data
