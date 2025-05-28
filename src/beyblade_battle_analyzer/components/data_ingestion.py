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
        # Check if dataset already exists
        if os.path.exists(self.config.root_dir) and os.listdir(self.config.root_dir):
            logger.info(f"Dataset already exists at {self.config.root_dir}. Skipping download.")
            class DummyDataset:
                def __init__(self, location):
                    self.location = location
            return DummyDataset(self.config.root_dir)

        # Initialize Roboflow with the provided API key
        rf = Roboflow(api_key=self.config.roboflow_api_key)

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

        # Log the workspace and project information
        project = rf.workspace(rf_workspace).project(rf_project)

        # Log the project version
        version = project.version(self.rf_project_version)

        # Download the dataset using the specified version and save to root_dir
        dataset = version.download(self.config.dataset_version)

        return dataset

    def config_dataset(self):
        # Download the dataset
        dataset = self.download_dataset()

        # Check if the dataset directory already exists in the root directory
        if os.listdir(self.config.root_dir) is not None:
            dataset_name = None
            for dataset_path in os.listdir(self.config.root_dir):
                dataset_name = dataset_path

            dataset_dir = os.path.join(self.config.root_dir, dataset_name)
        else:
            # Move the dataset to the root directory specified in the configuration
            dataset_dir = Path(dataset.location)
            dataset_dir = shutil.move(dataset_dir, self.config.root_dir)

        # Construct the path to the YAML file
        yaml_data = os.path.join(str(dataset_dir), 'data.yaml')

        # Write the dataset configuration to a YAML file
        with open(yaml_data, 'r') as file:
            data = yaml.safe_load(file)

        # Update the paths in the YAML data to point to the new dataset directory
        data['train'] = os.path.join(dataset_dir, 'train')
        data['val'] = os.path.join(dataset_dir, 'val')
        data['test'] = os.path.join(dataset_dir, 'test')

        # Write the updated YAML data back to the file
        with open(yaml_data, 'w') as file:
            yaml.dump(data, file)

        return yaml_data
