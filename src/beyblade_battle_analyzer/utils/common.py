import os
import yaml

from box import ConfigBox
from ensure import ensure_annotations
from box.exceptions import BoxValueError
from src.beyblade_battle_analyzer import logger


@ensure_annotations
def read_yaml(file_path: str) -> ConfigBox:
    """
    Reads a YAML file and returns its contents as a ConfigBox.

    :param file_path: Path to the YAML file.
    :return: ConfigBox containing the YAML data.
    """
    try:
        with open(file_path, 'r') as file:
            content = yaml.safe_load(file)
            logger.info(f'YAML file {file_path} loaded successfully.')

        return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f'Invalid YAML file format: {file_path}')
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(paths: list, verbose: bool = True):
    """
    Creates directories for the given paths if they do not already exist.

    :param paths: List of directory paths to create.
    :param verbose: If True, logs the creation of directories.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f'Directory created: {path}' if os.path.exists(path) else f'Directory already exists: {path}')
