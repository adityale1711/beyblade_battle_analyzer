import os
import shutil

from ultralytics import YOLO
from src.beyblade_battle_analyzer.entity.config_entity import ModelTrainingConfig


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        """
        Initializes the ModelTraining class with the provided configuration.

        :param config: ModelTrainingConfig object containing the configuration settings.
        """
        self.config = config
        self.model_file_path = os.path.join(self.config.weight_models_dir, self.config.weight_model)

    def train_model(self):
        try:
            if not self.config.weight_model.endswith('.pt'):
                raise ValueError("The weight model must be a .pt file.")

            # If the model file does not exist, create the directory and move the model file
            os.makedirs(self.config.weight_models_dir, exist_ok=True)
            if os.listdir(self.config.weight_models_dir) is None:
                # Load the YOLO model with the specified weight file
                model = YOLO(self.config.weight_model)
                shutil.move(str(self.config.weight_model), self.config.weight_models_dir)
            else:
                model = YOLO(self.model_file_path)

            # Get model information
            model.info()

            # Train the model with the specified parameters
            model.train(
                data=self.config.yaml_path,
                epochs=self.config.epochs,
                patience=self.config.patience,
                imgsz=self.config.image_size,
                project=str(self.config.root_dir),
                name=str(self.config.project_name),
                cos_lr=True
            )

            return model

        except Exception as e:
            raise ValueError(f"Error loading weight model: {e}")
