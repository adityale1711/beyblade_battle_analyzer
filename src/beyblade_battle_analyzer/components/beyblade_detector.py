import numpy as np

from typing import Tuple, List, Dict, Any
from ultralytics import YOLO
from src.beyblade_battle_analyzer import logger
from src.beyblade_battle_analyzer.entity.config_entity import BeybladeDetectorConfig
from src.beyblade_battle_analyzer.utils.device_manager import DeviceManager
from src.beyblade_battle_analyzer.components.ui_visualizer import UIVisualizer


class BeybladeDetector:
    def __init__(self, config: BeybladeDetectorConfig):
        """
        Initializes the BeybladeDetector with the specified configuration.

        :param config: Configuration settings for Beyblade detection.
        """
        self.config = config
        self.model = None
        
        # Initialize UI visualizer for detection visualization
        self.ui_visualizer = UIVisualizer()
        
        # Determine the device to use
        self.device = DeviceManager.get_device(self.config.device)

        if self.config.model_path:
            try:
                self.model = YOLO(self.config.model_path)
                # Set the device for the model
                self.model.to(self.device)
                logger.info(f'Beyblade detection model loaded from {self.config.model_path} on device: {self.device}')
            except Exception as e:
                logger.exception(f'Failed to load Beyblade detection model from {self.config.model_path}: {e}')
                raise e

    def _calculate_center(self, bbox: np.ndarray) -> Tuple[int, int]:
        """
        Calculates the center of a bounding box.

        :param bbox: Bounding box coordinates in the format [x1, y1, x2, y2].
        :return: Tuple containing the center coordinates (x, y).
        """
        x_center = int((bbox[0] + bbox[2]) / 2)
        y_center = int((bbox[1] + bbox[3]) / 2)
        return x_center, y_center

    def visualize_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Visualizes the detected Beyblades on the provided frame using UIVisualizer.

        :param frame: Input image to visualize detections on.
        :param detections: List of detected Beyblades with bounding boxes and confidence scores.
        :return: Annotated image with visualized detections.
        """
        return self.ui_visualizer.visualize_simple_detections(frame, detections)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects Beyblades in the provided image using the configured model.

        :param frame: Input image to detect Beyblades in.
        :return: Detection results or insights.
        """
        try:
            detections = []
            results = self.model(frame, verbose=False)

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
                confidences = result.boxes.conf.cpu().numpy()  # Get confidence values

                if len(boxes) > 0:
                    for i, box in enumerate(boxes):
                        confidence = float(confidences[i])
                        if confidence >= self.config.confidence_threshold:
                            bbox = box
                            detection = {
                                'bbox': [int(x) for x in bbox],
                                'confidence': confidence,
                                'center': self._calculate_center(bbox),
                                'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                            }

                            detections.append(detection)
            return detections
        except Exception as e:
            logger.exception(f'Error during Beyblade detection: {e}')
            raise e
