import cv2
import numpy as np

from typing import Tuple, List, Dict, Any
from ultralytics import YOLO
from src.beyblade_battle_analyzer import logger
from src.beyblade_battle_analyzer.entity.config_entity import BeybladeDetectorConfig


class BeybladeDetector:
    def __init__(self, config: BeybladeDetectorConfig):
        """
        Initializes the BeybladeDetector with the specified configuration.

        :param config: Configuration settings for Beyblade detection.
        """
        self.config = config
        self.model = None

        if self.config.model_path:
            try:
                self.model = YOLO(self.config.model_path)
                logger.info(f'Beyblade detection model loaded from {self.config.model_path}')
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
        Visualizes the detected Beyblades on the provided frame.

        :param frame: Input image to visualize detections on.
        :param detections: List of detected Beyblades with bounding boxes and confidence scores.
        :return: Annotated image with visualized detections.
        """

        # Create a copy of the frame to visualize detections
        vis_frame = frame.copy()

        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection['confidence']
            center = detection['center']

            # Draw bounding box
            cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            # Draw center point
            cv2.circle(vis_frame, center, 5, (255, 0, 0), -1)
            # Put confidence text
            cv2.putText(vis_frame, f'Beyblade {i + 1}: {confidence:.2f}', (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return vis_frame

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
