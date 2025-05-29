import cv2
import math
import numpy as np
import pandas as pd

from ultralytics import YOLO
from src.beyblade_battle_analyzer import logger
from src.beyblade_battle_analyzer.entity.config_entity import AnalyzeVideoConfig
from src.beyblade_battle_analyzer.components.beyblade_detector import BeybladeDetector


class AnalyzeVideo:
    def __init__(self, config: AnalyzeVideoConfig):
        """
        Initializes the AnalyzeVideo class with the provided configuration.

        :param config: AnalyzeVideoConfig object containing the configuration settings.
        """
        self.config = config
        self.beyblade_model = YOLO(self.config.model_path)
        self.battle_start_time = None
        self.trackers = []
        self.tracking_enabled = True
        self.tracker_reuse_threshold = 15  # Frames until we reinitialize trackers

    def calculate_motion_metrics(self, current_detections, previous_detections):
        """
        Calculates motion metrics between current and previous detections.

        :param current_detections: List of current detections.
        :param previous_detections: List of previous detections.
        :return: Dictionary containing motion metrics.
        """

        metrics = {
            'angular_velocity': [],
            'collision_events': 0,
            'arena_coverage': 0,
            'stability_index': []
        }

        if len(previous_detections) > 0 and len(current_detections) > 0:
            for current in current_detections:
                # Find closest previous detection
                min_distance = float('inf')
                closest_previous = None

                for previous in previous_detections:
                    # Extract coordinates from detection dictionaries
                    current_xmin, current_ymin, current_xmax, current_ymax = current['bbox']
                    previous_xmin, previous_ymin, previous_xmax, previous_ymax = previous['bbox']

                    distance = math.sqrt((current_xmin - previous_xmin) ** 2 + (current_ymin - previous_ymin) ** 2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_previous = previous

                if closest_previous is not None and min_distance < 50:
                    # Calculate angular velocity using bounding box centers
                    current_xmin, current_ymin, current_xmax, current_ymax = current['bbox']
                    previous_xmin, previous_ymin, previous_xmax, previous_ymax = closest_previous['bbox']

                    center_x = (current_xmin + current_xmax) / 2
                    center_y = (current_ymin + current_ymax) / 2
                    previous_center_x = (previous_xmin + previous_xmax) / 2
                    previous_center_y = (previous_ymin + previous_ymax) / 2
                    angular_velocity = math.sqrt((center_x - previous_center_x) ** 2 + (center_y - previous_center_y) ** 2)

                    metrics['angular_velocity'].append(angular_velocity)

                    # Stability index based on bounding box size consistency
                    current_area = (current_xmax - current_xmin) * (current_ymax - current_ymin)
                    previous_area = (previous_xmax - previous_xmin) * (previous_ymax - previous_ymin)
                    stability = 1 - abs(current_area - previous_area) / max(current_area, previous_area)

                    metrics['stability_index'].append(stability)

        return metrics

    def battle_status(self, current_detections, motion_metrics):
        """
        Determines the battle status based on current detections and motion metrics.

        :param current_detections: List of current detections.
        :param motion_metrics: Dictionary containing motion metrics.
        :return: Boolean indicating if a battle is ongoing.
        """
        if len(current_detections) < 2:
            return True, 'The other beyblade has been defeated or exited from the arena!'

        # Check if motion has significantly decreased
        average_velocity = np.mean(motion_metrics['angular_velocity']) if motion_metrics['angular_velocity'] else 0
        if average_velocity < 2.0:
            return True, 'The battle is ongoing but no significant motion detected.'

        return False, 'The battle is ongoing with significant motion detected.'

    def analyze(self):
        """
        Analyzes the video to detect beyblades and identify battles.

        :param confidence_threshold: Minimum confidence threshold for detections (default: 0.6)
        """
        # Opens the video file specified in the configuration
        cap = cv2.VideoCapture(self.config.video_path)

        # Get the frames rates (FPS) of the video
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Get the total number of frames in the video
        frame_count = 0

        # Initialize previous detections DataFrame
        previous_detections = pd.DataFrame()

        # Get image size for resizing from config
        image_size = self.config.image_size

        battle_metric = {
            'collision_count': 0,
            'max_velocity': 0,
            'average_stability': [],
            'position_history': [],
            'frame_data': []
        }

        while True:
            # Read a frame from the video
            ret, frame = cap.read()
            if not ret:
                break

            # Increment the frame count
            current_time = frame_count / fps

            # Resize the frame to match the model's training image size
            resized_frame = cv2.resize(frame, (image_size, image_size))

            detector = BeybladeDetector(model=self.beyblade_model, frame=resized_frame)

            # Perform detection with the model
            detector.detect()

            # First apply confidence filtering
            confident_detections = detector.filter_by_confidence(threshold=0.2)

            # Update detections with confident ones
            detector.detections = confident_detections

            # Filter for "spins" class only
            spins_detections = detector.filter_by_class("spin")

            # Update detections with spins class only
            detector.detections = spins_detections

            annotated_frame = detector.draw_detections()
            current_detection = len(spins_detections) >= 2

            if self.battle_start_time is None and current_detection:
                self.battle_start_time = current_time

            # Calculate motion metrics
            motion_metrics = self.calculate_motion_metrics(spins_detections, previous_detections)

            if motion_metrics['angular_velocity']:
                battle_metric['max_velocity'] = max(battle_metric['max_velocity'], max(motion_metrics['angular_velocity']))
                battle_metric['average_stability'].extend(motion_metrics['stability_index'])

            frame_data = {
                'frame_number': frame_count,
                'timestamp': current_time,
                'beyblade_count': len(spins_detections),
                'average_velocity': np.mean(motion_metrics['angular_velocity']) if motion_metrics['angular_velocity'] else 0
            }
            battle_metric['frame_data'].append(frame_data)

            is_battle_ongoing, battle_message = self.battle_status(spins_detections, motion_metrics)
            print(battle_message)

            # Update previous detections for next frame
            previous_detections = spins_detections.copy()

            frame_count += 1

            # Resize annotated frame back to original size for display
            annotated_frame = cv2.resize(annotated_frame, (frame.shape[1], frame.shape[0]))

            cv2.imshow('Beyblade Battle Analyzer', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
