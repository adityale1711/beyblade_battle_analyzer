import os
import cv2
import numpy as np

from typing import Dict, Any
from pathlib import Path
from src.beyblade_battle_analyzer import logger
from src.beyblade_battle_analyzer.entity.config_entity import (VideoProcessorConfig, BattleAnalyzerConfig,
                                                               BeybladeDetectorConfig)
from src.beyblade_battle_analyzer.components.data_manager import DataManager
from src.beyblade_battle_analyzer.components.battle_analyzer import BattleAnalyzer
from src.beyblade_battle_analyzer.components.beyblade_detector import BeybladeDetector
from src.beyblade_battle_analyzer.components.ui_visualizer import UIVisualizer


class VideoProcessor:
    def __init__(self, video_processor_config: VideoProcessorConfig, battle_analyzer_config: BattleAnalyzerConfig,
                 beyblade_detector_config: BeybladeDetectorConfig, arena_bounds):
        """
        Initializes the VideoProcessor with the specified configuration.
        :param video_processor_config:
        :param battle_analyzer_config:
        """
        self.video_processor_config = video_processor_config
        self.battle_analyzer_config = battle_analyzer_config
        self.beyblade_detector_config = beyblade_detector_config
        self.output_video_path = Path(video_processor_config.output_video_path)
        self.output_video_path.mkdir(parents=True, exist_ok=True)

        self.arena_bounds = arena_bounds
        self.detector = BeybladeDetector(self.beyblade_detector_config)
        self.data_manager = DataManager(str(self.video_processor_config.output_video_path))
        self.battle_analyzer = None
        
        # Initialize UI visualizer component
        self.ui_visualizer = UIVisualizer(arena_bounds)

    def _process_frame(self, frame: np.ndarray, frame_number: int, video_path: str) -> Dict[str, Any]:
        """
        Processes a single frame of the video.

        :param frame: The video frame to process.
        :param frame_number: The current frame number.
        :param video_path: The path to the video file.
        :return: Processed frame results.
        """

        # Detect Beyblades in the frame
        detections = self.detector.detect(frame)

        # Analyze the frame using the BattleAnalyzer
        frame_analysis = self.battle_analyzer.analyze(frame_number, detections)

        # Add the frame analysis and detection data to the data manager
        self.data_manager.add_frame_analysis(frame_analysis, video_path)
        self.data_manager.add_detection_data(detections, frame_number, video_path)

        return {
            'detections': detections,
            'analysis': frame_analysis,
        }

    def _filter_detections_in_arena(self, detections: list) -> list:
        """
        Filters detections to only include those within arena bounds.

        :param detections: List of all detected Beyblades.
        :return: List of detections within arena bounds.
        """
        if not self.arena_bounds:
            return detections
        
        filtered_detections = []
        x1, y1, x2, y2 = self.arena_bounds
        
        for detection in detections:
            bbox = detection.get('bbox', [])
            if len(bbox) >= 4:
                # Calculate center of bounding box
                center_x = bbox[0] + bbox[2] / 2
                center_y = bbox[1] + bbox[3] / 2
                
                # Check if center is within arena bounds
                if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                    filtered_detections.append(detection)
        
        return filtered_detections

    def _create_annotated_frame(self, frame: np.ndarray, detections: list, analysis: Dict[str, Any]) -> np.ndarray:
        """
        Creates an enhanced annotated frame with modern UI and comprehensive battle visualization.

        :param frame: The original video frame.
        :param detections: List of detected Beyblades.
        :param analysis: Analysis results for the frame.
        :return: Annotated video frame.
        """

        # Filter detections to only show those within arena bounds
        arena_detections = self._filter_detections_in_arena(detections)
        
        # Use UI visualizer to create the annotated frame
        annotated = self.ui_visualizer.create_annotated_frame(frame, arena_detections, analysis, self.battle_analyzer)

        return annotated

    @staticmethod
    def progress_callback(current: int, total: int):
        """
        Callback function to report progress during video processing.

        :param current: Current frame number being processed.
        :param total: Total number of frames in the video.
        """
        if current % 30 == 0:  # Log progress every 30 frames
            progress = (current / total) * 100
            logger.info(f'Processing progress: {progress:.2f}% ({current}/{total} frames)')

    def _export_results(self) -> Dict[str, Any]:
        """
        Exports the results of the video processing.

        :return: Dictionary containing export results.
        """

        # Initialize export results dictionary
        export_results = {}

        # Export data to CSV
        csv_file = self.data_manager.export_to_csv('all')
        export_results.update(csv_file)

        # Export data to JSON
        json_file = self.data_manager.export_to_json()
        if json_file:
            export_results['json'] = json_file

        return export_results

    def process(self):
        """
        Processes the video using the specified configuration.

        :return: Processed video data or results.
        """

        # Validate the input video path
        video_path = Path(self.video_processor_config.input_video_path)
        if not self.video_processor_config.input_video_path:
            raise ValueError("Input video path is not specified in the configuration.")

        # Open the video file
        logger.info(f'Processing video from {self.video_processor_config.input_video_path}')
        cap = cv2.VideoCapture(self.video_processor_config.input_video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {self.video_processor_config.input_video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f'Video properties - FPS: {fps}, Total Frames: {total_frames}, Width: {width}, Height: {height}')

        self.battle_analyzer = BattleAnalyzer(self.battle_analyzer_config, fps, self.arena_bounds)

        video_writer = None
        if self.video_processor_config.visualization:
            output_video_path = os.path.join(str(self.output_video_path), f'{video_path.stem}_result.mp4')
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        try:
            frame_number = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process the current frame
                frame_results = self._process_frame(frame, frame_number, str(video_path))

                # Save and show visualization if requested
                if self.video_processor_config.visualization and video_writer:
                    annotated_frame = self._create_annotated_frame(
                        frame, frame_results['detections'], frame_results['analysis']
                    )

                    # Write the annotated frame to the video file
                    video_writer.write(annotated_frame)

                    # Display the annotated frame
                    cv2.imshow("Beyblade Analysis", annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break

                # Call the progress callback
                if self.progress_callback:
                    self.progress_callback(frame_number, total_frames)

                # Increment the frame number
                frame_number += 1

            # Finalize the battle analysis and save the summary
            battle_summary = self.battle_analyzer.get_battle_summary()
            self.data_manager.add_battle_summary(battle_summary, str(video_path))

            # Export results after processing
            export_results = self._export_results()
            logger.info(f'Video processing completed. Results exported: {export_results}')

            return {
                'success': True,
                'video_path': str(video_path),
                'frame_processed': frame_number,
                'battle_summary': battle_summary,
                'export_results': export_results,
                'output_video': str(self.output_video_path) if self.video_processor_config.visualization else None
            }

        finally:
            cap.release()
            cv2.destroyAllWindows()
            if video_writer:
                video_writer.release()
