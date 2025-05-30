import json
import pandas as pd

from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime
from src.beyblade_battle_analyzer import logger


class DataManager:
    """
    A class to manage data operations for the Beyblade Battle Analyzer.
    This class is responsible for loading, saving, and processing data related to battles.
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.frame_analysis: List[Dict[str, Any]] = []
        self.detection_data: List[Dict[str, Any]] = []
        self.battle_summaries: List[Dict[str, Any]] = []

    def add_frame_analysis(self, frame_data: Dict[str, Any], video_path: str) -> None:
        """
        Adds frame analysis data to the specified video.

        :param frame_data: Dictionary containing frame analysis data.
        :param video_path: Path to the video file.
        """

        # Ensure the video path is valid
        enhanced_frame_data = frame_data.copy()
        enhanced_frame_data.update({
            'video_path': video_path,
            'video_filename': Path(video_path).name,
        })

        self.frame_analysis.append(enhanced_frame_data)

    def add_detection_data(self, detections: List[Dict[str, Any]], frame_number: int, video_path: str) -> None:
        """
        Adds detection data for a specific frame.

        :param detections: List of detection results for the frame.
        :param frame_number: The frame number in the video.
        :param video_path: Path to the video file.
        """
        for detection in detections:
            detection_record = detection.copy()
            detection_record.update({
                'frame_number': frame_number,
                'video_path': video_path,
                'video_filename': Path(video_path).name,
            })
            self.detection_data.append(detection_record)

    def add_battle_summary(self, summary: Dict[str, Any], video_path: str) -> None:
        """
        Adds a summary of the battle to the data manager.

        :param summary: Dictionary containing the battle summary.
        :param video_path: Path to the video file.
        """
        enhanced_summary = summary.copy()
        enhanced_summary.update({
            'video_path': video_path,
            'analysis_timestamp': datetime.now().isoformat(),
            'video_filename': Path(video_path).name,
        })
        self.battle_summaries.append(enhanced_summary)

        logger.info(f'Battle summary added for video: {Path(video_path).name}')

    def _calculate_dataset_statistics(self) -> Dict[str, Any]:
        """
        Calculates statistics for the collected data.

        :return: Dictionary containing dataset statistics.
        """

        # Initialize an empty dictionary to hold statistics
        stats = {}

        try:
            # Battle duration statistics
            durations = [
                s.get('battle_duration', 0) for s in self.battle_summaries if s.get('battle_duration') is not None
            ]
            if durations:
                stats['avg_battle_duration'] = sum(durations) / len(durations)
                stats['max_battle_duration'] = max(durations)
                stats['min_battle_duration'] = min(durations)

            # Winner statistics
            winners = [s.get('winner') for s in self.battle_summaries if s.get('winner') is not None]
            if winners:
                winner_counts = {}
                for winner in winners:
                    winner_counts[winner] = winner_counts.get(winner, 0) + 1

                stats['most_common_winner'] = max(winner_counts, key=winner_counts.get)
                stats['winner_counts'] = winner_counts

            # Detection confidence statistics
            confidence = [
                d.get('confidence', 0) for d in self.detection_data if d.get('confidence') is not None
            ]
            if confidence:
                stats['avg_confidence'] = sum(confidence) / len(confidence)
                stats['max_confidence'] = max(confidence)
                stats['min_confidence'] = min(confidence)

            # Frame analysis statistics
            if self.frame_analysis:
                total_frames = len(self.frame_analysis)
                active_frames = sum(1 for f in self.frame_analysis if f.get('active_beyblades', 0) > 0)
                stats['active_frame_ratio'] = active_frames / total_frames if total_frames > 0 else 0
        except Exception as e:
            logger.exception(f'Error calculating dataset statistics: {e}')

        return stats


    def export_to_csv(self, export_type: str = 'all') -> Dict[str, str]:
        """
        Exports the collected data to CSV files.

        :param export_type: Type of data to export ('frame_analysis', 'detection_data', 'battle_summaries', or 'all').
        :return: Dictionary with paths to the exported CSV files.
        """

        exported_files = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if export_type in ['frame_analysis', 'all']:
            frame_analysis_df = pd.DataFrame(self.frame_analysis)
            frame_analysis_path = self.output_dir / f'frame_analysis_{timestamp}.csv'
            frame_analysis_df.to_csv(frame_analysis_path, index=False)
            exported_files['frame_analysis'] = str(frame_analysis_path)
            logger.info(f'Frame analysis data exported to {frame_analysis_path}')

        if export_type in ['detection_data', 'all']:
            detection_data_df = pd.DataFrame(self.detection_data)
            detection_data_path = self.output_dir / f'detection_data_{timestamp}.csv'
            detection_data_df.to_csv(detection_data_path, index=False)
            exported_files['detection_data'] = str(detection_data_path)
            logger.info(f'Detection data exported to {detection_data_path}')

        if export_type in ['battle_summaries', 'all']:
            battle_summaries_df = pd.DataFrame(self.battle_summaries)
            battle_summaries_path = self.output_dir / f'battle_summaries_{timestamp}.csv'
            battle_summaries_df.to_csv(battle_summaries_path, index=False)
            exported_files['battle_summaries'] = str(battle_summaries_path)
            logger.info(f'Battle summaries exported to {battle_summaries_path}')

        return exported_files

    def export_to_json(self) -> str:
        """
        Exports the collected data to a JSON file.

        :return: Path to the exported JSON file.
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            json_path = self.output_dir / f'complete_analysis_data_{timestamp}.json'

            complete_data = {
                'export_timestamp': datetime.now().isoformat(),
                'battle_summaries': self.battle_summaries,
                'frame_analysis': self.frame_analysis,
                'detection_data': self.detection_data,
                'statistics': self._calculate_dataset_statistics(),
            }

            with open(json_path, 'w') as json_file:
                json.dump(complete_data, json_file, indent=4, default=str)

            logger.info(f'Complete analysis data exported to {json_path}')
            return str(json_path)
        except Exception as e:
            logger.exception(f'Error exporting data to JSON: {e}')
            return ''
