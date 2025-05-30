import numpy as np

from enum import Enum
from typing import List, Dict, Any, Tuple
from src.beyblade_battle_analyzer.entity.config_entity import BattleAnalyzerConfig, BeybladeTracker


class BattleState(Enum):
    STARTING = "starting"
    ACTIVE = "active"
    ENDING = 'ending'
    FINISHED = 'finished'


class BattleAnalyzer:
    def __init__(self, config: BattleAnalyzerConfig, fps, arena_bounds):
        """
        Initializes the BattleAnalyzer with the specified configuration.

        :param config: Configuration settings for battle analysis.
        """
        self.config = config
        self.fps = fps
        self.arena_bounds = arena_bounds
        self.battle_state = BattleState.STARTING
        self.trackers: Dict[int, BeybladeTracker] = {}
        self.stopped_frames_threshold = int(fps * 2)

        self.winner_id = None
        self.battle_start_frame = None
        self.battle_end_frame = None

    def _is_in_arena(self, position: Tuple[int, int]) -> bool:
        """
        Checks if the given position is within the defined arena bounds.

        :param position: Tuple containing the (x, y) coordinates of the position.
        :return: True if the position is within the arena bounds, False otherwise.
        """

        # If arena bounds are not defined, consider all positions valid
        if not self.arena_bounds:
            return True

        # Check if the position is within the arena bounds
        x, y = position

        # Assuming arena_bounds is a tuple of (x1, y1, x2, y2)
        x1, y1, x2, y2 = self.arena_bounds

        return x1 <= x <= x2 and y1 <= y <= y2

    def _update_tracker(self, tracker: BeybladeTracker, frame_number: int, detection: Dict[str, Any]) -> None:
        """
        Updates the tracker with the current frame's detection.

        :param tracker: The BeybladeTracker to update.
        :param frame_number: The current frame number.
        :param detection: The detected Beyblade information.
        """

        # Update the tracker's position and velocity
        current_pos = detection['center']

        # If the tracker has no previous position, initialize it
        tracker.position.append(current_pos)
        tracker.last_seen_frame = frame_number

        # Calculate velocity if there is a previous position
        if len(tracker.position) >= 2:
            last_pos = tracker.position[-2]
            distance = np.sqrt((current_pos[0] - last_pos[0]) ** 2 + (current_pos[1] - last_pos[1]) ** 2)
            velocity = distance * self.fps  # Convert to pixels per second
            tracker.velocities.append(velocity)

            avg_velocity = np.mean(tracker.velocities[-10:])
            tracker.is_spinning = avg_velocity > self.config.movement_threshold
            tracker.spin_confidence = min(1.0, avg_velocity / 20.0)

            # Check if the tracker has stopped spinning
            if not tracker.is_spinning and tracker.stopped_frame is None:
                tracker.stopped_frame = frame_number

        # If the tracker is spinning, reset the stopped frame
        if self.arena_bounds and not self._is_in_arena(current_pos):
            # If the Beyblade is out of the arena, mark it as exited
            if tracker.exit_frame is None:
                tracker.exit_frame = frame_number
                tracker.is_spinning = False

    def _check_tracker_status(self, tracker: BeybladeTracker, frame_number: int) -> None:
        """
        Checks the status of a tracker and updates its state.

        :param tracker: The BeybladeTracker to check.
        :param frame_number: The current frame number.
        """

        # Check if the tracker has been seen in the last 10 frames
        frame_since_seen = frame_number - tracker.last_seen_frame

        # If the tracker has not been seen for more than the threshold, mark it as stopped
        if frame_since_seen > self.stopped_frames_threshold:
            tracker.is_spinning = False

            # If the tracker has stopped spinning, set the stopped frame
            if tracker.stopped_frame is None:
                tracker.stopped_frame = tracker.last_seen_frame

    def _create_new_tracker(self, frame_number: int, detection: Dict[str, Any]) -> None:
        """
        Creates a new tracker for a detected Beyblade.

        :param frame_number: The current frame number.
        :param detection: The detected Beyblade information.
        """

        # Create a new tracker with the detection information
        tracker_id = len(self.trackers)
        tracker = BeybladeTracker(
            id=tracker_id,
            position=[detection['center']],
            velocities=[],
            is_spinning=True,
            last_seen_frame=frame_number,
            stopped_frame=None,
            exit_frame=None,
            spin_confidence=1.0
        )

        self.trackers[tracker_id] = tracker

    def _update_trackers(self, frame_number: int, detections: List[Dict[str, Any]]) -> None:
        """
        Updates the trackers with the current frame's detections.

        :param frame_number: The current frame number.
        :param detections: List of detected Beyblades in the current frame.
        """

        # Simple tracking: match detections to existing trackers by proximity
        unmatched_detections = detections.copy()

        # Iterate through existing trackers to find matches
        for tracker_id, tracker in self.trackers.items():
            best_match = None
            best_distance = float('inf')

            # If the tracker has a position, find the closest unmatched detection
            if tracker.position:
                last_pos = tracker.position[-1]

                # Iterate through unmatched detections to find the closest one
                for i, detection in enumerate(unmatched_detections):
                    current_pos = detection['center']
                    distance = np.sqrt((current_pos[0] - last_pos[0]) ** 2 + (current_pos[1] - last_pos[1]) ** 2)

                    # Check if this detection is a better match
                    if distance < best_distance and distance < 50:
                        best_distance = distance
                        best_match = i

            if best_match is not None:
                detection = unmatched_detections.pop(best_match)
                self._update_tracker(tracker, frame_number, detection)
            else:
                self._check_tracker_status(tracker, frame_number)

        # Create new trackers for unmatched detections
        for detection in unmatched_detections:
            self._create_new_tracker(frame_number, detection)

    def _determine_winner_by_duration(self) -> None:
        """
        Determines the winner based on the duration of spinning.
        The Beyblade with the longest duration of spinning is declared the winner.
        """

        # Reset the winner and latest stop frame
        latest_stop_frame = -1
        winner_id = None

        # Iterate through all trackers to find the one with the longest spinning duration
        for tracker in self.trackers.values():
            stop_frame = tracker.stopped_frame or tracker.last_seen_frame
            if stop_frame > latest_stop_frame:
                latest_stop_frame = stop_frame
                winner_id = tracker.id

        # Set the winner based on the longest spinning duration
        self.winner_id = winner_id

    def _analyze_battle_state(self, frame_number: int) -> None:
        """
        Analyzes the current battle state based on the trackers.

        :param frame_number: The current frame number.
        """

        # Check if there are any active trackers
        active_trackers = [tracker for tracker in self.trackers.values() if tracker.is_spinning and
                           (frame_number - tracker.last_seen_frame) <= self.stopped_frames_threshold]

        # Determine the battle state based on active trackers
        if self.battle_state == BattleState.STARTING:
            # If there are at least two active trackers, and they have enough frames, set the battle state to ACTIVE
            if len(active_trackers) >= 2 and len(self.trackers) >= 2:
                # Check if all active trackers have been seen for at least 2 frames
                oldest_tracker_frames = min(len(tracker.position) for tracker in self.trackers.values())

                # If the oldest tracker has been seen for at least 2 frames, start the battle
                if oldest_tracker_frames >= 2:
                    self.battle_state = BattleState.ACTIVE
                    self.battle_start_frame = frame_number
        elif self.battle_state == BattleState.ACTIVE:
            if len(active_trackers) <= 1:
                self.battle_state = BattleState.ENDING
                self.battle_start_frame = frame_number

                if len(active_trackers) == 1:
                    self.winner_id = active_trackers[0].id
                else:
                    self._determine_winner_by_duration()

        elif self.battle_state == BattleState.ENDING:
            if frame_number - self.battle_end_frame > self.fps:
                self.battle_state = BattleState.FINISHED

    def analyze(self, frame_number: int, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes the current frame for battle statistics.

        :param frame_number: The current frame number.
        :param detections: List of detected Beyblades in the current frame.
        :return: Analysis results including statistics and insights.
        """

        # Initialize analysis results
        frame_analysis = {
            'frame_number': frame_number,
            'detections_count': len(detections),
            'battle_state': self.battle_state.value,
            'active_beyblades': 0,
            'winner': self.winner_id
        }

        # Count active Beyblades and update their states
        self._update_trackers(frame_number, detections)

        # Count active Beyblades
        self._analyze_battle_state(frame_number)

        # Update the battle state based on the current frame
        frame_analysis['active_beyblades'] = sum(1 for tracker in self.trackers.values() if tracker.is_spinning
                                                 and tracker.last_seen_frame == frame_number)

        # If the battle is ending, set the end frame
        frame_analysis['battle_state'] = self.battle_state.value
        frame_analysis['winner'] = self.winner_id

        return frame_analysis

    def _analyze_movement_pattern(self, positions: List[Tuple[int, int]]) -> Dict[str, Any]:
        """
        Analyzes the movement pattern of a Beyblade based on its positions.

        :param positions: List of (x, y) positions of the Beyblade.
        :return: Dictionary containing movement pattern statistics.
        """

        # Ensure there are enough positions to analyze
        if len(positions) < 10:
            return {'pattern': 'insufficient_data'}

        # Calculate the center of the positions
        center_x = float(np.mean([pos[0] for pos in positions]))
        center_y = float(np.mean([pos[1] for pos in positions]))

        # Calculate distances from the center
        distances = [np.sqrt((pos[0] - center_x) ** 2 + (pos[1] - center_y) ** 2) for pos in positions]

        # Calculate movement radius and variance
        movement_radius = np.mean(distances)
        movement_variance = np.var(distances)

        return {
            'movement_radius': movement_radius,
            'movement_variance': movement_variance,
            'center_x': center_x,
            'center_y': center_y,
        }

    def _calculate_beyblade_stats(self, tracker: BeybladeTracker) -> Dict[str, Any]:
        """
        Calculates statistics for a given Beyblade tracker.

        :param tracker: The BeybladeTracker to calculate statistics for.
        :return: Dictionary containing the Beyblade's statistics.
        """
        stats = {
            'total_frames_tracked': len(tracker.position),
            'stopped_frames': tracker.stopped_frame,
            'exit_frame': tracker.exit_frame,
            'final_spinning_state': tracker.is_spinning,
            'max_velocity': max(tracker.velocities) if tracker.velocities else 0,
            'average_velocity': np.mean(tracker.velocities) if tracker.velocities else 0,
            'movement_pattern': self._analyze_movement_pattern(tracker.position)
        }

        # If the Beyblade has stopped, calculate survival time
        if tracker.stopped_frame and self.battle_start_frame:
            stats['survival_time'] = (tracker.stopped_frame - self.battle_start_frame) / self.fps

        return stats

    def get_battle_summary(self) -> Dict[str, Any]:
        """
        Returns a summary of the battle analysis.

        :return: Dictionary containing the battle summary.
        """

        if not self.battle_start_frame and self.battle_state == BattleState.STARTING:
            return {'status': 'No battle started yet.'}

        battle_duration = 0
        if self.battle_end_frame and self.battle_start_frame:
            battle_duration = (self.battle_end_frame - self.battle_start_frame) / self.fps

        summary = {
            'battle_duration': battle_duration,
            'battle_start_frame': self.battle_start_frame,
            'battle_end_frame': self.battle_end_frame,
            'winner_id': self.winner_id,
            'battle_state': self.battle_state.value,
            'total_beyblades_detected': len(self.trackers),
            'beyblades_stats': {}
        }

        for tracker_id, tracker in self.trackers.items():
            stats = self._calculate_beyblade_stats(tracker)
            summary['beyblade_stats'][tracker_id] = stats

        return summary
