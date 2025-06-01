import numpy as np

from enum import Enum
from typing import List, Dict, Any, Tuple
from src.beyblade_battle_analyzer.entity.config_entity import BattleAnalyzerConfig, BeybladeTrackerConfig


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
        self.trackers: Dict[int, BeybladeTrackerConfig] = {}
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

    def _update_tracker(self, tracker: BeybladeTrackerConfig, frame_number: int, detection: Dict[str, Any]) -> None:
        """
        Updates the tracker with the current frame's detection using improved logic.

        :param tracker: The BeybladeTrackerConfig to update.
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

            # Use more sophisticated spinning detection
            if len(tracker.velocities) >= 3:
                # Consider recent velocity trend
                recent_velocities = tracker.velocities[-5:] if len(tracker.velocities) >= 5 else tracker.velocities
                avg_velocity = np.mean(recent_velocities)
                
                # Dynamic threshold based on detection confidence
                detection_confidence = detection.get('confidence', 0.5)
                adjusted_threshold = self.config.movement_threshold * (0.7 + 0.3 * detection_confidence)
                
                was_spinning = tracker.is_spinning
                tracker.is_spinning = avg_velocity > adjusted_threshold
                
                # Update spin confidence based on velocity and consistency
                velocity_consistency = 1.0 - min(1.0, np.std(recent_velocities) / max(1.0, avg_velocity))
                tracker.spin_confidence = min(1.0, (avg_velocity / (adjusted_threshold * 2)) * velocity_consistency)

                # Check if the tracker has stopped spinning (with hysteresis to prevent flickering)
                if was_spinning and not tracker.is_spinning and tracker.stopped_frame is None:
                    # Only mark as stopped if velocity has been low for multiple frames
                    if len(recent_velocities) >= 3 and all(v < adjusted_threshold for v in recent_velocities[-3:]):
                        tracker.stopped_frame = frame_number
                elif not was_spinning and tracker.is_spinning and tracker.stopped_frame is not None:
                    # Resumed spinning - clear stopped frame
                    tracker.stopped_frame = None

        # Check arena bounds
        if self.arena_bounds and not self._is_in_arena(current_pos):
            # If the Beyblade is out of the arena, mark it as exited
            if tracker.exit_frame is None:
                tracker.exit_frame = frame_number
                tracker.is_spinning = False
                if tracker.stopped_frame is None:
                    tracker.stopped_frame = frame_number

    def _check_tracker_status(self, tracker: BeybladeTrackerConfig, frame_number: int) -> None:
        """
        Checks the status of a tracker and updates its state with improved logic.

        :param tracker: The BeybladeTrackerConfig to check.
        :param frame_number: The current frame number.
        """

        # Check if the tracker has been seen in the last frames
        frames_since_seen = frame_number - tracker.last_seen_frame

        # If the tracker has not been seen for more than the threshold, mark it as stopped
        if frames_since_seen > self.stopped_frames_threshold:
            if tracker.is_spinning:
                tracker.is_spinning = False
                # Set stopped frame to when it was last actually seen, not current frame
                if tracker.stopped_frame is None:
                    tracker.stopped_frame = tracker.last_seen_frame
        
        # Additional logic: if tracker has very low recent velocities, mark as stopped
        elif len(tracker.velocities) >= 5:
            recent_velocities = tracker.velocities[-5:]
            avg_recent_velocity = np.mean(recent_velocities)
            
            # If velocity is consistently very low, consider it stopped
            if avg_recent_velocity < self.config.movement_threshold * 0.5:
                if tracker.is_spinning and tracker.stopped_frame is None:
                    tracker.stopped_frame = frame_number
                    tracker.is_spinning = False

    def _create_new_tracker(self, frame_number: int, detection: Dict[str, Any]) -> None:
        """
        Creates a new tracker for a detected Beyblade.

        :param frame_number: The current frame number.
        :param detection: The detected Beyblade information.
        """

        # Double-check that the detection is within arena bounds before creating tracker
        current_pos = detection['center']
        if self.arena_bounds and not self._is_in_arena(current_pos):
            return  # Don't create tracker for Beyblades outside arena

        # Create a new tracker with the detection information
        tracker_id = len(self.trackers)
        tracker = BeybladeTrackerConfig(
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
        Updates the trackers with the current frame's detections using improved matching.

        :param frame_number: The current frame number.
        :param detections: List of detected Beyblades in the current frame.
        """

        # Improved tracking: match detections to existing trackers by proximity
        unmatched_detections = detections.copy()
        matched_trackers = set()

        # Sort trackers by most recently seen first
        sorted_trackers = sorted(
            self.trackers.items(), 
            key=lambda x: x[1].last_seen_frame, 
            reverse=True
        )

        # Iterate through existing trackers to find matches
        for tracker_id, tracker in sorted_trackers:
            if tracker_id in matched_trackers:
                continue
                
            best_match = None
            best_distance = float('inf')

            # If the tracker has a position, find the closest unmatched detection
            if tracker.position:
                last_pos = tracker.position[-1]

                # Iterate through unmatched detections to find the closest one
                for i, detection in enumerate(unmatched_detections):
                    current_pos = detection['center']
                    distance = np.sqrt((current_pos[0] - last_pos[0]) ** 2 + (current_pos[1] - last_pos[1]) ** 2)

                    # Adaptive distance threshold based on tracker velocity
                    avg_velocity = np.mean(tracker.velocities[-5:]) if len(tracker.velocities) >= 5 else 10.0
                    max_distance = min(80, max(30, avg_velocity * 2))  # Dynamic threshold

                    # Check if this detection is a better match
                    if distance < best_distance and distance < max_distance:
                        best_distance = distance
                        best_match = i

            if best_match is not None:
                detection = unmatched_detections.pop(best_match)
                self._update_tracker(tracker, frame_number, detection)
                matched_trackers.add(tracker_id)
            else:
                self._check_tracker_status(tracker, frame_number)

        # Create new trackers for unmatched detections (limit to prevent spurious trackers)
        max_new_trackers = 4  # Reasonable limit for Beyblade battles
        for i, detection in enumerate(unmatched_detections):
            if len(self.trackers) < max_new_trackers:
                self._create_new_tracker(frame_number, detection)

    def _determine_winner_by_duration(self) -> None:
        """
        Determines the winner using an enhanced scoring system that considers:
        - Spinning duration (primary factor)
        - Movement quality and stability
        - Arena exit penalties
        - Detection confidence
        """
        if not self.trackers:
            return

        winner_scores = {}
        
        for tracker_id, tracker in self.trackers.items():
            score = self._calculate_winner_score(tracker)
            winner_scores[tracker_id] = score

        # Find the tracker with the highest score
        if winner_scores:
            self.winner_id = max(winner_scores, key=winner_scores.get)

    def _calculate_winner_score(self, tracker: BeybladeTrackerConfig) -> float:
        """
        Calculates a comprehensive winner score for a tracker.
        
        :param tracker: The BeybladeTrackerConfig to score
        :return: Score value (higher is better)
        """
        if not tracker.position:
            return 0.0

        # Base score: survival time (frames spinning)
        survival_frames = tracker.stopped_frame or tracker.last_seen_frame
        if self.battle_start_frame:
            survival_frames = max(0, survival_frames - self.battle_start_frame)
        
        base_score = survival_frames
        
        # Movement quality bonus (based on consistent velocity)
        movement_quality = self._calculate_movement_quality(tracker)
        
        # Arena exit penalty
        arena_penalty = 1.0
        if tracker.exit_frame and self.battle_start_frame:
            # Heavy penalty if exited early in the battle
            exit_time_ratio = (tracker.exit_frame - self.battle_start_frame) / max(1, survival_frames)
            arena_penalty = max(0.3, 1.0 - (1.0 - exit_time_ratio) * 0.7)
        
        # Detection confidence bonus
        confidence_bonus = min(tracker.spin_confidence, 1.0)
        
        # Stability bonus (less erratic movement is better)
        stability_bonus = self._calculate_stability_bonus(tracker)
        
        # Final score calculation
        final_score = (base_score * 
                      movement_quality * 
                      arena_penalty * 
                      (0.8 + 0.2 * confidence_bonus) * 
                      stability_bonus)
        
        return final_score

    def _calculate_movement_quality(self, tracker: BeybladeTrackerConfig) -> float:
        """
        Calculates movement quality based on velocity consistency.
        
        :param tracker: The BeybladeTrackerConfig to analyze
        :return: Quality score between 0.5 and 1.2
        """
        if len(tracker.velocities) < 3:
            return 0.8  # Neutral score for insufficient data
        
        # Consistent high velocity is good
        avg_velocity = np.mean(tracker.velocities)
        velocity_std = np.std(tracker.velocities)
        
        # Normalize velocity (assume good spinning is 10-50 pixels/frame)
        velocity_score = min(1.0, max(0.3, avg_velocity / 30.0))
        
        # Low variance in velocity indicates stable spinning
        consistency_score = max(0.5, 1.0 - min(1.0, velocity_std / max(1.0, avg_velocity)))
        
        return velocity_score * consistency_score * 1.2

    def _calculate_stability_bonus(self, tracker: BeybladeTrackerConfig) -> float:
        """
        Calculates stability bonus based on movement pattern.
        
        :param tracker: The BeybladeTrackerConfig to analyze
        :return: Stability score between 0.7 and 1.1
        """
        if len(tracker.position) < 10:
            return 1.0  # Neutral for insufficient data
        
        movement_pattern = self._analyze_movement_pattern(tracker.position)
        
        if movement_pattern.get('pattern') == 'insufficient_data':
            return 1.0
        
        # Lower variance indicates more stable movement
        movement_variance = movement_pattern.get('movement_variance', 0)
        movement_radius = movement_pattern.get('movement_radius', 1)
        
        # Stable spinning should have consistent radius
        stability = max(0.7, 1.1 - min(0.4, movement_variance / max(1.0, movement_radius * movement_radius)))
        
        return stability

    def _analyze_battle_state(self, frame_number: int) -> None:
        """
        Analyzes the current battle state based on the trackers with improved logic.

        :param frame_number: The current frame number.
        """

        # Check if there are any active trackers
        active_trackers = [tracker for tracker in self.trackers.values() if tracker.is_spinning and
                           (frame_number - tracker.last_seen_frame) <= self.stopped_frames_threshold]

        # Determine the battle state based on active trackers
        if self.battle_state == BattleState.STARTING:
            # Improved battle start detection: require multiple consistent trackers
            if len(active_trackers) >= 2 and len(self.trackers) >= 2:
                # Check if we have stable tracking for multiple beyblades
                stable_trackers = [
                    tracker for tracker in self.trackers.values() 
                    if len(tracker.position) >= 3 and  # At least 3 position records
                    len(tracker.velocities) >= 2  # At least 2 velocity measurements
                ]
                
                # If we have at least 2 stable trackers, start the battle
                if len(stable_trackers) >= 2:
                    self.battle_state = BattleState.ACTIVE
                    self.battle_start_frame = frame_number
                    
        elif self.battle_state == BattleState.ACTIVE:
            # Battle ending logic - when only 1 or 0 active trackers remain
            if len(active_trackers) <= 1:
                self.battle_state = BattleState.ENDING
                self.battle_end_frame = frame_number

                if len(active_trackers) == 1:
                    # Last Beyblade standing wins
                    self.winner_id = active_trackers[0].id
                else:
                    # No active Beyblades, determine winner by comprehensive scoring
                    self._determine_winner_by_duration()

        elif self.battle_state == BattleState.ENDING:
            # Wait a bit before marking as finished to ensure no false endings
            if self.battle_end_frame and frame_number - self.battle_end_frame > self.fps:
                self.battle_state = BattleState.FINISHED

    def analyze(self, frame_number: int, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes the current frame for battle statistics.

        :param frame_number: The current frame number.
        :param detections: List of detected Beyblades in the current frame.
        :return: Analysis results including statistics and insights.
        """

        # Filter out detections outside arena bounds before processing
        if self.arena_bounds:
            arena_detections = []
            for detection in detections:
                center = detection['center']
                if self._is_in_arena(center):
                    arena_detections.append(detection)
            filtered_detections = arena_detections
        else:
            filtered_detections = detections

        # Initialize analysis results
        frame_analysis = {
            'frame_number': frame_number,
            'detections_count': len(filtered_detections),  # Count only arena detections
            'battle_state': self.battle_state.value,
            'active_beyblades': 0,
            'winner': self.winner_id
        }

        # Count active Beyblades and update their states using filtered detections
        self._update_trackers(frame_number, filtered_detections)

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

    def _calculate_beyblade_stats(self, tracker: BeybladeTrackerConfig) -> Dict[str, Any]:
        """
        Calculates statistics for a given Beyblade tracker.

        :param tracker: The BeybladeTrackerConfig to calculate statistics for.
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
        Returns a comprehensive summary of the battle analysis.

        :return: Dictionary containing the battle summary.
        """

        if not self.battle_start_frame and self.battle_state == BattleState.STARTING:
            return {'status': 'No battle started yet.'}

        battle_duration = 0
        if self.battle_end_frame and self.battle_start_frame:
            battle_duration = (self.battle_end_frame - self.battle_start_frame) / self.fps
        elif self.battle_start_frame:
            # Battle started but not ended yet
            last_frame = max((t.last_seen_frame for t in self.trackers.values()), default=0)
            battle_duration = (last_frame - self.battle_start_frame) / self.fps

        summary = {
            'battle_duration': battle_duration,
            'battle_start_frame': self.battle_start_frame,
            'battle_end_frame': self.battle_end_frame,
            'winner_id': self.winner_id,
            'battle_state': self.battle_state.value,
            'total_beyblades_detected': len(self.trackers),
            'beyblade_stats': {},
            'winner_analysis': self._get_winner_analysis()
        }

        for tracker_id, tracker in self.trackers.items():
            stats = self._calculate_beyblade_stats(tracker)
            summary['beyblade_stats'][tracker_id] = stats

        return summary

    def _get_winner_analysis(self) -> Dict[str, Any]:
        """
        Provides detailed analysis of how the winner was determined.
        Enhanced to provide real-time analysis during active battles.
        
        :return: Dictionary containing winner analysis details
        """
        if not self.trackers:
            return {'reason': 'No trackers found', 'scores': {}}
        
        # Calculate scores for all trackers
        scores = {}
        for tracker_id, tracker in self.trackers.items():
            scores[tracker_id] = self._calculate_winner_score(tracker)
        
        # Check active Beyblades
        active_trackers = [
            (tid, tracker) for tid, tracker in self.trackers.items() 
            if tracker.is_spinning and tracker.stopped_frame is None
        ]
        
        # If battle has ended (winner_id is set), provide final analysis
        if self.winner_id is not None:
            winner_tracker = self.trackers.get(self.winner_id)
            if not winner_tracker:
                return {'reason': 'Winner tracker not found', 'scores': scores}
            
            # Determine reason for winning
            active_count = len(active_trackers)
            
            if active_count == 1 and winner_tracker.is_spinning:
                reason = 'Last Beyblade spinning'
            elif winner_tracker.exit_frame:
                reason = 'Winner by comprehensive scoring despite arena exit'
            elif winner_tracker.stopped_frame:
                reason = 'Winner by comprehensive scoring (longest survival + quality)'
            else:
                reason = 'Winner by comprehensive scoring (still spinning)'
            
            return {
                'reason': reason,
                'winner_score': scores.get(self.winner_id, 0),
                'all_scores': scores,
                'winner_stats': {
                    'survival_frames': (winner_tracker.stopped_frame or winner_tracker.last_seen_frame) - (self.battle_start_frame or 0),
                    'exit_frame': winner_tracker.exit_frame,
                    'final_spinning': winner_tracker.is_spinning,
                    'avg_velocity': np.mean(winner_tracker.velocities) if winner_tracker.velocities else 0,
                    'spin_confidence': winner_tracker.spin_confidence
                }
            }
        
        # Real-time analysis for active battles
        if self.battle_state == BattleState.ACTIVE and active_trackers:
            # Find current leader based on scores
            current_leader_id = max(scores.keys(), key=lambda k: scores[k]) if scores else None
            
            if current_leader_id is not None:
                leader_tracker = self.trackers[current_leader_id]
                active_count = len(active_trackers)
                
                return {
                    'reason': f'Current leader in active battle ({active_count} Beyblades still spinning)',
                    'current_leader_id': current_leader_id,
                    'leader_score': scores[current_leader_id],
                    'all_scores': scores,
                    'active_beyblades': [tid for tid, _ in active_trackers],
                    'leader_stats': {
                        'survival_frames': leader_tracker.last_seen_frame - (self.battle_start_frame or 0),
                        'exit_frame': leader_tracker.exit_frame,
                        'currently_spinning': leader_tracker.is_spinning,
                        'avg_velocity': np.mean(leader_tracker.velocities) if leader_tracker.velocities else 0,
                        'spin_confidence': leader_tracker.spin_confidence
                    },
                    'battle_progress': {
                        'total_beyblades': len(self.trackers),
                        'still_active': active_count,
                        'stopped': len(self.trackers) - active_count
                    }
                }
        
        # Default case: battle starting or no clear state
        if scores:
            best_id = max(scores.keys(), key=lambda k: scores[k])
            return {
                'reason': 'Battle analysis in progress',
                'potential_leader_id': best_id,
                'all_scores': scores,
                'active_beyblades': [tid for tid, _ in active_trackers] if active_trackers else [],
                'battle_state': self.battle_state.value
            }
        
        return {'reason': 'No winner determined', 'scores': {}}
