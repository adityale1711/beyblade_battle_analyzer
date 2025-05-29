import numpy as np
from collections import deque
import cv2


class BeybladeTracker:
    """
    Class to handle inconsistent detections and maintain tracking history.
    Works as a wrapper around the detector to provide more stable tracking.
    """
    def __init__(self, max_disappeared=30, max_distance=50, history_size=30, max_objects=2,
                 stationary_threshold=5, stationary_frames=20):
        """
        Initialize the tracker.

        Args:
            max_disappeared: Maximum number of frames a beyblade can disappear before its ID is removed
            max_distance: Maximum pixel distance to associate detections between frames
            history_size: Number of previous positions to store for each tracked object
            max_objects: Maximum number of objects expected (for ID recycling)
            stationary_threshold: Maximum velocity (pixels/frame) to consider a Beyblade as stationary
            stationary_frames: Number of consecutive frames with low motion to confirm a Beyblade has stopped
        """
        self.next_object_id = 1  # Start from 1 instead of 0
        self.objects = {}  # Dictionary: object_id -> centroid
        self.disappeared = {}  # Dictionary: object_id -> number of frames disappeared

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.max_objects = max_objects

        # Track available IDs for recycling - start from 1 instead of 0
        self.available_ids = set(range(1, max_objects + 1))

        # Track motion history
        self.history = {}  # Dictionary: object_id -> deque of previous positions
        self.history_size = history_size

        # Store class information
        self.class_info = {}  # Dictionary: object_id -> most common class name

        # Motion analysis parameters
        self.stationary_threshold = stationary_threshold  # Max velocity to consider as stationary
        self.stationary_frames = stationary_frames  # Required consecutive stationary frames
        self.stationary_counts = {}  # Count of consecutive stationary frames
        self.velocities = {}  # Recent velocities for each object
        self.is_stationary = {}  # Whether each Beyblade is currently stationary

        # Arena detection
        self.arena_bounds = None  # Can be set externally: (x_min, y_min, x_max, y_max)
        self.outside_arena = {}  # Track whether Beyblades are outside the arena

    def _calculate_centroid(self, bbox):
        """Calculate centroid from bounding box [x1, y1, x2, y2]"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _calculate_iou(self, bbox1, bbox2):
        """Calculate IoU (Intersection over Union) between two bboxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            # No intersection
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area

        return iou

    def _get_most_probable_match(self, bbox, prev_tracks, method='combined'):
        """
        Find the best match for a detection using different metrics

        Args:
            bbox: Current detection bounding box
            prev_tracks: Dictionary of existing tracks
            method: Matching method ('distance', 'iou', or 'combined')

        Returns:
            best_id: ID of best matching track or None if no good match
            best_score: Best match score
        """
        best_id = None
        best_score = float('inf') if method in ['distance', 'combined'] else -float('inf')

        centroid = self._calculate_centroid(bbox)

        for object_id, track_info in prev_tracks.items():
            prev_centroid = track_info['centroid']
            prev_bbox = track_info['bbox']

            # Calculate distance between centroids
            distance = np.sqrt((centroid[0] - prev_centroid[0])**2 +
                              (centroid[1] - prev_centroid[1])**2)

            # Calculate IoU
            iou = self._calculate_iou(bbox, prev_bbox)

            # Determine score based on method
            if method == 'distance':
                score = distance  # Lower is better
                if distance < self.max_distance and distance < best_score:
                    best_score = distance
                    best_id = object_id
            elif method == 'iou':
                score = iou  # Higher is better
                if iou > 0.2 and iou > best_score:
                    best_score = iou
                    best_id = object_id
            elif method == 'combined':
                # Combined score (weighted sum)
                score = distance - (iou * 50)  # Lower is better
                if distance < self.max_distance and score < best_score:
                    best_score = score
                    best_id = object_id

        return best_id, best_score

    def update(self, detections):
        """
        Update tracker with new detections

        Args:
            detections: List of detection dictionaries from BeybladeDetector

        Returns:
            List of stabilized detections with consistent IDs
        """
        # If no detections, mark all existing objects as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                # If object has disappeared for too many frames, remove it
                if self.disappeared[object_id] > self.max_disappeared:
                    self._remove_object(object_id)

            # Return empty list since there's nothing to track
            return []

        # Initialize array of input centroids for the current frame
        current_objects = {}

        # Process new detections
        for detection in detections:
            bbox = detection['bbox']
            centroid = self._calculate_centroid(bbox)

            # Store detection info
            current_objects[len(current_objects)] = {
                'centroid': centroid,
                'bbox': bbox,
                'detection': detection
            }

        # If we currently aren't tracking any objects, register all new ones
        if len(self.objects) == 0:
            for i in range(len(current_objects)):
                self._register_object(current_objects[i])
        else:
            # Try to match current detections with existing objects
            object_ids = list(self.objects.keys())
            used_rows = set()
            used_cols = set()

            # First pass: Match using tracker IDs when available
            for detection_idx, detection_info in current_objects.items():
                detection = detection_info['detection']

                # If the detection already has a track_id that we're tracking
                if detection['track_id'] is not None and detection['track_id'] in self.objects:
                    object_id = detection['track_id']

                    # Update the existing object with this detection
                    self._update_object(object_id, detection_info)

                    # Mark as used
                    used_rows.add(object_id)
                    used_cols.add(detection_idx)

            # Second pass: For remaining unmatched detections, use distance/IoU
            for detection_idx, detection_info in current_objects.items():
                # Skip already matched detections
                if detection_idx in used_cols:
                    continue

                bbox = detection_info['bbox']

                # Get available tracks (not yet matched)
                available_tracks = {obj_id: {
                    'centroid': self.objects[obj_id],
                    'bbox': self._get_current_bbox(obj_id)
                } for obj_id in object_ids if obj_id not in used_rows}

                # Find best match
                best_id, _ = self._get_most_probable_match(bbox, available_tracks)

                if best_id is not None:
                    # Update the existing object with this detection
                    self._update_object(best_id, detection_info)

                    # Mark as used
                    used_rows.add(best_id)
                    used_cols.add(detection_idx)

            # Handle unmatched existing objects
            for object_id in object_ids:
                if object_id not in used_rows:
                    self.disappeared[object_id] += 1

                    # If object has disappeared for too many frames, remove it
                    if self.disappeared[object_id] > self.max_disappeared:
                        self._remove_object(object_id)

            # Register any new detections
            for detection_idx in range(len(current_objects)):
                if detection_idx not in used_cols:
                    self._register_object(current_objects[detection_idx])

        # Return stable detections
        return self._get_stable_detections()

    def _register_object(self, object_info):
        """Register a new object"""
        # Choose the lowest available ID
        if self.available_ids:
            object_id = min(self.available_ids)
            self.available_ids.remove(object_id)
        else:
            # If no IDs available, use the counter (fallback)
            object_id = self.next_object_id % self.max_objects
            self.next_object_id += 1

            # Remove the old object with this ID if it exists
            if object_id in self.objects:
                self._remove_object(object_id)

        self.objects[object_id] = object_info['centroid']
        self.disappeared[object_id] = 0

        # Initialize history
        self.history[object_id] = deque(maxlen=self.history_size)
        self.history[object_id].append((object_info['bbox'], object_info['centroid']))

        # Initialize class info
        detection = object_info['detection']
        self.class_info[object_id] = {
            'class_name': detection['class_name'],
            'class_id': detection['class_id'],
            'count': 1
        }

        # Update the detection's track_id
        object_info['detection']['track_id'] = object_id

    def _update_object(self, object_id, object_info):
        """Update an existing object"""
        # Reset disappeared counter
        self.disappeared[object_id] = 0

        # Update centroid
        self.objects[object_id] = object_info['centroid']

        # Store current and previous position for velocity calculation
        current_centroid = object_info['centroid']
        current_bbox = object_info['bbox']

        # Update history
        self.history[object_id].append((current_bbox, current_centroid))

        # Calculate velocity if we have enough history
        if len(self.history[object_id]) >= 2:
            self._calculate_velocity(object_id)
            self._check_if_stationary(object_id)

            # Check if outside arena (if arena bounds are defined)
            if self.arena_bounds is not None:
                self._check_if_outside_arena(object_id, current_bbox)

        # Update class info for stability
        detection = object_info['detection']
        if object_id in self.class_info:
            if detection['class_name'] == self.class_info[object_id]['class_name']:
                self.class_info[object_id]['count'] += 1
            else:
                # Keep track of most common class
                if self.class_info[object_id]['count'] > 5:
                    # Class is stable, keep it
                    detection['class_name'] = self.class_info[object_id]['class_name']
                    detection['class_id'] = self.class_info[object_id]['class_id']
                else:
                    # Class is not stable yet, update it
                    self.class_info[object_id] = {
                        'class_name': detection['class_name'],
                        'class_id': detection['class_id'],
                        'count': 1
                    }

        # Update the detection's track_id
        object_info['detection']['track_id'] = object_id

    def _remove_object(self, object_id):
        """Remove an object that has disappeared"""
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
        if object_id in self.history:
            del self.history[object_id]
        if object_id in self.class_info:
            del self.class_info[object_id]

        # Add the ID back to the available pool for recycling
        self.available_ids.add(object_id)

    def _get_current_bbox(self, object_id):
        """Get the current bounding box for an object"""
        if object_id in self.history and len(self.history[object_id]) > 0:
            return self.history[object_id][-1][0]
        return None

    def _get_stable_detections(self):
        """Get stable detections with consistent IDs"""
        stable_detections = []

        for object_id in self.objects:
            if object_id in self.history and len(self.history[object_id]) > 0:
                # Get the most recent detection
                bbox, _ = self.history[object_id][-1]

                # Create a stable detection
                detection = {
                    'bbox': bbox,
                    'track_id': object_id,
                }

                # Add class info if available
                if object_id in self.class_info:
                    detection['class_name'] = self.class_info[object_id]['class_name']
                    detection['class_id'] = self.class_info[object_id]['class_id']

                # Add confidence (use 1.0 if not available)
                detection['confidence'] = 1.0

                stable_detections.append(detection)

        return stable_detections

    def draw_tracks(self, frame, show_history=True, track_length=10):
        """
        Draw tracks on the frame

        Args:
            frame: Frame to draw on
            show_history: Whether to draw track history
            track_length: Number of previous positions to draw

        Returns:
            Frame with drawn tracks
        """
        for object_id, centroid in self.objects.items():
            # Draw track history
            if show_history and object_id in self.history:
                positions = list(self.history[object_id])[-track_length:]

                # Draw path line
                for i in range(1, len(positions)):
                    thickness = int(np.sqrt(64 / float(i + 1)) * 2)

                    # Ensure previous and current positions are integers
                    prev_pos = (int(positions[i-1][1][0]), int(positions[i-1][1][1]))
                    curr_pos = (int(positions[i][1][0]), int(positions[i][1][1]))

                    cv2.line(frame, prev_pos, curr_pos, (0, 0, 255), thickness)

        return frame

    def reset(self):
        """Reset the tracker completely"""
        self.objects = {}
        self.disappeared = {}
        self.history = {}
        self.class_info = {}
        self.available_ids = set(range(1, self.max_objects + 1))
        self.next_object_id = 1
        self.stationary_counts = {}
        self.velocities = {}
        self.is_stationary = {}
        self.outside_arena = {}

    def _calculate_velocity(self, object_id):
        """Calculate the current velocity of the tracked object"""
        if len(self.history[object_id]) < 2:
            # Not enough history to calculate velocity
            self.velocities[object_id] = 0
            return 0

        # Get current and previous positions
        _, current_centroid = self.history[object_id][-1]
        _, prev_centroid = self.history[object_id][-2]

        # Calculate distance between positions (pixel distance per frame)
        dx = current_centroid[0] - prev_centroid[0]
        dy = current_centroid[1] - prev_centroid[1]
        distance = np.sqrt(dx**2 + dy**2)

        # Store velocity
        self.velocities[object_id] = distance
        return distance

    def _check_if_stationary(self, object_id):
        """Check if an object is stationary based on its velocity"""
        if object_id not in self.velocities:
            self.is_stationary[object_id] = False
            self.stationary_counts[object_id] = 0
            return False

        velocity = self.velocities[object_id]

        # Check if velocity is below threshold
        if velocity <= self.stationary_threshold:
            # Increment stationary counter
            if object_id not in self.stationary_counts:
                self.stationary_counts[object_id] = 0

            self.stationary_counts[object_id] += 1

            # Mark as stationary if we've seen enough consecutive frames with low motion
            if self.stationary_counts[object_id] >= self.stationary_frames:
                self.is_stationary[object_id] = True
                return True
        else:
            # Reset counter if velocity is above threshold
            self.stationary_counts[object_id] = 0
            self.is_stationary[object_id] = False

        return self.is_stationary.get(object_id, False)

    def _check_if_outside_arena(self, object_id, bbox):
        """Check if a beyblade is outside the defined arena"""
        if self.arena_bounds is None:
            return False

        arena_x_min, arena_y_min, arena_x_max, arena_y_max = self.arena_bounds
        x1, y1, x2, y2 = bbox

        # Calculate the center of the bounding box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Check if center is outside arena
        is_outside = (center_x < arena_x_min or center_x > arena_x_max or
                      center_y < arena_y_min or center_y > arena_y_max)

        self.outside_arena[object_id] = is_outside
        return is_outside

    def set_arena_bounds(self, x_min, y_min, x_max, y_max):
        """Set the boundaries of the battle arena"""
        self.arena_bounds = (x_min, y_min, x_max, y_max)

    def is_game_over(self):
        """
        Check if the Beyblade battle is over

        Returns:
            tuple: (is_game_over, reason, loser_id)
                - is_game_over: True if the battle is over, False otherwise
                - reason: String explaining why the battle is over ('stopped' or 'exited_arena')
                - loser_id: ID of the losing Beyblade
        """
        # Require at least two tracked objects for a battle
        if len(self.objects) < 2:
            return False, None, None

        # Check if any Beyblade has stopped
        for object_id, is_stationary in self.is_stationary.items():
            if is_stationary:
                return True, "stopped", object_id

        # Check if any Beyblade has left the arena (if arena bounds are defined)
        if self.arena_bounds is not None:
            for object_id, is_outside in self.outside_arena.items():
                if is_outside:
                    return True, "exited_arena", object_id

        # Battle is still ongoing
        return False, None, None

    def get_game_over_details(self):
        """
        Get details about why the game is over

        Returns:
            tuple: (reason, loser_id)
                - reason: String explaining why the battle is over ('stopped' or 'exited_arena')
                - loser_id: ID of the losing Beyblade
        """
        is_over, reason, loser_id = self.is_game_over()
        return reason, loser_id

    def get_beyblade_info(self, beyblade_id):
        """
        Get information about a specific Beyblade

        Args:
            beyblade_id: ID of the Beyblade to get information for

        Returns:
            dict: Information about the Beyblade, including class name and position
        """
        if beyblade_id not in self.objects:
            return None

        info = {}
        if beyblade_id in self.class_info:
            info['class_name'] = self.class_info[beyblade_id]['class_name']

        if beyblade_id in self.history and len(self.history[beyblade_id]) > 0:
            bbox, centroid = self.history[beyblade_id][-1]
            info['bbox'] = bbox
            info['centroid'] = centroid

        if beyblade_id in self.velocities:
            info['velocity'] = self.velocities[beyblade_id]

        if beyblade_id in self.is_stationary:
            info['is_stationary'] = self.is_stationary[beyblade_id]

        return info

    def print_stationary_parameters(self):
        """
        Print the current stationary detection parameters

        This displays the threshold for velocity below which a Beyblade is considered stationary,
        and how many consecutive frames with low motion are required to confirm it has stopped
        """
        print("\n=== Beyblade Stationary Detection Parameters ===")
        print(f"Stationary threshold: {self.stationary_threshold} pixels/frame")
        print(f"Stationary frames required: {self.stationary_frames} consecutive frames")
        print("===============================================\n")

    def get_stationary_parameters(self):
        """
        Get the current stationary detection parameters as a dictionary

        Returns:
            dict: Dictionary containing the stationary detection parameters
        """
        return {
            "stationary_threshold": self.stationary_threshold,
            "stationary_frames": self.stationary_frames
        }

    def get_stationary_counts(self):
        """
        Get the current stationary frame counts for all tracked objects

        Returns:
            dict: Dictionary mapping object_id to count of consecutive stationary frames
        """
        return self.stationary_counts.copy()

