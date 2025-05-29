import os
import cv2
import numpy as np

from tqdm import tqdm
from ultralytics import YOLO
from src.beyblade_battle_analyzer.entity.config_entity import AnalyzeVideoConfig
from src.beyblade_battle_analyzer.components.beyblade_detector import BeybladeDetector
from src.beyblade_battle_analyzer.components.beyblade_tracker import BeybladeTracker


class VideoAnalyzer:
    def __init__(self, config: AnalyzeVideoConfig):
        """
        Initialize the VideoAnalyzer with the provided video path and model path.

        Args:
            video_path: Path to the video file to analyze
            model_path: Path to the model weights file
            output_dir: Directory where output results will be saved
        """
        self.config = config
        self.model = None
        self.video_capture = None
        self.frame_count = 0
        self.fps = 0
        self.frame_width = 0
        self.frame_height = 0

        # Initialize the Beyblade tracker for consistent detections
        self.tracker = BeybladeTracker(
            max_disappeared=15,
            max_distance=50,
            history_size=30,
            stationary_threshold=13,   # Maximum pixels/frame to be considered stationary
            stationary_frames=10      # Number of consecutive frames with low motion to confirm stopped
        )

        # Game state
        self.game_over = False
        self.game_over_reason = None
        self.loser_id = None

    def load_model(self):
        """
        Load the Beyblade detection model.
        """
        try:
            self.model = YOLO(self.config.model_path)
            print(f"Model loaded successfully from {self.config.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def open_video(self):
        """
        Open the video file and extract its metadata.
        """
        try:
            self.video_capture = cv2.VideoCapture(self.config.video_path)
            if not self.video_capture.isOpened():
                raise Exception(f"Error opening video file: {self.config.video_path}")

            # Get video properties
            self.frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            self.frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            print(f"Video opened successfully: {self.config.video_path}")
            print(f"Frame count: {self.frame_count}, FPS: {self.fps}, Resolution: {self.frame_width}x{self.frame_height}")
        except Exception as e:
            print(f"Error opening video: {e}")
            raise

    def analyze(self, start_frame=0, end_frame=None, display=True, save_output=True, conf_threshold=0.75, min_size=80,
                max_size=150):
        """
        Analyze the video and detect Beyblades.

        Args:
            start_frame: Frame number to start analysis from (default: 0)
            end_frame: Frame number to end analysis at (default: None, which means analyze until the end)
            display: Whether to display the frames during analysis (default: True)
            save_output: Whether to save the output video (default: True)
            conf_threshold: Minimum confidence threshold for detections (default: 0.4)
            min_size: Minimum size of detection box in pixels (default: 10)
            max_size: Maximum size of detection box in pixels (default: 300)
        """
        # Make sure the model is loaded
        if self.model is None:
            print("Model not loaded. Loading model...")
            self.load_model()

        # Make sure the video is opened
        if self.video_capture is None:
            print("Video not opened. Opening video...")
            self.open_video()
        elif not self.video_capture.isOpened():
            print("Video was closed. Reopening video...")
            self.open_video()

        # Double check that video was successfully opened
        if self.video_capture is None or not self.video_capture.isOpened():
            raise ValueError("Failed to open video for analysis. Please check the video path and try again.")

        if end_frame is None:
            end_frame = self.frame_count

        # Ensure start_frame and end_frame are valid
        start_frame = max(0, min(start_frame, self.frame_count - 1))
        end_frame = max(start_frame + 1, min(end_frame, self.frame_count))

        # Seek to the start frame
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Create output video writer if saving output
        output_video_writer = None
        if save_output:
            # Create output directory if it doesn't exist
            os.makedirs(self.config.output_dir, exist_ok=True)

            # Get video filename without path and extension
            video_name = os.path.splitext(os.path.basename(self.config.video_path))[0]
            output_video_path = os.path.join(self.config.output_dir, f"{video_name}_analyzed.mp4")

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video_writer = cv2.VideoWriter(output_video_path, fourcc, self.fps,
                                                 (self.frame_width, self.frame_height))
            print(f"Output video will be saved to: {output_video_path}")

        # Reset game state
        self.game_over = False
        self.game_over_reason = None
        self.loser_id = None

        # Auto-detect arena boundaries from the first frame
        self._detect_arena_boundaries()

        try:
            # Create progress bar
            progress_bar = tqdm(total=end_frame - start_frame, desc="Analyzing video")

            # Process frames
            for frame_idx in range(start_frame, end_frame):
                # Read frame
                ret, frame = self.video_capture.read()
                if not ret:
                    print(f"Error reading frame {frame_idx}")
                    break

                # Detect Beyblades
                detector = BeybladeDetector(self.model, frame)
                detections = detector.detect()

                # Apply confidence and size filtering
                filtered_detections = detector.filter_detections(
                    conf_threshold=conf_threshold,
                    min_size=min_size,
                    max_size=max_size
                )

                # Update tracker with filtered detections for consistent tracking
                stable_detections = self.tracker.update(filtered_detections)

                # Draw detections on frame
                annotated_frame = detector.draw_detections(detections=filtered_detections)
                annotated_frame = self.tracker.draw_tracks(annotated_frame, show_history=True, track_length=15)

                # Draw arena boundaries on frame
                annotated_frame = self._draw_arena_boundaries(annotated_frame)

                # Draw stationary thresholds on frame
                annotated_frame = self._draw_stationary_info(annotated_frame)

                # Check for game over condition
                game_is_over, reason, loser_id = self.tracker.is_game_over()
                if game_is_over:
                    self.game_over = True
                    self.game_over_reason = reason
                    self.loser_id = loser_id

                # Draw game status on frame
                annotated_frame = self._draw_game_status(annotated_frame)

                # If game is over, continue recording for a few more seconds to show the result
                if self.game_over and frame_idx >= end_frame - 90:  # Show result for ~3 seconds at 30fps
                    break

                # Display frame if requested
                if display:
                    cv2.imshow("Beyblade Analysis", annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break

                # Write frame to output video if requested
                if save_output and output_video_writer:
                    output_video_writer.write(annotated_frame)

                # Update progress bar
                progress_bar.update(1)

            # If game is over, add summary frames to the end of the video
            if self.game_over and save_output and output_video_writer:
                self._add_game_over_summary(output_video_writer)

            # Close progress bar
            progress_bar.close()
        finally:
            # Release resources
            if save_output and output_video_writer:
                output_video_writer.release()

            # Close display window
            if display:
                cv2.destroyAllWindows()

    def close(self):
        """
        Close the video file.
        """
        if self.video_capture:
            self.video_capture.release()

    def _draw_game_status(self, frame):
        """
        Draw game status on the frame, including which Beyblade is winning/losing

        Args:
            frame: Frame to draw on

        Returns:
            Frame with drawn game status
        """
        # Make a copy to avoid modifying the original frame
        annotated_frame = frame.copy()

        # Get frame dimensions
        frame_height, frame_width = annotated_frame.shape[:2]

        # Draw game status
        if self.game_over:
            # Game over banner
            cv2.rectangle(annotated_frame, (0, 0), (frame_width, 80), (0, 0, 0), -1)

            # Text for game over reason
            if self.game_over_reason == "stopped":
                message = "GAME OVER: Beyblade stopped spinning!"
            elif self.game_over_reason == "exited_arena":
                message = "GAME OVER: Beyblade left the arena!"
            else:
                message = "GAME OVER!"

            cv2.putText(annotated_frame, message, (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Show which Beyblade lost
            if self.loser_id is not None:
                loser_info = self.tracker.get_beyblade_info(self.loser_id)
                if loser_info and 'class_name' in loser_info:
                    loser_text = f"Loser: {loser_info['class_name']} (ID: {self.loser_id})"
                else:
                    loser_text = f"Loser: Beyblade ID {self.loser_id}"

                cv2.putText(annotated_frame, loser_text, (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Show battle status
            beyblades = []
            for object_id in self.tracker.objects:
                info = self.tracker.get_beyblade_info(object_id)
                if info:
                    beyblades.append((object_id, info))

            # Draw battle status panel
            if len(beyblades) >= 2:
                cv2.rectangle(annotated_frame, (0, 0), (frame_width, 40), (0, 0, 0), -1)
                cv2.putText(annotated_frame, "BEYBLADE BATTLE IN PROGRESS", (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        return annotated_frame

    def _add_game_over_summary(self, video_writer, duration_secs=3):
        """
        Add a game over summary to the end of the video

        Args:
            video_writer: OpenCV VideoWriter object
            duration_secs: Duration of the summary in seconds
        """
        if not self.game_over:
            return

        # Create a blank frame
        blank_frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

        # Calculate frames to add based on FPS
        frames_to_add = int(self.fps * duration_secs)

        # Get loser info
        loser_info = None
        if self.loser_id is not None:
            loser_info = self.tracker.get_beyblade_info(self.loser_id)

        # Create winner ID (the one that's not the loser)
        winner_id = None
        winner_info = None
        for obj_id in self.tracker.objects:
            if obj_id != self.loser_id:
                winner_id = obj_id
                winner_info = self.tracker.get_beyblade_info(winner_id)
                break

        # Add summary frames
        for _ in range(frames_to_add):
            # Create a copy of the blank frame
            summary_frame = blank_frame.copy()

            # Draw a dark background with gradient
            cv2.rectangle(summary_frame, (0, 0), (self.frame_width, self.frame_height), (30, 30, 30), -1)

            # Draw game over title
            cv2.putText(summary_frame, "GAME OVER",
                       (self.frame_width//2 - 150, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            # Draw reason
            if self.game_over_reason == "stopped":
                reason_text = "Beyblade stopped spinning"
            elif self.game_over_reason == "exited_arena":
                reason_text = "Beyblade left the arena"
            else:
                reason_text = "Battle ended"

            cv2.putText(summary_frame, reason_text,
                       (self.frame_width//2 - 180, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Draw winner/loser info
            y_pos = 250

            # Winner info
            if winner_id is not None:
                winner_name = winner_info.get('class_name', f"Beyblade {winner_id}") if winner_info else f"Beyblade {winner_id}"
                cv2.putText(summary_frame, f"Winner: {winner_name}",
                           (100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                y_pos += 60

            # Loser info
            if self.loser_id is not None:
                loser_name = loser_info.get('class_name', f"Beyblade {self.loser_id}") if loser_info else f"Beyblade {self.loser_id}"
                cv2.putText(summary_frame, f"Loser: {loser_name}",
                           (100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                y_pos += 60

                # Show reason for loss
                if self.game_over_reason == "stopped":
                    cv2.putText(summary_frame, f"Beyblade {self.loser_id} stopped spinning",
                              (100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                elif self.game_over_reason == "exited_arena":
                    cv2.putText(summary_frame, f"Beyblade {self.loser_id} exited the arena",
                              (100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            # Write the summary frame to the video
            video_writer.write(summary_frame)

    def _detect_arena_boundaries(self):
        """
        Automatically detect the arena boundaries from the first frame.
        This is a placeholder for potential future enhancement.
        Currently, users would need to manually set arena bounds if needed.
        """
        # Get the first frame
        if self.video_capture is None:
            self.open_video()

        # Reset to the beginning of the video
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_frame = self.video_capture.read()

        if not ret:
            print("Could not read first frame for arena detection")
            return

        # For now, we'll just use a default arena size in the middle of the frame
        # This could be enhanced with actual arena detection using computer vision techniques
        frame_height, frame_width = first_frame.shape[:2]
        center_x, center_y = frame_width // 2, frame_height // 2

        # Define arena as 80% of the frame size
        arena_size_x = int(frame_width * 0.8)
        arena_size_y = int(frame_height * 0.8)

        # Calculate arena boundaries
        x_min = center_x - arena_size_x // 2
        y_min = center_y - arena_size_y // 2
        x_max = center_x + arena_size_x // 2
        y_max = center_y + arena_size_y // 2

        # Set arena boundaries in tracker
        self.tracker.set_arena_bounds(x_min, y_min, x_max, y_max)

        # Reset video position
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _draw_arena_boundaries(self, frame):
        """
        Draw arena boundaries on the frame to visualize the battle area

        Args:
            frame: Frame to draw on

        Returns:
            Frame with drawn arena boundaries
        """
        # Make a copy to avoid modifying the original frame
        annotated_frame = frame.copy()

        # Check if arena bounds are defined
        if self.tracker.arena_bounds is not None:
            x_min, y_min, x_max, y_max = self.tracker.arena_bounds

            # Draw arena rectangle
            cv2.rectangle(annotated_frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)),
                         (0, 255, 255), 2)  # Yellow color for arena boundary

            # Add "Arena" label in the top-left corner of the boundary
            cv2.putText(annotated_frame, "Battle Arena", (int(x_min) + 10, int(y_min) + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return annotated_frame

    def _draw_stationary_info(self, frame):
        """
        Draw stationary detection parameters and current stationary frame counts on the frame

        Args:
            frame: Frame to draw on

        Returns:
            Frame with drawn stationary information
        """
        # Make a copy to avoid modifying the original frame
        annotated_frame = frame.copy()

        # Get frame dimensions
        frame_height, frame_width = annotated_frame.shape[:2]

        # Get stationary parameters
        params = self.tracker.get_stationary_parameters()
        threshold = params['stationary_threshold']
        frames_required = params['stationary_frames']

        # Create a semi-transparent background for better text visibility
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (frame_width - 280, frame_height - 90),
                     (frame_width - 10, frame_height - 10), (0, 0, 0), -1)
        # Apply transparency
        cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

        # Draw stationary parameters
        cv2.putText(annotated_frame, f"Stop Detection Settings:", (frame_width - 270, frame_height - 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated_frame, f"Threshold: {threshold} pixels/frame", (frame_width - 270, frame_height - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated_frame, f"Frames required: {frames_required}", (frame_width - 270, frame_height - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Get stationary counts for tracked objects and display them if they have any count
        counts = self.tracker.get_stationary_counts()
        if counts:
            y_pos = 80
            for obj_id, count in counts.items():
                if count > 0:  # Only show beyblades that have some stationary frames
                    info = self.tracker.get_beyblade_info(obj_id)
                    name = info.get('class_name', f"Beyblade {obj_id}") if info else f"Beyblade {obj_id}"
                    # Create a color that gets redder as the count approaches the required frames
                    progress = min(1.0, count / frames_required)
                    color = (0, 255 * (1 - progress), 255 * progress)  # Goes from green to red

                    # Display the stationary count on the left side
                    cv2.putText(annotated_frame, f"{name}: {count}/{frames_required} stationary frames",
                              (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    y_pos += 20

        return annotated_frame

