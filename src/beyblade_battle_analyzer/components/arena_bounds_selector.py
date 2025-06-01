import cv2

from src.beyblade_battle_analyzer import logger
from src.beyblade_battle_analyzer.entity.config_entity import ArenaBoundsSelectorConfig


class ArenaBoundsSelector:
    """
    Class to handle the selection of arena bounds from a video.
    """

    def __init__(self, config: ArenaBoundsSelectorConfig):
        self.config = config
        self.start_x = -1
        self.start_y = -1
        self.end_x = -1
        self.end_y = -1
        self.drawing = False
        self.arena_bounds = None
        self.original_frame = None
        self.display_frame = None

    def _update_display(self):
        """
        Update the display frame with the current rectangle drawn by the user.
        """

        # Check if the original frame is available
        if self.original_frame is None:
            return

        # Create a copy of the original frame to draw on
        self.display_frame = self.original_frame.copy()

        # Draw the rectangle if the user is currently drawing
        if self.drawing:
            # Draw the rectangle on the display frame
            cv2.rectangle(self.display_frame, (self.start_x, self.start_y), (self.end_x, self.end_y), (0, 255, 0), 2)

            # Display the frame with the rectangle
            width = abs(self.end_x - self.start_x)
            height = abs(self.end_y - self.start_y)
            text = f"Size: {width}x{height}"
            cv2.putText(self.display_frame, text, (self.start_x, self.start_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

        # If the user has finished drawing, set the arena bounds
        if self.arena_bounds:
            # Draw the final arena bounds rectangle
            x1, y1, x2, y2 = self.arena_bounds
            cv2.rectangle(self.display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Draw the center point and display size and center coordinates
            width = x2 - x1
            height = y2 - y1
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Draw the center point
            cv2.circle(self.display_frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # Display the arena bounds information
            cv2.putText(self.display_frame, "ARENA",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(self.display_frame, f"Size: {width}x{height}",
                        (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(self.display_frame, f"Center: ({center_x}, {center_y})",
                        (x1, y2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Display the instructions on the frame
        instructions = [
            "INSTRUCTIONS:",
            "• Drag to select arena bounds",
            "• Right-click to reset",
            "• Press SPACE to confirm",
            "• Press ESC to cancel"
        ]

        y_offset = 30
        for i, instruction in enumerate(instructions):
            color = (255, 255, 255) if i == 0 else (200, 200, 200)
            thickness = 2 if i == 0 else 1
            cv2.putText(self.display_frame, instruction,
                        (10, y_offset + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

        cv2.imshow(self.config.window_name, self.display_frame)

    def _finalize_selection(self):
        """
        Finalize the selection of arena bounds and store them.
        """

        # Calculate the arena bounds from the start and end points
        x1 = min(self.start_x, self.end_x)
        y1 = min(self.start_y, self.end_y)
        x2 = max(self.start_x, self.end_x)
        y2 = max(self.start_y, self.end_y)

        # Store the arena bounds
        self.arena_bounds = (x1, y1, x2, y2)

        # Update the display with the finalized bounds
        self._update_display()

        logger.info(f"Arena bounds selected: {self.arena_bounds}")
        logger.info(f'Arena size: {x2 - x1}x{y2 - y1} pixels')

    def _reset_selection(self):
        """
        Reset the selection of arena bounds.
        """

        # Reset the drawing state and bounds
        self.arena_bounds = None
        self.drawing = False
        self.start_x = self.start_y = self.end_x = self.end_y = -1

        # Update the display to show the original frame
        self._update_display()

        logger.info("Arena bounds selection reset")

    def _mouse_callback(self, event, x, y, flags, param):
        """
        Mouse callback function to handle mouse events for selecting arena bounds.
        """

        # Handle mouse events for drawing rectangle
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing rectangle
            self.drawing = True
            self.start_x, self.start_y = x, y
            self.end_x, self.end_y = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            # Update rectangle end point while drawing
            if self.drawing:
                self.end_x, self.end_y = x, y
                self._update_display()
        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing rectangle
            self.drawing = False
            self.end_x, self.end_y = x, y
            self._finalize_selection()
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Reset the selection on right-click
            self._reset_selection()

    def select_bounds(self):
        """
        Method to select the arena bounds from the input video.
        This method should implement the logic to display the video and allow the user to select bounds.
        """

        # Open the video file
        cap = cv2.VideoCapture(self.config.input_video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {self.config.input_video_path}")
            return None

        # Read the first frame
        ret, frame = cap.read()
        cap.release()

        # Check if the frame was read successfully
        if not ret:
            logger.error("Could not read first frame from video")
            return None

        # Store the original and display frames
        self.original_frame = frame.copy()
        self.display_frame = frame.copy()

        # Create a named window and set the mouse callback
        cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.config.window_name, 1000, 700)
        cv2.setMouseCallback(self.config.window_name, self._mouse_callback)
        logger.info('Arena bounds selection started. Use the mouse to draw a rectangle around the arena.')
        logger.info('Controls: Drag=Select, Right-click=Reset, SPACE=Confirm, ESC=Cancel')

        # Update the display with the original frame
        self._update_display()

        # Main loop to display the video and handle user input
        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC key - cancel
                logger.info("Arena selection cancelled")
                break
            elif key == 32:  # SPACE key - confirm
                if self.arena_bounds:
                    logger.info(f"Arena bounds confirmed: {self.arena_bounds}")
                    cv2.destroyAllWindows()
                    return self.arena_bounds
                else:
                    logger.warning("No arena bounds selected. Please drag to select an area.")
                    # Display a message on screen
                    text = "No arena bounds selected. Please drag to select an area."
                    cv2.putText(self.display_frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)
                    cv2.imshow(self.config.window_name, self.display_frame)
                    # Reset the display after 2 seconds
                    cv2.waitKey(2000)
                    self._update_display()
            elif key == ord('r'):  # R key - reset
                self._reset_selection()
            elif key == ord('q'):  # Q key - quit
                logger.info("Arena selection cancelled")
                break

        cv2.destroyAllWindows()
        return None

