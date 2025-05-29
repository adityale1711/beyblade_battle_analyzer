import cv2


class BeybladeDetector:
    def __init__(self, model, frame):
        """ Initializes the BeybladeDetector with the specified model.
        :param model: YOLO model for detecting Beyblades.
        :param frame: Frame from the video to be analyzed.
        """
        self.model = model
        self.frame = frame
        self.results = None
        self.detections = []

    def detect(self):
        """ Detects Beyblades in the video using the YOLO model.
        :return: List of detected Beyblades.
        """
        # Run inference on the frame
        self.results = self.model.track(self.frame, tracker='botsort.yaml', verbose=False)

        # Process detections
        self._process_detections()

        # Print available classes for debugging
        class_counts = {}
        for detection in self.detections:
            class_name = detection['class_name']
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1

        return self.detections

    def _process_detections(self):
        """Process detection results and extract relevant information."""
        self.detections = []

        # Extract detections from results
        for result in self.results:
            boxes = result.boxes
            for box in boxes:
                # Extract coordinates, confidence, and class
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Get box coordinates
                confidence = float(box.conf[0])         # Get confidence score
                class_id = int(box.cls[0])              # Get class name
                class_name = result.names[class_id]     # Get class name

                # Extract track ID (will be None if tracking is not enabled)
                track_id = None
                if hasattr(box, 'id') and box.id is not None:
                    track_id = int(box.id[0])

                # Create detection object
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name,
                    'track_id': track_id  # Add track ID to detection
                }

                self.detections.append(detection)

    def draw_detections(self, detections=None, color=(0, 255, 0), thickness=2):
        """Draw bounding boxes on the frame for detected objects.

        :param detections: List of detections to draw (default: None, which uses self.detections)
        :param color: RGB color tuple for the bounding box
        :param thickness: Thickness of the bounding box lines
        :return: Frame with drawn bounding boxes
        """

        # Create a copy of the frame to avoid modifying the original
        annotated_frame = self.frame.copy()

        # Use provided detections or fall back to self.detections
        detections_to_draw = detections if detections is not None else self.detections

        # Draw bounding boxes and labels
        for detection in detections_to_draw:
            # Extract coordinates
            x1, y1, x2, y2 = [int(coord) for coord in detection['bbox']]

            # Calculate box dimensions
            box_width = x2 - x1
            box_height = y2 - y1
            box_size = min(box_width, box_height)  # The smaller dimension

            # Draw rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

            # Create label text with size info, class name, confidence and track ID (if available)
            if detection['track_id'] is not None:
                label = f"ID:{detection['track_id']} | {box_size} | {detection['class_name']}: {detection['confidence']:.2f}"
            else:
                label = f"{box_size} | {detection['class_name']}: {detection['confidence']:.2f}"

            # Draw label background
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame,
                          (x1, y1 - text_size[1] - 5),
                          (x1 + text_size[0], y1),
                          color,
                          -1)

            # Draw text
            cv2.putText(annotated_frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1)

        return annotated_frame

    def get_detection_count(self):
        """Get the number of detected objects.

        :return: Number of detections
        """
        return len(self.detections)

    def filter_by_confidence(self, threshold=0.5):
        """Filter detections by confidence threshold.

        :param threshold: Minimum confidence threshold
        :return: Filtered list of detections
        """
        return [detection for detection in self.detections if detection['confidence'] >= threshold]

    def filter_detections(self, conf_threshold=0.5, min_size=20, max_size=200):
        """Filter detections by both confidence threshold and box size.

        Args:
            conf_threshold: Minimum confidence threshold
            min_size: Minimum size of bounding box in pixels
            max_size: Maximum size of bounding box in pixels

        Returns:
            List of detections that pass both filters
        """
        # First filter by confidence
        conf_filtered = self.filter_by_confidence(threshold=conf_threshold)

        # Apply size filtering to the confidence-filtered results
        filtered = []

        for detection in conf_filtered:
            x1, y1, x2, y2 = detection['bbox']
            box_width = x2 - x1
            box_height = y2 - y1

            # Use the smaller dimension for size comparison (in pixels)
            size = min(box_width, box_height)

            # Check if box is within size constraints
            if min_size <= size and (max_size is None or size <= max_size):
                filtered.append(detection)

        return filtered
