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
                class_id = int(box.cls[0])              # Get class ID
                class_name = result.names[class_id]     # Get class name

                # Create detection object
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                }

                self.detections.append(detection)

    def draw_detections(self, color=(0, 255, 0), thickness=2):
        """Draw bounding boxes on the frame for detected objects.

        :param color: RGB color tuple for the bounding box
        :param thickness: Thickness of the bounding box lines
        :return: Frame with drawn bounding boxes
        """

        # Create a copy of the frame to avoid modifying the original
        annotated_frame = self.frame.copy()

        # Draw bounding boxes and labels
        for detection in self.detections:
            # Extract coordinates
            x1, y1, x2, y2 = [int(coord) for coord in detection['bbox']]

            # Draw rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

            # Create label text with class name and confidence
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"

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

    def filter_by_box_size(self, min_size=10, max_size=None, relative=True):
        """Filter detections by bounding box size.

        :param min_size: Minimum size of bounding box (either absolute pixels or relative percentage)
        :param max_size: Maximum size of bounding box (either absolute pixels or relative percentage)
        :param relative: If True, min_size and max_size are percentages of frame dimensions
                         If False, they are absolute pixel values
        :return: Filtered list of detections
        """
        filtered = []
        frame_height, frame_width = self.frame.shape[:2]

        for detection in self.detections:
            x1, y1, x2, y2 = detection['bbox']
            box_width = x2 - x1
            box_height = y2 - y1

            # Calculate size relative to frame if needed
            if relative:
                rel_width = (box_width / frame_width) * 100
                rel_height = (box_height / frame_height) * 100
                # Use the smaller dimension for size comparison
                size = min(rel_width, rel_height)
            else:
                # Use the smaller dimension for size comparison
                size = min(box_width, box_height)

            # Check if box is within size constraints
            if min_size <= size and (max_size is None or size <= max_size):
                filtered.append(detection)

        return filtered

    def filter_by_class(self, class_name):
        """Filter detections by class name.

        :param class_name: Name of the class to filter for (e.g., 'spins')
        :return: Filtered list of detections
        """
        return [detection for detection in self.detections if detection['class_name'] == class_name]

