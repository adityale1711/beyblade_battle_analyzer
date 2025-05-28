import cv2

from src.beyblade_battle_analyzer.entity.config_entity import AnalyzeVideoConfig


class AnalyzeVideo:
    def __init__(self, config: AnalyzeVideoConfig):
        """
        Initializes the AnalyzeVideo class with the provided configuration.

        :param config: AnalyzeVideoConfig object containing the configuration settings.
        """
        self.config = config

    def analyze(self):
        # Opens the video file specified in the configuration
        cap = cv2.VideoCapture(self.config.video_path)

        # Get the frames rates (FPS) of the video
        fps = cap.get(cv2.CAP_PROP_FPS)

        while True:
            # Read a frame from the video
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow('Beyblade Battle Analyzer', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
