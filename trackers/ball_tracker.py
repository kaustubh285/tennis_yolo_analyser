from ultralytics import YOLO
from utils import log_function_call
import cv2
import pickle
import pandas as pd


class BallTracker:
    @log_function_call
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    @log_function_call
    def detect_frames(
        self,
        frames,
        stub_path,
        read_from_stub: bool = False,
    ) -> list:
        ball_detections = []

        if read_from_stub and stub_path is not None:
            try:
                with open(stub_path, "rb") as f:
                    ball_detections = pickle.load(f)
                return ball_detections
            except FileNotFoundError:
                print(f"Stub file {stub_path} not found. Processing frames...")

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    @log_function_call
    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]

            ball_dict[1] = result

        return ball_dict

    def draw_bbox(self, frames, ball_detections) -> list:
        """
        Draw bounding boxes on the frames based on ball detections.
        """
        for frame, ball_dict in zip(frames, ball_detections):
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    f"ball: {track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

        return frames

    @log_function_call
    def interpolate_ball(self, ball_positions):
        """
        Interpolate missing ball detections.
        """
        ball_positions = [x.get(1, []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(
            ball_positions, columns=["x1", "y1", "x2", "y2"]
        )
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        return [{1: x} for x in df_ball_positions.to_numpy().tolist()]
