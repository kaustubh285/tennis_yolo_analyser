from ultralytics import YOLO
from utils import log_function_call, get_bbox_center, measure_distance
import cv2
import pickle


class PlayerTracker:
    @log_function_call
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    @log_function_call
    def detect_frames(
        self,
        frames,
        read_from_stub: bool = False,
        stub_path: str = "player_detections.pkl",
    ) -> list:
        player_detections = []

        if read_from_stub and stub_path is not None:
            try:
                with open(stub_path, "rb") as f:
                    player_detections = pickle.load(f)
                return player_detections
            except FileNotFoundError:
                print(f"Stub file {stub_path} not found. Processing frames...")

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(player_detections, f)

        return player_detections

    @log_function_call
    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True, conf=0.2)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result

        return player_dict

    def draw_bbox(self, frames, player_detections) -> list:
        """
        Draw bounding boxes on the frames based on player detections.
        """
        for frame, player_dict in zip(frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    f"Player: {track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

        return frames

    def filter_players_near_court(self, player_detections, court_keypoints):
        player_detections_first_frame = player_detections[0]
        chosen_players = self.choose_players(
            court_keypoints, player_detections_first_frame
        )
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_detections.append(
                {
                    track_id: bbox
                    for track_id, bbox in player_dict.items()
                    if track_id in chosen_players
                }
            )
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_detections_first_frame):
        """
        Choose players who are close to the court based on their bounding boxes.
        """
        distances = []
        for track_id, bbox in player_detections_first_frame.items():

            min_distance_threshold = float("inf")
            bbox_center = get_bbox_center(bbox)

            for i in range(0, len(court_keypoints), 2):
                keypoint = (court_keypoints[i], court_keypoints[i + 1])
                distance = measure_distance(bbox_center, keypoint)
                if distance < min_distance_threshold:
                    min_distance_threshold = distance
                    distances.append((track_id, distance))

        distances.sort(key=lambda x: x[1])
        # Choose top 2 players closest to the court
        chosen_players = [x[0] for x in distances[:2]]

        return chosen_players
