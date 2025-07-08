from utils import log_function_call, read_video, save_video
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector


@log_function_call
def main(input_path, output_path):

    video_frames = read_video(input_path)

    # Initialize Trackers
    player_tracker = PlayerTracker(model_path="yolov8m")
    ball_tracker = BallTracker(model_path="models/best.pt")
    court_line_detector = CourtLineDetector(model_path="models/keypoints_model.pth")

    # Detections
    player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stub=True,
        stub_path="tracker_stubs/player_detections.pkl",
    )
    ball_detections = ball_tracker.detect_frames(
        video_frames,
        read_from_stub=True,
        stub_path="tracker_stubs/ball_detections.pkl",
    )
    ball_detections = ball_tracker.interpolate_ball(ball_detections)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Draw
    updated_video = player_tracker.draw_bbox(video_frames, player_detections)
    updated_video = ball_tracker.draw_bbox(updated_video, ball_detections)
    updated_video = court_line_detector.draw_keypoints_video(
        updated_video, court_keypoints
    )

    save_video(updated_video, output_path)


if __name__ == "__main__":
    file_name = "input_video_small"
    input_path = f"input/{file_name}.mp4"
    output_path = f"output/{file_name}.avi"
    main(input_path, output_path)
