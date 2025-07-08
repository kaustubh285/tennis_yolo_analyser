from utils import log_function_call, read_video, save_video
from trackers import PlayerTracker


@log_function_call
def main(input_path, output_path):
    video_frames = read_video(input_path)
    player_tracker = PlayerTracker(model_path="yolov8m")

    player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stub=False,
        stub_path="tracker_stubs/player_detections.pkl",
    )

    updated_video = player_tracker.draw_bbox(video_frames, player_detections)
    save_video(updated_video, output_path)


if __name__ == "__main__":
    file_name = "input_video_small"
    input_path = f"input/{file_name}.mp4"
    output_path = f"output/{file_name}.avi"
    main(input_path, output_path)
