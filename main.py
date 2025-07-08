from utils import log_function_call, read_video, save_video


@log_function_call
def main(input_path, output_path):
    video_frames = read_video(input_path)

    save_video(video_frames, output_path)


if __name__ == "__main__":
    input_path = "input/input_video.mp4"
    output_path = "output/output_video.avi"
    main(input_path, output_path)
