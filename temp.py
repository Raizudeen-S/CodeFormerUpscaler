import os
from math import ceil
import cv2
from basicsr.utils.video_util import VideoReader, VideoWriter


def divide_video_into_n_videos(input_path, output_folder, n_parts):
    """
    Divides a video into `n_parts` smaller videos.

    Args:
        input_path (str): Path to the input video.
        output_folder (str): Directory to store the smaller videos.
        n_parts (int): Number of parts to divide the video into.
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists

    # Initialize video reader
    vidreader = VideoReader(input_path)
    total_frames = vidreader.__len__()
    fps = vidreader.get_fps()
    frame_height, frame_width = vidreader.get_dimensions()
    chunk_size = ceil(total_frames / n_parts)

    print(f"Total frames: {total_frames}, FPS: {fps}, Chunk size: {chunk_size}")

    for part in range(n_parts):
        start_idx = part * chunk_size
        end_idx = min((part + 1) * chunk_size, total_frames)

        output_video_path = os.path.join(output_folder, f"part_{part + 1}.mp4")
        print(f"Processing part {part + 1}: Frames {start_idx} to {end_idx}")

        # Initialize video writer for the current part
        vidwriter = VideoWriter(output_video_path, frame_height, frame_width, fps, audio=None)

        for frame_idx in range(start_idx, end_idx):
            frame = vidreader.get_frame()
            if frame is None:
                break
            vidwriter.write_frame(frame)

        vidwriter.close()
        print(f"Part {part + 1} saved to {output_video_path}")

    vidreader.close()
    print(f"Video divided into {n_parts} parts successfully!")


if __name__ == "__main__":
    input_path = "inputq.mp4"
    output_folder = "outputs"
    n_parts = 3  # Divide the video into 3 smaller videos

    divide_video_into_n_videos(input_path, output_folder, n_parts)