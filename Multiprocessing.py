import os
import cv2
import argparse
import glob
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import gpu_is_available, get_device
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray
from multiprocessing import Process
from math import ceil

from basicsr.utils.registry import ARCH_REGISTRY

pretrain_model_url = {
    "restoration": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
}

device = get_device()

net = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=["32", "64", "128", "256"]).to(device)

# ckpt_path = 'weights/CodeFormer/codeformer.pth'
ckpt_path = load_file_from_url(url=pretrain_model_url["restoration"], model_dir="weights/CodeFormer", progress=True, file_name=None)
checkpoint = torch.load(ckpt_path)["params_ema"]
net.load_state_dict(checkpoint)
net.eval()


def image_process(
    input_img_list,
    test_img_num,
    result_root,
    video_name,
    fidelity_weight=1,
    upscale=2,
    has_aligned=False,
    detection_model="retinaface_resnet50",
    only_center_face=False,
    draw_box=False,
    suffix=None,
):
    # ------------------ set up CodeFormer restorer ------------------
    face_helper = FaceRestoreHelper(upscale, face_size=512, crop_ratio=(1, 1), det_model=detection_model, save_ext="png", use_parse=True, device=device)
    for i, img_path in enumerate(input_img_list):
        # clean all the intermediate results to process the next image
        face_helper.clean_all()

        if isinstance(img_path, str):
            img_name = os.path.basename(img_path)
            basename, ext = os.path.splitext(img_name)
            print(f"[{i+1}/{test_img_num}] Processing: {img_name} {img_path}")
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        else:  # for video processing
            basename = str(i).zfill(6)
            img_name = f"{video_name}_{basename}"
            print(f"[{i+1}/{test_img_num}] Processing: {img_name}")
            img = img_path

        face_helper.read_image(img)
        num_det_faces = face_helper.get_face_landmarks_5(only_center_face=only_center_face, resize=640, eye_dist_threshold=5)
        print(f"\tdetect {num_det_faces} faces")

        # align and warp each face
        face_helper.align_warp_face()

        # face restoration for each cropped face
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    output = net(cropped_face_t, w=fidelity_weight, adain=True)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except Exception as error:
                print(f"\tFailed inference for CodeFormer: {error}")
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

            restored_face = restored_face.astype("uint8")
            face_helper.add_restored_face(restored_face, cropped_face)

        # paste_back
        if not has_aligned:
            bg_img = None
            face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=draw_box)

        # save restored img
        if not has_aligned and restored_img is not None:
            if suffix is not None:
                basename = f"{basename}_{suffix}"
            save_restore_path = os.path.join(result_root, f"{basename}.png")
            imwrite(restored_img, save_restore_path)


def process_images(images, result_root, video_name, part_num):
    """
    Function to process a chunk of images and save them in the corresponding result directory.
    """
    part_result_root = os.path.join(result_root, f"part_{part_num}")
    os.makedirs(part_result_root, exist_ok=True)
    test_img_num = len(images)
    print(f"Processing part {part_num} with {test_img_num} images...")
    image_process(images, test_img_num, part_result_root, f"{video_name}_part_{part_num}")
    print(f"Part {part_num} completed!")


def process_video(input_path="", fidelity_weight=1, suffix=None, save_video_fps=None, num_cores = 2):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------ input & output ------------------------
    w = fidelity_weight
    from basicsr.utils.video_util import VideoReader, VideoWriter

    input_img_list = []
    vidreader = VideoReader(input_path)
    image = vidreader.get_frame()
    while image is not None:
        input_img_list.append(image)
        image = vidreader.get_frame()
    audio = vidreader.get_audio()
    fps = vidreader.get_fps() if save_video_fps is None else save_video_fps
    video_name = os.path.basename(input_path)[:-4]
    result_root = f"outputs/{video_name}"
    # input_video = True
    vidreader.close()

    test_img_num = len(input_img_list)
    if test_img_num == 0:
        raise FileNotFoundError("No input image/video is found...\n" "\tNote that --input_path for video should end with .mp4|.mov|.avi")

    # -------------------- start to processing ---------------------
    
    chunk_size = ceil(test_img_num / num_cores)
    processes = []

    print(f"Splitting {test_img_num} images into {num_cores} parts...")

    for i in range(num_cores):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, test_img_num)
        chunk = input_img_list[start_idx:end_idx]

        if chunk:  # Ensure chunk is not empty
            p = Process(target=process_images, args=(chunk, result_root, video_name, i + 1))
            processes.append(p)
            p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Final Video generation or Appending

    output_path = os.path.join("outputs", f"{video_name}_final_video.mp4")

    part_folders = sorted([os.path.join(result_root, d) for d in os.listdir(result_root) if d.startswith("part_")])
    frame_paths = []

    # Collect all frame paths in order
    for part_folder in part_folders:
        frames = sorted([os.path.join(part_folder, f) for f in os.listdir(part_folder) if f.endswith(".png")])
        frame_paths.extend(frames)

    if not frame_paths:
        raise FileNotFoundError("No frames found for concatenation.")
    video_writer = cv2.VideoWriter(output_path, fps, audio)
    print(f"Writing final video to {output_path}...")

    # Write each frame to the video
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    video_writer.release()
    print("Final video saved successfully!")


if __name__ == "__main__":
    process_video(input_path="inputq.mp4")