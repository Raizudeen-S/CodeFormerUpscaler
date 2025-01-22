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

from basicsr.utils.registry import ARCH_REGISTRY

pretrain_model_url = {
    "restoration": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
}


def process_video(
    input_path="",
    fidelity_weight=1,
    upscale=2,
    has_aligned=False,
    only_center_face=False,
    draw_box=False,
    detection_model="retinaface_resnet50",
    bg_upsampler=None,
    suffix=None,
    save_video_fps=None,
):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = get_device()
    # ------------------------ input & output ------------------------
    w = fidelity_weight
    from basicsr.utils.video_util import VideoReader, VideoWriter

    # input_img_list = []
    print("Reading video...")
    vidreader = VideoReader(input_path)
    print("Reading video done.")
    img_path = vidreader.get_frame()
    test_img_num = vidreader.__len__()
    audio = vidreader.get_audio()
    fps = vidreader.get_fps() if save_video_fps is None else save_video_fps
    video_name = os.path.basename(input_path)[:-4]
    result_root = f"results/{video_name}_{w}"
    input_video = True
    # vidreader.close()

    # test_img_num = len(input_img_list)
    if test_img_num == 0:
        raise FileNotFoundError("No input image/video is found...\n" "\tNote that --input_path for video should end with .mp4|.mov|.avi")

    # ------------------ set up CodeFormer restorer -------------------
    net = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=["32", "64", "128", "256"]).to(device)

    # ckpt_path = 'weights/CodeFormer/codeformer.pth'
    ckpt_path = load_file_from_url(url=pretrain_model_url["restoration"], model_dir="weights/CodeFormer", progress=True, file_name=None)
    checkpoint = torch.load(ckpt_path)["params_ema"]
    net.load_state_dict(checkpoint)
    net.eval()

    face_helper = FaceRestoreHelper(upscale, face_size=512, crop_ratio=(1, 1), det_model=detection_model, save_ext="png", use_parse=True, device=device)

    # -------------------- start to processing ---------------------
    i = 0
    print(f"Processing {test_img_num} frames...")
    while img_path is not None:
        # clean all the intermediate results to process the next image
        face_helper.clean_all()

        if isinstance(img_path, str):
            img_name = os.path.basename(img_path)
            basename, ext = os.path.splitext(img_name)
            print(f"[{i+1}/{test_img_num}] Processing: {img_name}")
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        else:  # for video processing
            basename = str(i).zfill(6)
            img_name = f"{video_name}_{basename}" if input_video else basename
            print(f"[{i+1}/{test_img_num}] Processing: {img_name}")
            img = img_path

        if has_aligned:
            # the input faces are already cropped and aligned
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_helper.is_gray = is_gray(img, threshold=10)
            if face_helper.is_gray:
                print("Grayscale input: True")
            face_helper.cropped_faces = [img]
        else:
            face_helper.read_image(img)
            # get face landmarks for each face
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
                    output = net(cropped_face_t, w=w, adain=True)[0]
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
            # upsample the background
            if bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
            else:
                bg_img = None
            face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=draw_box)

        # save faces
        for idx, (cropped_face, restored_face) in enumerate(zip(face_helper.cropped_faces, face_helper.restored_faces)):
            # save cropped face
            if not has_aligned:
                save_crop_path = os.path.join(result_root, "cropped_faces", f"{basename}_{idx:02d}.png")
                imwrite(cropped_face, save_crop_path)
            # save restored face
            if has_aligned:
                save_face_name = f"{basename}.png"
            else:
                save_face_name = f"{basename}_{idx:02d}.png"
            if suffix is not None:
                save_face_name = f"{save_face_name[:-4]}_{suffix}.png"
            save_restore_path = os.path.join(result_root, "restored_faces", save_face_name)
            imwrite(restored_face, save_restore_path)

        # save restored img
        if not has_aligned and restored_img is not None:
            if suffix is not None:
                basename = f"{basename}_{suffix}"
            save_restore_path = os.path.join(result_root, "final_results", f"{basename}.png")
            imwrite(restored_img, save_restore_path)
        i += 1
        img_path = vidreader.get_frame()
    vidreader.close()

    # save enhanced video
    if input_video:
        print("Video Saving...")
        # Define paths
        img_list = sorted(glob.glob(os.path.join(result_root, "final_results", "*.[jp][pn]g")))

        if not img_list:
            raise ValueError("No images found in the specified path.")

        # Get frame dimensions from the first image
        first_frame = cv2.imread(img_list[0])
        height, width = first_frame.shape[:2]

        # Set video name and save path
        if suffix is not None:
            video_name = f"{video_name}_{suffix}"
        save_restore_path = os.path.join(result_root, f"{video_name}.mp4")

        # Initialize video writer
        vidwriter = VideoWriter(save_restore_path, height, width, fps, audio)

        # Process and write each frame one by one
        for img_path in img_list:
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Warning: Failed to read {img_path}")
                continue
            vidwriter.write_frame(frame)

        # Finalize and close the video writer
        vidwriter.close()
        print(f"Video saved at {save_restore_path}")

        return save_restore_path


if __name__ == "__main__":
    process_video(input_path="inpu.mp4")
