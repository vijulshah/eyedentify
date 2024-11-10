import os
import cv2
import sys
import argparse
import os.path as osp

root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
sys.path.append(root_path)

from super_resolution.inference.inference_hat import HAT
from super_resolution.inference.inference_gfpgan import GFPGAN
from super_resolution.inference.inference_realesr import RealEsr
from super_resolution.inference.inference_srresnet import SRResNet
from super_resolution.inference.inference_codeformer import CodeFormer

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Super-resolution of eye images")
    parser.add_argument("--sr_method", type=str, default="HAT", help="SR Model name")
    parser.add_argument("--eyes", type=str, default="left_eyes", help="Type of eyes (e.g., 'left_eyes')")
    parser.add_argument("--pid", type=str, default="1", help="Person ID (e.g., '1')")
    parser.add_argument("--session_id", type=str, default="1", help="Session ID (e.g., '1')")
    parser.add_argument("--frame_id", type=str, default="01", help="Frame ID (e.g., '01')")
    parser.add_argument("--upscale", type=int, default=2, help="Upscale factor for SRResNet (default: 2)")
    parser.add_argument(
        "--input_img_path",
        type=str,
        default=None,
        help="Custom input image path for the image. If not provided, a default path will be constructed.",
    )
    parser.add_argument(
        "--output_img_path",
        type=str,
        default=None,
        help="Custom output image path for the image. If not provided, a default path will be constructed.",
    )
    args = parser.parse_args()

    sr_method = args.sr_method
    eyes = args.eyes
    pid = args.pid
    session_id = args.session_id
    frame_id = args.frame_id
    upscale = args.upscale
    input_img_path = (
        args.input_img_path or f"local/data/EyeDentify/Wo_SR/{eyes}/{pid}/{session_id}/frame_{frame_id}.png"
    )
    output_img_path = args.output_img_path or f"local/rough_works/SR_imgs/{sr_method}"
    img = cv2.imread(os.path.join(root_path, input_img_path))

    if sr_method == "HAT":
        sr_model = HAT(upscale=upscale)
    elif sr_method == "GFPGAN":
        sr_model = GFPGAN(
            upscale=upscale,
            bg_upsampler_name="realesrgan",
            prefered_net_in_upsampler="RRDBNet",
        )
    elif sr_method == "SRResNet":
        sr_model = SRResNet(upscale=upscale)
    elif sr_method == "RealEsr":
        sr_model = RealEsr(
            upscale=upscale,
            bg_upsampler_name="realesrgan",
            prefered_net_in_upsampler="RRDBNet",
        )
    elif sr_method == "CodeFormer":
        sr_model = CodeFormer(upscale=upscale, fidelity_weight=1.0)
    sr_img = sr_model(img=img)

    saving_dir = os.path.join(root_path, output_img_path)
    os.makedirs(saving_dir, exist_ok=True)
    base_file_name = os.path.basename(input_img_path)
    cv2.imwrite(f"{saving_dir}/{base_file_name}", sr_img)
