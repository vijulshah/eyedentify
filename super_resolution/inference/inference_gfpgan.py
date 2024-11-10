import os
import sys
import torch
import os.path as osp
from gfpgan import GFPGANer
from basicsr.utils.download_util import load_file_from_url

root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
sys.path.append(root_path)
from super_resolution.inference.inference_sr_utils import RealEsrUpsamplerZoo


class GFPGAN:

    def __init__(
        self,
        upscale=2,
        bg_upsampler_name="realesrgan",
        prefered_net_in_upsampler="RRDBNet",
    ):

        upscale = int(upscale)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ------------------------ set up background upsampler ------------------------
        upsampler_zoo = RealEsrUpsamplerZoo(
            upscale=upscale,
            bg_upsampler_name=bg_upsampler_name,
            prefered_net_in_upsampler=prefered_net_in_upsampler,
        )
        bg_upsampler = upsampler_zoo.bg_upsampler

        # ------------------------ load model ------------------------
        gfpgan_weights_path = os.path.join(root_path, "super_resolution", "inference", "gfpgan", "weights")
        gfpgan_model_path = os.path.join(gfpgan_weights_path, "GFPGANv1.3.pth")

        if not os.path.isfile(gfpgan_model_path):
            url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
            gfpgan_model_path = load_file_from_url(
                url=url,
                model_dir=gfpgan_weights_path,
                progress=True,
                file_name="GFPGANv1.3.pth",
            )

        self.sr_model = GFPGANer(
            upscale=upscale,
            bg_upsampler=bg_upsampler,
            model_path=gfpgan_model_path,
            device=device,
        )

    def __call__(self, img):
        # ------------------------ restore/enhance image using GFPGAN model ------------------------
        cropped_faces, sr_faces, sr_img = self.sr_model.enhance(img)

        return sr_img
