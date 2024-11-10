import os
import sys
import torch
import os.path as osp

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
sys.path.append(root_path)
from super_resolution.inference.inference_sr_utils import RealEsrUpsamplerZoo


class RealEsr:

    def __init__(
        self,
        upscale=2,
        bg_upsampler_name="realesrgan",
        prefered_net_in_upsampler="RRDBNet",
    ):

        self.upscale = int(upscale)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ------------------------ set up background upsampler ------------------------
        self.upsampler_zoo = RealEsrUpsamplerZoo(
            upscale=self.upscale,
            bg_upsampler_name=bg_upsampler_name,
            prefered_net_in_upsampler=prefered_net_in_upsampler,
        )
        self.bg_upsampler = self.upsampler_zoo.bg_upsampler

    def __call__(self, img):
        # ---------------- restore/enhance image using the selected RealESR model ----------------
        sr_img = self.bg_upsampler.enhance(img, outscale=self.upscale)[0]

        return sr_img
