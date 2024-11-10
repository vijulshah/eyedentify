import os
import cv2
import sys
import torch
import numpy as np
import os.path as osp
from PIL import Image
from basicsr.utils import img2tensor
from basicsr.archs.srresnet_arch import MSRResNet

root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
sys.path.append(root_path)


class SRResNet:

    def __init__(self, upscale=2, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16):

        self.upscale = int(upscale)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ------------------ load model for img enhancement -------------------
        self.sr_model = MSRResNet(
            upscale=self.upscale,
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            num_feat=num_feat,
            num_block=num_block,
        ).to(self.device)

        ckpt_path = os.path.join(
            root_path,
            "super_resolution",
            "inference",
            "srresnet",
            "weights",
            f"SRResNet_{str(self.upscale)}x.pth",
        )
        loadnet = torch.load(ckpt_path, map_location=self.device)
        if "params_ema" in loadnet:
            keyname = "params_ema"
        else:
            keyname = "params"

        self.sr_model.load_state_dict(loadnet[keyname])
        self.sr_model.eval()

    @torch.no_grad()
    def __call__(self, img):
        img_tensor = img2tensor(imgs=img / 255.0, bgr2rgb=True, float32=True).unsqueeze(0).to(self.device)
        restored_img = self.sr_model(img_tensor)[0]
        restored_img = restored_img.permute(1, 2, 0).cpu().numpy()
        restored_img = (restored_img - restored_img.min()) / (restored_img.max() - restored_img.min())
        restored_img = (restored_img * 255).astype(np.uint8)
        restored_img = Image.fromarray(restored_img)
        restored_img = np.array(restored_img)
        sr_img = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)

        return sr_img
