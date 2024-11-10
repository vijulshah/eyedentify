import os
import cv2
import sys
import torch
import numpy as np
import os.path as osp
from PIL import Image
from basicsr.utils import img2tensor

root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
sys.path.append(root_path)
from super_resolution.inference.hat.hat_arch import HATArch


class HAT:

    def __init__(
        self,
        upscale=2,
        in_chans=3,
        img_size=(480, 640),
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    ):
        upscale = int(upscale)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ------------------ load model for img enhancement -------------------
        self.sr_model = HATArch(
            img_size=img_size,
            upscale=upscale,
            in_chans=in_chans,
            window_size=window_size,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            conv_scale=conv_scale,
            overlap_ratio=overlap_ratio,
            img_range=img_range,
            depths=depths,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            upsampler=upsampler,
            resi_connection=resi_connection,
        ).to(self.device)

        ckpt_path = os.path.join(
            root_path,
            "super_resolution",
            "inference",
            "hat",
            "weights",
            f"HAT_SRx{str(upscale)}_ImageNet-pretrain.pth",
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
