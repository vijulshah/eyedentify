import os
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.utils.download_util import load_file_from_url

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RealEsrUpsamplerZoo:

    def __init__(
        self,
        upscale=2,
        bg_upsampler_name="realesrgan",
        prefered_net_in_upsampler="RRDBNet",
    ):

        self.upscale = int(upscale)

        # ------------------------ set up background upsampler ------------------------
        weights_path = os.path.join(ROOT_DIR, "super_resolution", "inference", f"{bg_upsampler_name}", "weights")

        if bg_upsampler_name == "realesrgan":
            model = self.get_prefered_net(prefered_net_in_upsampler, upscale)
            if self.upscale == 2:
                model_path = os.path.join(weights_path, "RealESRGAN_x2plus.pth")
                url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
            elif self.upscale == 4:
                model_path = os.path.join(weights_path, "RealESRGAN_x4plus.pth")
                url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            else:
                raise Exception(f"{bg_upsampler_name} model not available for upscaling x{str(self.upscale)}")
        elif bg_upsampler_name == "realesrnet":
            model = self.get_prefered_net(prefered_net_in_upsampler, upscale)
            if self.upscale == 4:
                model_path = os.path.join(weights_path, "RealESRNet_x4plus.pth")
                url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
            else:
                raise Exception(f"{bg_upsampler_name} model not available for upscaling x{str(self.upscale)}")
        elif bg_upsampler_name == "anime":
            model = self.get_prefered_net(prefered_net_in_upsampler, upscale)
            if self.upscale == 4:
                model_path = os.path.join(weights_path, "RealESRGAN_x4plus_anime_6B.pth")
                url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
            else:
                raise Exception(f"{bg_upsampler_name} model not available for upscaling x{str(self.upscale)}")
        else:
            raise Exception(f"No model implemented for: {bg_upsampler_name}")

        # ------------------------ load background upsampler model ------------------------
        if not os.path.isfile(model_path):
            model_path = load_file_from_url(url=url, model_dir=weights_path, progress=True, file_name=None)

        self.bg_upsampler = RealESRGANer(
            scale=int(upscale),
            model_path=model_path,
            model=model,
            tile=0,
            tile_pad=0,
            pre_pad=0,
            half=False,
        )

    @staticmethod
    def get_prefered_net(prefered_net_in_upsampler, upscale=2):
        if prefered_net_in_upsampler == "RRDBNet":
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=int(upscale),
            )
        elif prefered_net_in_upsampler == "SRVGGNetCompact":
            model = SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=16,
                upscale=int(upscale),
                act_type="prelu",
            )
        else:
            raise Exception(f"No net named: {prefered_net_in_upsampler} implemented!")
        return model
