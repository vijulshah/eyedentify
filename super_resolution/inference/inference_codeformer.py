import os
import cv2
import sys
import torch
import os.path as osp
from basicsr.utils import img2tensor, tensor2img
from torchvision.transforms.functional import normalize
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
sys.path.append(root_path)
from super_resolution.inference.inference_sr_utils import RealEsrUpsamplerZoo
from super_resolution.inference.codeformer.codeformer_arch import CodeFormerArch


class CodeFormer:

    def __init__(
        self,
        upscale=2,
        bg_upsampler_name="realesrgan",
        prefered_net_in_upsampler="RRDBNet",
        fidelity_weight=0.8,
    ):

        self.upscale = int(upscale)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fidelity_weight = fidelity_weight

        # ------------------------ set up background upsampler ------------------------
        upsampler_zoo = RealEsrUpsamplerZoo(
            upscale=self.upscale,
            bg_upsampler_name=bg_upsampler_name,
            prefered_net_in_upsampler=prefered_net_in_upsampler,
        )
        self.bg_upsampler = upsampler_zoo.bg_upsampler

        # ------------------ set up FaceRestoreHelper -------------------
        gfpgan_weights_path = os.path.join(root_path, "super_resolution", "inference", "gfpgan", "weights")
        self.face_restorer_helper = FaceRestoreHelper(
            upscale_factor=self.upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=self.device,
            # model_rootpath="gfpgan/weights",
            model_rootpath=gfpgan_weights_path,
        )

        # ------------------ load model -------------------
        self.sr_model = CodeFormerArch().to(self.device)
        ckpt_path = os.path.join(
            root_path, "super_resolution", "inference", "codeformer", "weights", "codeformer_v0.1.0.pth"
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

        bg_img = self.bg_upsampler.enhance(img, outscale=self.upscale)[0]

        self.face_restorer_helper.clean_all()
        self.face_restorer_helper.read_image(img)
        self.face_restorer_helper.get_face_landmarks_5(
            only_keep_largest=True, only_center_face=False, eye_dist_threshold=5
        )
        self.face_restorer_helper.align_warp_face()

        if len(self.face_restorer_helper.cropped_faces) > 0:

            cropped_face = self.face_restorer_helper.cropped_faces[0]

            cropped_face_t = img2tensor(imgs=cropped_face / 255.0, bgr2rgb=True, float32=True)
            normalize(
                tensor=cropped_face_t,
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                inplace=True,
            )
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            # ------------------- restore/enhance image using CodeFormerArch model -------------------
            output = self.sr_model(cropped_face_t, w=self.fidelity_weight, adain=True)[0]

            restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            restored_face = restored_face.astype("uint8")

            self.face_restorer_helper.add_restored_face(restored_face)
            self.face_restorer_helper.get_inverse_affine(None)

            sr_img = self.face_restorer_helper.paste_faces_to_input_image(upsample_img=bg_img)
        else:
            sr_img = bg_img

        return sr_img
