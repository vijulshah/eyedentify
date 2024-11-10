import cv2
import torch
from PIL import Image
from transformers import pipeline
from facexlib.utils.face_restoration_helper import FaceRestoreHelper


class ExtractorFacexlib:

    def __init__(self, upscale=1):

        self.upscale = int(upscale)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ========== Face Extraction ==========
        self.face_restorer_helper = FaceRestoreHelper(
            upscale=self.upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=self.device,
            model_rootpath="gfpgan/weights",
        )
        self.landmarks = None

        # ========== Eyes Extraction ==========
        self.pipe = pipeline(
            "image-classification",
            model="dima806/closed_eyes_image_detection",
            device=self.device,
        )
        self.blink_thresh = 0.5

    def extract_face(self, image):

        h, w = image.shape[0:2]
        self.face_restorer_helper.clean_all()
        self.face_restorer_helper.read_image(image)
        landmarks_array = self.face_restorer_helper.get_face_landmarks_5(
            only_keep_largest=True, only_center_face=False, eye_dist_threshold=5
        )
        bbox = self.face_restorer_helper.det_faces

        if len(bbox) == 0:
            print("No face detected")
            return None

        self.landmarks = self.face_restorer_helper.all_landmarks_5[0]

        # Ensure bbox coordinates are integers
        x1, y1, x2, y2, confidence = map(int, bbox[:])

        # Calculate the center of the bounding box
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # Calculate the coordinates for the 256x256 crop
        crop_x1 = max(center_x - 128 * self.upscale, 0)
        crop_y1 = max(center_y - 128 * self.upscale, 0)
        crop_x2 = min(center_x + 128 * self.upscale, w)
        crop_y2 = min(center_y + 128 * self.upscale, h)

        cropped_face = image[crop_y1:crop_y2, crop_x1:crop_x2]

        return cropped_face

    def extract_eyes(self, image, blink_detection=False):

        if self.landmarks is None or len(self.landmarks) == 0:
            if self.extract_face(image) is None:
                return

        left_eye_x, left_eye_y = map(int, self.landmarks[0])
        right_eye_x, right_eye_y = map(int, self.landmarks[1])

        left_eye = image[
            left_eye_y - 8 * self.upscale : left_eye_y + 8 * self.upscale,
            left_eye_x - 16 * self.upscale : left_eye_x + 16 * self.upscale,
        ]

        right_eye = image[
            right_eye_y - 8 * self.upscale : right_eye_y + 8 * self.upscale,
            right_eye_x - 16 * self.upscale : right_eye_x + 16 * self.upscale,
        ]

        return {
            "left_eye": left_eye,
            "right_eye": right_eye,
        }

    def detect_blink(self, image):

        eyes_data = self.extract_eyes(image)
        left_eye = cv2.cvtColor(eyes_data["left_eye"], cv2.COLOR_RGB2GRAY)
        right_eye = cv2.cvtColor(eyes_data["right_eye"], cv2.COLOR_RGB2GRAY)

        left_eye = Image.fromarray(left_eye)
        preds_left = self.pipe(left_eye)
        if preds_left[0]["label"] == "closeEye":
            closed_left = preds_left[0]["score"] > self.blink_thresh
        else:
            closed_left = preds_left[1]["score"] > self.blink_thresh

        right_eye = Image.fromarray(right_eye)
        preds_right = self.pipe(right_eye)
        if preds_right[0]["label"] == "closeEye":
            closed_right = preds_right[0]["score"] > self.blink_thresh
        else:
            closed_right = preds_right[1]["score"] > self.blink_thresh

        if closed_left or closed_right:
            print("preds_left = ", preds_left)
            print("preds_right = ", preds_right)

        return closed_left or closed_right

    @staticmethod
    def segment_iris(iris_img):

        # Convert RGB image to grayscale
        iris_img_gray = cv2.cvtColor(iris_img, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur for denoising
        iris_img_blur = cv2.GaussianBlur(iris_img_gray, (5, 5), 0)

        # Perform adaptive thresholding
        _, iris_img_mask = cv2.threshold(iris_img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert the mask
        segmented_mask = cv2.bitwise_not(iris_img_mask)
        segmented_mask = cv2.cvtColor(segmented_mask, cv2.COLOR_GRAY2RGB)
        segmented_iris = cv2.bitwise_and(iris_img, segmented_mask)

        return {
            "segmented_iris": segmented_iris,
            "segmented_mask": segmented_mask,
        }

    def extract_iris(self, image):

        eyes_data = self.extract_eyes(image)

        if eyes_data is None:
            return

        left_iris_segmented_data = self.segment_iris(eyes_data["left_eye"])
        right_iris_segmented_data = self.segment_iris(eyes_data["right_eye"])

        return {
            "left_iris": {
                "img": eyes_data["left_eye"],
                "segmented_iris": left_iris_segmented_data["segmented_iris"],
                "segmented_mask": left_iris_segmented_data["segmented_mask"],
            },
            "right_iris": {
                "img": eyes_data["right_eye"],
                "segmented_iris": right_iris_segmented_data["segmented_iris"],
                "segmented_mask": right_iris_segmented_data["segmented_mask"],
            },
        }
