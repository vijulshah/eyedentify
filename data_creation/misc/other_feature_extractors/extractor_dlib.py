import cv2
import dlib
import torch
import os.path as osp
from PIL import Image
from imutils import face_utils
from transformers import pipeline
from scipy.spatial import distance as dist

dlib.DLIB_USE_CUDA = torch.cuda.is_available()
root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))


class ExtractorDlib:

    def __init__(self, upscale=1):

        self.upscale = int(upscale)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ========== Face Extraction ==========
        self.frontal_face_detector = dlib.get_frontal_face_detector()
        self.face_landmarks_predictor = dlib.shape_predictor(f"{root_path}/data/shape_predictor_68_face_landmarks.dat")

        # ========== Eyes Extraction ==========
        # Eye landmarks
        (self.L_start, self.L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.R_start, self.R_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        # FULL_POINTS = list(range(0, 68))
        # FACE_POINTS = list(range(17, 68))
        # JAWLINE_POINTS = list(range(0, 17))
        # RIGHT_EYEBROW_POINTS = list(range(17, 22))
        # LEFT_EYEBROW_POINTS = list(range(22, 27))
        # NOSE_POINTS = list(range(27, 36))
        # RIGHT_EYE_POINTS = list(range(36, 42))
        # LEFT_EYE_POINTS = list(range(42, 48))
        # MOUTH_OUTLINE_POINTS = list(range(48, 61))
        # MOUTH_INNER_POINTS = list(range(61, 68))

        # https://huggingface.co/dima806/closed_eyes_image_detection
        # https://www.kaggle.com/code/dima806/closed-eye-image-detection-vit
        self.pipe = pipeline(
            "image-classification",
            model="dima806/closed_eyes_image_detection",
            device=self.device,
        )
        self.blink_lower_thresh = 0.22
        self.blink_upper_thresh = 0.25
        self.blink_confidence = 0.50

    def extract_face(self, image):

        faces = self.frontal_face_detector(image)

        if len(faces) == 0:
            return None

        face = faces[0]
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        ih, iw, _ = image.shape

        # Desired size of the cropped image
        crop_size = 256 * self.upscale

        # Calculate the center of the face
        center_x, center_y = x + w // 2, y + h // 2

        # Calculate the cropping box coordinates
        x1 = max(center_x - crop_size // 2, 0)
        y1 = max(center_y - crop_size // 2, 0)
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        # Ensure the cropping box is within the image boundaries
        if x2 > iw:
            x1 = iw - crop_size
            x2 = iw
        if y2 > ih:
            y1 = ih - crop_size
            y2 = ih

        # Crop the face
        cropped_face = image[y1:y2, x1:x2]

        # bicubic upsampling
        # if self.upscale != 1:
        #     cropped_face = cv2.resize(
        #         cropped_face,
        #         (256 * self.upscale, 256 * self.upscale),
        #         interpolation=cv2.INTER_CUBIC,
        #     )

        return cropped_face

    @staticmethod
    def calculate_EAR(eye):

        # calculate the vertical distances
        # euclidean distance is basically
        # the same when you calculate the
        # hypotenuse in a right triangle
        y1 = dist.euclidean(eye[1], eye[5])
        y2 = dist.euclidean(eye[2], eye[4])

        # calculate the horizontal distance
        x1 = dist.euclidean(eye[0], eye[3])

        # calculate the EAR
        EAR = (y1 + y2) / x1

        return EAR

    def blink_detection_model(self, left_eye, right_eye):

        left_eye = cv2.cvtColor(left_eye, cv2.COLOR_RGB2GRAY)
        left_eye = Image.fromarray(left_eye)
        preds_left = self.pipe(left_eye)
        if preds_left[0]["label"] == "closeEye":
            closed_left = preds_left[0]["score"] >= self.blink_confidence
        else:
            closed_left = preds_left[1]["score"] >= self.blink_confidence

        right_eye = cv2.cvtColor(right_eye, cv2.COLOR_RGB2GRAY)
        right_eye = Image.fromarray(right_eye)
        preds_right = self.pipe(right_eye)
        if preds_right[0]["label"] == "closeEye":
            closed_right = preds_right[0]["score"] >= self.blink_confidence
        else:
            closed_right = preds_right[1]["score"] >= self.blink_confidence

        print("preds_left = ", preds_left)
        print("preds_right = ", preds_right)

        return closed_left or closed_right

    def extract_eyes(self, image, blink_detection=False):

        detected_faces = self.frontal_face_detector(image)

        if len(detected_faces) == 0:
            print("No face found! Skipped eyes extraction!")
            return None

        shape = self.face_landmarks_predictor(image, detected_faces[0])

        target_width = 32 * self.upscale
        target_height = 16 * self.upscale

        # Calculate the centers of the left and right eyes
        left_eye_center_x = (shape.part(36).x + shape.part(39).x) // 2
        left_eye_center_y = (shape.part(37).y + shape.part(40).y) // 2

        right_eye_center_x = (shape.part(42).x + shape.part(45).x) // 2
        right_eye_center_y = (shape.part(43).y + shape.part(46).y) // 2

        # Calculate the bounding boxes for cropping
        left_x1 = max(left_eye_center_x - target_width // 2, 0)
        left_x2 = left_x1 + target_width
        left_y1 = max(left_eye_center_y - target_height // 2, 0)
        left_y2 = left_y1 + target_height

        right_x1 = max(right_eye_center_x - target_width // 2, 0)
        right_x2 = right_x1 + target_width
        right_y1 = max(right_eye_center_y - target_height // 2, 0)
        right_y2 = right_y1 + target_height

        # Ensure the crop is within image boundaries
        left_eye = image[left_y1:left_y2, left_x1:left_x2]
        right_eye = image[right_y1:right_y2, right_x1:right_x2]

        blinked = False

        if blink_detection:
            # converting the shape class directly
            # to a list of (x,y) coordinates
            shape = face_utils.shape_to_np(shape)

            # parsing the landmarks list to extract
            # lefteye and righteye landmarks--#
            lefteye_lm = shape[self.L_start : self.L_end]
            righteye_lm = shape[self.R_start : self.R_end]

            # Calculate the EAR
            left_EAR = self.calculate_EAR(lefteye_lm)
            right_EAR = self.calculate_EAR(righteye_lm)

            # Avg of left and right eye EAR
            eyes_ratio = (left_EAR + right_EAR) / 2

            if eyes_ratio > self.blink_lower_thresh and eyes_ratio <= self.blink_upper_thresh:
                print(
                    "I think person blinked. eyes_ratio = ",
                    eyes_ratio,
                    "Confirming with ViT model...",
                )
                blinked = self.blink_detection_model(left_eye=left_eye, right_eye=right_eye)
                if blinked:
                    print("Yes, person blinked. Confirmed by model")
                else:
                    print("No, person didn't blinked. False Alarm")
            elif eyes_ratio <= self.blink_lower_thresh:
                blinked = True
                print("Surely person blinked. eyes_ratio = ", eyes_ratio)
            else:
                blinked = False

        return {"left_eye": left_eye, "right_eye": right_eye, "blinked": blinked}

    @staticmethod
    def center_crop(image, crop_width=16, crop_height=16):
        # Get the dimensions of the image
        height, width = image.shape[:2]

        # Calculate the center of the image
        center_x, center_y = width // 2, height // 2

        # Calculate the top left and bottom right coordinates of the crop
        start_x = center_x - crop_width // 2
        start_y = center_y - crop_height // 2
        end_x = start_x + crop_width
        end_y = start_y + crop_height

        # Crop the image
        cropped_image = image[start_y:end_y, start_x:end_x]

        return cropped_image

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

        cropped_left_iris = self.center_crop(eyes_data["left_eye"], 16 * self.upscale, 16 * self.upscale)
        cropped_right_iris = self.center_crop(eyes_data["right_eye"], 16 * self.upscale, 16 * self.upscale)

        left_iris_segmented_data = self.segment_iris(cropped_left_iris)
        right_iris_segmented_data = self.segment_iris(cropped_right_iris)

        return {
            "left_iris": {
                "img": cropped_left_iris,
                "segmented_iris": left_iris_segmented_data["segmented_iris"],
                "segmented_mask": left_iris_segmented_data["segmented_mask"],
            },
            "right_iris": {
                "img": cropped_right_iris,
                "segmented_iris": right_iris_segmented_data["segmented_iris"],
                "segmented_mask": right_iris_segmented_data["segmented_mask"],
            },
        }
