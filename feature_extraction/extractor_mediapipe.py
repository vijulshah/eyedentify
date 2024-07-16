import cv2
import torch
import warnings
import numpy as np
from PIL import Image
from math import sqrt
import mediapipe as mp
from transformers import pipeline

warnings.filterwarnings("ignore")


class ExtractorMediaPipe:

    def __init__(self, upscale=1):

        self.upscale = int(upscale)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ========== Face Extraction ==========
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            static_image_mode=True,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # ========== Eyes Extraction ==========
        self.RIGHT_EYE = [
            362,
            382,
            381,
            380,
            374,
            373,
            390,
            249,
            263,
            466,
            388,
            387,
            386,
            385,
            384,
            398,
        ]
        self.LEFT_EYE = [
            33,
            7,
            163,
            144,
            145,
            153,
            154,
            155,
            133,
            173,
            157,
            158,
            159,
            160,
            161,
            246,
        ]
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

        # ========== Iris Extraction ==========
        self.RIGHT_IRIS = [474, 475, 476, 477]
        self.LEFT_IRIS = [469, 470, 471, 472]

    def extract_face(self, image):

        tmp_image = image.copy()
        results = self.face_detector.process(tmp_image)

        if not results.detections:
            # print("No face detected")
            return None
        else:
            bboxC = results.detections[0].location_data.relative_bounding_box
            ih, iw, _ = image.shape

            # Get bounding box coordinates
            x, y, w, h = (
                int(bboxC.xmin * iw),
                int(bboxC.ymin * ih),
                int(bboxC.width * iw),
                int(bboxC.height * ih),
            )

            # Calculate the center of the bounding box
            center_x = x + w // 2
            center_y = y + h // 2

            # Calculate new bounds ensuring they fit within the image dimensions
            half_size = 128 * self.upscale
            x1 = max(center_x - half_size, 0)
            y1 = max(center_y - half_size, 0)
            x2 = min(center_x + half_size, iw)
            y2 = min(center_y + half_size, ih)

            # Adjust x1, x2, y1, and y2 to ensure the cropped region is exactly (256 * self.upscale) x (256 * self.upscale)
            if x2 - x1 < (256 * self.upscale):
                if x1 == 0:
                    x2 = min((256 * self.upscale), iw)
                elif x2 == iw:
                    x1 = max(iw - (256 * self.upscale), 0)

            if y2 - y1 < (256 * self.upscale):
                if y1 == 0:
                    y2 = min((256 * self.upscale), ih)
                elif y2 == ih:
                    y1 = max(ih - (256 * self.upscale), 0)

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
    def landmarksDetection(image, results, draw=False):
        image_height, image_width = image.shape[:2]
        mesh_coordinates = [
            (int(point.x * image_width), int(point.y * image_height))
            for point in results.multi_face_landmarks[0].landmark
        ]
        if draw:
            [cv2.circle(image, i, 2, (0, 255, 0), -1) for i in mesh_coordinates]
        return mesh_coordinates

    @staticmethod
    def euclideanDistance(point, point1):
        x, y = point
        x1, y1 = point1
        distance = sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
        return distance

    def blinkRatio(self, landmarks, right_indices, left_indices):

        right_eye_landmark1 = landmarks[right_indices[0]]
        right_eye_landmark2 = landmarks[right_indices[8]]

        right_eye_landmark3 = landmarks[right_indices[12]]
        right_eye_landmark4 = landmarks[right_indices[4]]

        left_eye_landmark1 = landmarks[left_indices[0]]
        left_eye_landmark2 = landmarks[left_indices[8]]

        left_eye_landmark3 = landmarks[left_indices[12]]
        left_eye_landmark4 = landmarks[left_indices[4]]

        right_eye_horizontal_distance = self.euclideanDistance(
            right_eye_landmark1, right_eye_landmark2
        )
        right_eye_vertical_distance = self.euclideanDistance(
            right_eye_landmark3, right_eye_landmark4
        )

        left_eye_vertical_distance = self.euclideanDistance(
            left_eye_landmark3, left_eye_landmark4
        )
        left_eye_horizontal_distance = self.euclideanDistance(
            left_eye_landmark1, left_eye_landmark2
        )

        right_eye_ratio = right_eye_vertical_distance / right_eye_horizontal_distance
        left_eye_ratio = left_eye_vertical_distance / left_eye_horizontal_distance

        eyes_ratio = (right_eye_ratio + left_eye_ratio) / 2

        return eyes_ratio

    def extract_eyes_regions(self, image, landmarks, eye_indices):
        h, w, _ = image.shape
        points = [
            (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
            for idx in eye_indices
        ]

        x_min = min([p[0] for p in points])
        x_max = max([p[0] for p in points])
        y_min = min([p[1] for p in points])
        y_max = max([p[1] for p in points])

        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        target_width = 32 * self.upscale
        target_height = 16 * self.upscale

        x1 = max(center_x - target_width // 2, 0)
        y1 = max(center_y - target_height // 2, 0)
        x2 = x1 + target_width
        y2 = y1 + target_height

        if x2 > w:
            x1 = w - target_width
            x2 = w
        if y2 > h:
            y1 = h - target_height
            y2 = h

        return image[y1:y2, x1:x2]

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

        tmp_face = image.copy()
        results = self.face_mesh.process(tmp_face)

        if results.multi_face_landmarks is None:
            return None

        face_landmarks = results.multi_face_landmarks[0].landmark

        left_eye = self.extract_eyes_regions(image, face_landmarks, self.LEFT_EYE)
        right_eye = self.extract_eyes_regions(image, face_landmarks, self.RIGHT_EYE)
        blinked = False

        if blink_detection:
            mesh_coordinates = self.landmarksDetection(image, results, False)
            eyes_ratio = self.blinkRatio(
                mesh_coordinates, self.RIGHT_EYE, self.LEFT_EYE
            )
            if (
                eyes_ratio > self.blink_lower_thresh
                and eyes_ratio <= self.blink_upper_thresh
            ):
                print(
                    "I think person blinked. eyes_ratio = ",
                    eyes_ratio,
                    "Confirming with ViT model...",
                )
                blinked = self.blink_detection_model(
                    left_eye=left_eye, right_eye=right_eye
                )
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
    def segment_iris(iris_img):

        # Convert RGB image to grayscale
        iris_img_gray = cv2.cvtColor(iris_img, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur for denoising
        iris_img_blur = cv2.GaussianBlur(iris_img_gray, (5, 5), 0)

        # Perform adaptive thresholding
        _, iris_img_mask = cv2.threshold(
            iris_img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Invert the mask
        segmented_mask = cv2.bitwise_not(iris_img_mask)
        segmented_mask = cv2.cvtColor(segmented_mask, cv2.COLOR_GRAY2RGB)
        segmented_iris = cv2.bitwise_and(iris_img, segmented_mask)

        return {
            "segmented_iris": segmented_iris,
            "segmented_mask": segmented_mask,
        }

    def extract_iris(self, image):

        ih, iw, _ = image.shape
        tmp_face = image.copy()
        results = self.face_mesh.process(tmp_face)

        if results.multi_face_landmarks is None:
            return None

        mesh_coordinates = self.landmarksDetection(image, results, False)
        mesh_points = np.array(mesh_coordinates)

        (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[self.LEFT_IRIS])
        (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[self.RIGHT_IRIS])

        # Crop the left iris to be exactly 16*upscaled x 16*upscaled
        l_x1 = max(int(l_cx) - (8 * self.upscale), 0)
        l_y1 = max(int(l_cy) - (8 * self.upscale), 0)
        l_x2 = min(int(l_cx) + (8 * self.upscale), iw)
        l_y2 = min(int(l_cy) + (8 * self.upscale), ih)

        cropped_left_iris = image[l_y1:l_y2, l_x1:l_x2]

        left_iris_segmented_data = self.segment_iris(
            cv2.cvtColor(cropped_left_iris, cv2.COLOR_BGR2RGB)
        )

        # Crop the right iris to be exactly 16*upscaled x 16*upscaled
        r_x1 = max(int(r_cx) - (8 * self.upscale), 0)
        r_y1 = max(int(r_cy) - (8 * self.upscale), 0)
        r_x2 = min(int(r_cx) + (8 * self.upscale), iw)
        r_y2 = min(int(r_cy) + (8 * self.upscale), ih)

        cropped_right_iris = image[r_y1:r_y2, r_x1:r_x2]

        right_iris_segmented_data = self.segment_iris(
            cv2.cvtColor(cropped_right_iris, cv2.COLOR_BGR2RGB)
        )

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
