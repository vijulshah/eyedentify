import sys
import cv2
import torch
import numpy as np
import os.path as osp
from PIL import Image
from math import sqrt
import mediapipe as mp
from transformers import pipeline
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
sys.path.append(root_path)

from data_creation.features_extraction.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2


class ExtractorMediaPipe:

    def __init__(self, upscale=1):

        self.upscale = int(upscale)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ========== Face Extraction ==========
        # https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_detection.md#solution-apis
        self.face_detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
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
        # self.blink_lower_thresh = 0.20
        # self.blink_upper_thresh = 0.22
        self.blink_lower_thresh = 0.22
        self.blink_upper_thresh = 0.25
        self.blink_confidence = 0.50

        # ========== Iris Extraction ==========
        self.RIGHT_IRIS = [474, 475, 476, 477]
        self.LEFT_IRIS = [469, 470, 471, 472]

        self.use_hf_pipeline_for_depth_model = False
        self.load_depth_model()

    def load_depth_model(self):
        if self.use_hf_pipeline_for_depth_model:
            self.depth_image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
            self.depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        else:
            model_configs = {
                "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
                "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
                "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
                "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
            }
            encoder = "vits"  # or 'vits', 'vitb', 'vitg'
            self.depth_model = DepthAnythingV2(**model_configs[encoder])
            self.depth_model.load_state_dict(
                torch.load(
                    f"{root_path}/data_creation/features_extraction/DepthAnythingV2/checkpoints/depth_anything_v2_{encoder}.pth",
                    map_location=self.device,
                )
            )
            self.depth_model = self.depth_model.to(self.device).eval()

    def get_depth_map(self, image, retun_img=False):
        if self.use_hf_pipeline_for_depth_model:
            inputs = self.depth_image_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.depth_model(**inputs)
                predicted_depth = outputs.predicted_depth
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            depth = prediction.squeeze().cpu().numpy()
        else:
            depth = self.depth_model.infer_image(image)

        if retun_img:
            formatted = (depth * 255 / np.max(depth)).astype("uint8")
            depth_img = Image.fromarray(formatted)
            return depth_img
        else:
            return depth

    def extract_face_regions(self, image, bboxC, image_depth=None):
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
        face_depth = None

        if image_depth is not None:
            face_depth = image_depth[y1:y2, x1:x2]

        return cropped_face, face_depth

    def extract_face(self, image, image_depth=None):

        tmp_image = image.copy()
        results = self.face_detector.process(tmp_image)
        if not results.detections:
            return None, None
        bboxC = results.detections[0].location_data.relative_bounding_box
        cropped_face, face_depth = self.extract_face_regions(image, bboxC, image_depth)

        return cropped_face, face_depth

    @staticmethod
    def landmarksDetection(image, results, draw=False):
        """
        Since mediapipe returns the landmarks in normalized values, we need to convert them into pixels values. This function turns normalized coordinates into pixel coordinates.
        Normalized to Pixel Coordinates:
        In the Face Mesh we get, 468 landmarks, so have to loop through each landmark, we will have x, and y values, for conversion purpose we need to multiply the width to x, and height to y, results would be pixel coordinates
        """
        image_height, image_width = image.shape[:2]
        mesh_coordinates = [
            (int(point.x * image_width), int(point.y * image_height))
            for point in results.multi_face_landmarks[0].landmark
        ]
        if draw:
            [cv2.circle(image, i, 1, (0, 255, 255), 1) for i in mesh_coordinates]
        return mesh_coordinates

    @staticmethod
    def euclideanDistance(point, point1):
        x, y = point
        x1, y1 = point1
        distance = sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
        return distance

    @staticmethod
    def l3_norm(point1, point2, p=3):
        # Calculate the L3 norm (Minkowski distance with p=3)
        return sum(abs(p1 - p2) ** p for p1, p2 in zip(point1, point2)) ** (1 / p)
        # return np.linalg.norm(np.array(point1) - np.array(point2), ord=p)

    def calculate_ear_def(self, landmarks, eye_indices, p=3):
        # Numerator
        numerator = (
            self.l3_norm(landmarks[eye_indices[3]], landmarks[eye_indices[13]])
            + self.l3_norm(landmarks[eye_indices[4]], landmarks[eye_indices[12]])
            + self.l3_norm(landmarks[eye_indices[5]], landmarks[eye_indices[11]])
        )

        # Denominator
        denominator = p * self.l3_norm(landmarks[eye_indices[0]], landmarks[eye_indices[8]])

        # Final EAR_def calculation
        ear_def = numerator / denominator

        return ear_def

    # Combination of these ideas and codes for robustness
    # https://dl.acm.org/doi/fullHtml/10.1145/3558884.3558890
    # https://github.com/Shakirsadiq6/Blink_Detection_Python
    def blinkRatioWithL3Norm(self, landmarks, right_indices, left_indices):

        left_EAR = self.calculate_ear_def(landmarks, right_indices)
        right_EAR = self.calculate_ear_def(landmarks, left_indices)

        return left_EAR, right_EAR

    def blinkRatioWithEuclideanDist(self, landmarks, right_indices, left_indices):

        right_eye_landmark1 = landmarks[right_indices[0]]
        right_eye_landmark2 = landmarks[right_indices[8]]

        right_eye_landmark3 = landmarks[right_indices[12]]
        right_eye_landmark4 = landmarks[right_indices[4]]

        left_eye_landmark1 = landmarks[left_indices[0]]
        left_eye_landmark2 = landmarks[left_indices[8]]

        left_eye_landmark3 = landmarks[left_indices[12]]
        left_eye_landmark4 = landmarks[left_indices[4]]

        right_eye_horizontal_distance = self.euclideanDistance(right_eye_landmark1, right_eye_landmark2)
        right_eye_vertical_distance = self.euclideanDistance(right_eye_landmark3, right_eye_landmark4)

        left_eye_vertical_distance = self.euclideanDistance(left_eye_landmark3, left_eye_landmark4)
        left_eye_horizontal_distance = self.euclideanDistance(left_eye_landmark1, left_eye_landmark2)

        right_eye_ratio = right_eye_vertical_distance / right_eye_horizontal_distance
        left_eye_ratio = left_eye_vertical_distance / left_eye_horizontal_distance

        return left_eye_ratio, right_eye_ratio

    def extract_eyes_regions(self, image, landmarks, eye_indices, image_depth=None):
        h, w, _ = image.shape
        points = [(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in eye_indices]

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

        eye = image[y1:y2, x1:x2]
        eye_depth = None
        if image_depth is not None:
            eye_depth = image_depth[y1:y2, x1:x2]

        return eye, eye_depth

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

    def create_outline_and_segment(self, mesh_coordinates, image, bboxC, on_eyes=True, on_iris=True, image_depth=None):
        image_copy = image.copy()
        outline_thickness = 1
        outline_color = (0, 0, 255)
        line_type = cv2.LINE_AA
        mesh_points = np.array(mesh_coordinates)
        cropped_face, face_depth = self.extract_face_regions(image, bboxC, image_depth)

        full_imgs = {"wo_outlines": image}
        face_imgs = {"wo_outlines": cropped_face, "depth": face_depth}

        segmented_images = {
            "left_eye": {},
            "right_eye": {},
            "left_iris": {},
            "right_iris": {},
        }

        if on_eyes:
            # Left Eye Outline
            left_eye_points = mesh_points[self.LEFT_EYE].astype(np.int32)
            right_eye_points = mesh_points[self.RIGHT_EYE].astype(np.int32)

            cv2.polylines(
                image_copy,
                [left_eye_points],
                isClosed=True,
                color=outline_color,
                thickness=outline_thickness,
                lineType=line_type,
            )
            cv2.polylines(
                image_copy,
                [right_eye_points],
                isClosed=True,
                color=outline_color,
                thickness=outline_thickness,
                lineType=line_type,
            )

            cropped_face, face_depth = self.extract_face_regions(image_copy, bboxC, image_depth)
            full_imgs["eyes_outlined"] = image_copy.copy()
            face_imgs["eyes_outlined"] = cropped_face.copy()

            # Create mask for left and right eyes
            left_eye_mask = np.zeros_like(image_copy)
            right_eye_mask = np.zeros_like(image_copy)

            cv2.fillPoly(left_eye_mask, [left_eye_points], (255, 255, 255))
            cv2.fillPoly(right_eye_mask, [right_eye_points], (255, 255, 255))

            # Segment eyes
            left_eye_segment = cv2.bitwise_and(image, left_eye_mask)
            right_eye_segment = cv2.bitwise_and(image, right_eye_mask)

            # Crop eyes to bounding box
            x, y, w, h = cv2.boundingRect(left_eye_points)
            left_eye_cropped = left_eye_segment[y : y + h, x : x + w]
            left_eye_mask_cropped = left_eye_mask[y : y + h, x : x + w]

            x, y, w, h = cv2.boundingRect(right_eye_points)
            right_eye_cropped = right_eye_segment[y : y + h, x : x + w]
            right_eye_mask_cropped = right_eye_mask[y : y + h, x : x + w]

            segmented_images["left_eye"]["segmented_img"] = left_eye_cropped
            segmented_images["left_eye"]["segmented_mask"] = left_eye_mask_cropped
            segmented_images["right_eye"]["segmented_img"] = right_eye_cropped
            segmented_images["right_eye"]["segmented_mask"] = right_eye_mask_cropped

        if on_iris:
            # Left and Right Iris Outline
            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[self.LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[self.RIGHT_IRIS])

            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            cv2.circle(
                image_copy,
                center_left,
                int(l_radius),
                outline_color,
                outline_thickness,
                line_type,
            )
            cv2.circle(
                image_copy,
                center_right,
                int(r_radius),
                outline_color,
                outline_thickness,
                line_type,
            )
            cropped_face, face_depth = self.extract_face_regions(image_copy, bboxC, image_depth)
            if on_eyes:
                full_imgs["eyes_n_iris_outlined"] = image_copy.copy()
                face_imgs["eyes_n_iris_outlined"] = cropped_face.copy()
            else:
                full_imgs["iris_outlined"] = image_copy.copy()
                face_imgs["iris_outlined"] = cropped_face.copy()

            # Create mask for left and right irises
            left_iris_mask = np.zeros_like(image_copy)
            right_iris_mask = np.zeros_like(image_copy)

            cv2.circle(left_iris_mask, center_left, int(l_radius), (255, 255, 255), -1)
            cv2.circle(right_iris_mask, center_right, int(r_radius), (255, 255, 255), -1)

            # Segment irises
            left_iris_segment = cv2.bitwise_and(image, left_iris_mask)
            right_iris_segment = cv2.bitwise_and(image, right_iris_mask)

            # Crop irises to bounding box
            x, y, w, h = cv2.boundingRect(np.array([center_left - int(l_radius), center_left + int(l_radius)]))
            left_iris_cropped = left_iris_segment[y : y + h, x : x + w]
            left_iris_mask_cropped = left_iris_mask[y : y + h, x : x + w]

            x, y, w, h = cv2.boundingRect(np.array([center_right - int(r_radius), center_right + int(r_radius)]))
            right_iris_cropped = right_iris_segment[y : y + h, x : x + w]
            right_iris_mask_cropped = right_iris_mask[y : y + h, x : x + w]

            segmented_images["left_iris"]["segmented_img"] = left_iris_cropped
            segmented_images["left_iris"]["segmented_mask"] = left_iris_mask_cropped
            segmented_images["right_iris"]["segmented_img"] = right_iris_cropped
            segmented_images["right_iris"]["segmented_mask"] = right_iris_mask_cropped

        return full_imgs, face_imgs, segmented_images

    def extract_eyes(self, image, blink_detection=False, image_depth=None):

        tmp_image = image.copy()
        results = self.face_detector.process(tmp_image)
        if not results.detections:
            return None
        bboxC = results.detections[0].location_data.relative_bounding_box
        cropped_face, face_depth = self.extract_face_regions(image, bboxC, image_depth)

        tmp_image = image.copy()
        results = self.face_mesh.process(tmp_image)
        if results.multi_face_landmarks is None:
            return None
        face_landmarks = results.multi_face_landmarks[0].landmark

        left_eye, left_eye_depth = self.extract_eyes_regions(image, face_landmarks, self.LEFT_EYE, image_depth)
        right_eye, right_eye_depth = self.extract_eyes_regions(image, face_landmarks, self.RIGHT_EYE, image_depth)
        blinked = False

        mesh_coordinates = self.landmarksDetection(image, results, False)

        if blink_detection:
            # left_EAR, right_EAR = self.blinkRatioWithL3Norm(mesh_coordinates, self.RIGHT_EYE, self.LEFT_EYE)
            left_EAR, right_EAR = self.blinkRatioWithEuclideanDist(mesh_coordinates, self.RIGHT_EYE, self.LEFT_EYE)
            avg_EAR = (left_EAR + right_EAR) / 2
            if avg_EAR > self.blink_lower_thresh and avg_EAR <= self.blink_upper_thresh:
                print(
                    "I think person blinked. avg_EAR = ",
                    avg_EAR,
                    "Confirming with ViT model...",
                )
                blinked = self.blink_detection_model(left_eye=left_eye, right_eye=right_eye)
                if blinked:
                    print("Yes, person blinked. Confirmed by model")
                else:
                    print("No, person didn't blinked. False Alarm")
            elif avg_EAR <= self.blink_lower_thresh:
                blinked = True
                print("Surely person blinked. avg_EAR = ", avg_EAR)
            else:
                blinked = False

        left_eye_outlined = left_eye
        right_eye_outlined = right_eye
        if not blinked:
            full_imgs, face_imgs, segmented_eyes_data = self.create_outline_and_segment(
                mesh_coordinates, image, bboxC, on_eyes=True, on_iris=False, image_depth=image_depth
            )
            full_img_with_eyes_outlined = full_imgs["eyes_outlined"]
            results = self.face_mesh.process(full_img_with_eyes_outlined)
            if results.multi_face_landmarks is None:
                left_eye_outlined = left_eye
                right_eye_outlined = right_eye
            else:
                face_landmarks = results.multi_face_landmarks[0].landmark
                left_eye_outlined, _ = self.extract_eyes_regions(
                    full_img_with_eyes_outlined, face_landmarks, self.LEFT_EYE
                )
                right_eye_outlined, _ = self.extract_eyes_regions(
                    full_img_with_eyes_outlined, face_landmarks, self.RIGHT_EYE
                )

        left_iris_in_eye_outlined = left_eye
        right_iris_in_eye_outlined = right_eye
        if not blinked:
            full_imgs_iris, face_imgs_iris, segmented_eyes_data_iris = self.create_outline_and_segment(
                mesh_coordinates, image, bboxC, on_eyes=False, on_iris=True, image_depth=image_depth
            )
            full_img_with_iris_in_eyes_outlined = full_imgs_iris["iris_outlined"]
            results = self.face_mesh.process(full_img_with_iris_in_eyes_outlined)
            if results.multi_face_landmarks is None:
                left_iris_in_eye_outlined = left_eye
                right_iris_in_eye_outlined = right_eye
            else:
                face_landmarks = results.multi_face_landmarks[0].landmark
                left_iris_in_eye_outlined, _ = self.extract_eyes_regions(
                    full_img_with_iris_in_eyes_outlined, face_landmarks, self.LEFT_EYE
                )
                right_iris_in_eye_outlined, _ = self.extract_eyes_regions(
                    full_img_with_iris_in_eyes_outlined, face_landmarks, self.RIGHT_EYE
                )

        left_eye_n_iris_outlined = left_eye
        right_eye_n_iris_outlined = right_eye
        left_eye_segmented_data = {"segmented_img": None, "segmented_mask": None}
        right_eye_segmented_data = {"segmented_img": None, "segmented_mask": None}
        if not blinked:
            left_eye_segmented_data = self.segment_img(left_eye)
            right_eye_segmented_data = self.segment_img(right_eye)
            full_imgs, face_imgs, segmented_eye_n_iris_data = self.create_outline_and_segment(
                mesh_coordinates, image, bboxC, on_eyes=True, on_iris=True, image_depth=image_depth
            )
            full_img_with_eyes_n_iris_outlined = full_imgs["eyes_n_iris_outlined"]
            results = self.face_mesh.process(full_img_with_eyes_n_iris_outlined)
            if results.multi_face_landmarks is None:
                left_eye_n_iris_outlined = left_eye
                right_eye_n_iris_outlined = right_eye
            else:
                face_landmarks = results.multi_face_landmarks[0].landmark
                left_eye_n_iris_outlined, _ = self.extract_eyes_regions(
                    full_img_with_eyes_n_iris_outlined, face_landmarks, self.LEFT_EYE
                )
                right_eye_n_iris_outlined, _ = self.extract_eyes_regions(
                    full_img_with_eyes_n_iris_outlined, face_landmarks, self.RIGHT_EYE
                )

        return {
            "full_imgs": {
                "wo_outlines": full_imgs["wo_outlines"] if not blinked else image,
                "eyes_outlined": full_imgs["eyes_outlined"] if not blinked else image,
                "iris_outlined": (full_imgs_iris["iris_outlined"] if not blinked else image),
                "eyes_n_iris_outlined": (full_imgs["eyes_n_iris_outlined"] if not blinked else image),
                "depth": image_depth,
            },
            "face_imgs": {
                "wo_outlines": (face_imgs["wo_outlines"] if not blinked else cropped_face),
                "eyes_outlined": (face_imgs["eyes_outlined"] if not blinked else cropped_face),
                "iris_outlined": (face_imgs_iris["iris_outlined"] if not blinked else cropped_face),
                "eyes_n_iris_outlined": (face_imgs["eyes_n_iris_outlined"] if not blinked else cropped_face),
                "depth": face_depth,
            },
            "left_eye": {
                "wo_outlines": left_eye,
                "eyes_outlined": left_eye_outlined,
                "iris_outlined": left_iris_in_eye_outlined,
                "eyes_n_iris_outlined": left_eye_n_iris_outlined,
                "segmented_polygon": (segmented_eyes_data["left_eye"]["segmented_img"] if not blinked else left_eye),
                "segmented_mask_polygon": (
                    segmented_eyes_data["left_eye"]["segmented_mask"] if not blinked else np.zeros_like(left_eye)
                ),
                "segmented_otsu": (left_eye_segmented_data["segmented_img"] if not blinked else left_eye),
                "segmented_mask_otsu": (
                    left_eye_segmented_data["segmented_mask"] if not blinked else np.zeros_like(left_eye)
                ),
                "EAR": left_EAR,
                "depth": left_eye_depth,
            },
            "right_eye": {
                "wo_outlines": right_eye,
                "eyes_outlined": right_eye_outlined,
                "iris_outlined": right_iris_in_eye_outlined,
                "eyes_n_iris_outlined": right_eye_n_iris_outlined,
                "segmented_polygon": (segmented_eyes_data["right_eye"]["segmented_img"] if not blinked else right_eye),
                "segmented_mask_polygon": (
                    segmented_eyes_data["right_eye"]["segmented_mask"] if not blinked else np.zeros_like(right_eye)
                ),
                "segmented_otsu": (right_eye_segmented_data["segmented_img"] if not blinked else right_eye),
                "segmented_mask_otsu": (
                    right_eye_segmented_data["segmented_mask"] if not blinked else np.zeros_like(right_eye)
                ),
                "EAR": right_EAR,
                "depth": right_eye_depth,
            },
            "avg_EAR": avg_EAR,
            "blinked": blinked,
        }

    @staticmethod
    def segment_img(img):

        # Convert RGB image to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur for denoising
        img_gray_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

        # Perform adaptive thresholding
        _, img_mask = cv2.threshold(img_gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert the mask
        segmented_mask = cv2.bitwise_not(img_mask)
        segmented_mask = cv2.cvtColor(segmented_mask, cv2.COLOR_GRAY2RGB)
        segmented_img = cv2.bitwise_and(img, segmented_mask)

        return {
            "segmented_img": segmented_img,
            "segmented_mask": segmented_mask,
        }

    def extract_iris_regions(self, image, mesh_coordinates, eye_indices, image_depth=None):
        ih, iw, _ = image.shape
        mesh_points = np.array(mesh_coordinates)
        (cx, cy), radius = cv2.minEnclosingCircle(mesh_points[eye_indices])

        # Crop the left iris to be exactly 16*upscaled x 16*upscaled
        x1 = max(int(cx) - (8 * self.upscale), 0)
        y1 = max(int(cy) - (8 * self.upscale), 0)
        x2 = min(int(cx) + (8 * self.upscale), iw)
        y2 = min(int(cy) + (8 * self.upscale), ih)

        iris_region = image[y1:y2, x1:x2]
        iris_depth = image_depth[y1:y2, x1:x2]
        return iris_region, iris_depth

    def extract_iris(self, image, image_depth=None):

        tmp_image = image.copy()
        results = self.face_detector.process(tmp_image)
        if not results.detections:
            return None
        bboxC = results.detections[0].location_data.relative_bounding_box
        cropped_face, face_depth = self.extract_face_regions(image, bboxC, image_depth)

        ih, iw, _ = image.shape
        tmp_image = image.copy()
        results = self.face_mesh.process(tmp_image)
        if results.multi_face_landmarks is None:
            return None
        mesh_coordinates = self.landmarksDetection(image, results, False)

        cropped_left_iris, left_iris_depth = self.extract_iris_regions(
            image, mesh_coordinates, self.LEFT_IRIS, image_depth
        )
        cropped_right_iris, right_iris_depth = self.extract_iris_regions(
            image, mesh_coordinates, self.RIGHT_IRIS, image_depth
        )

        left_iris_segmented_data = self.segment_img(cropped_left_iris)
        right_iris_segmented_data = self.segment_img(cropped_right_iris)

        full_imgs, face_imgs, segmented_iris_data = self.create_outline_and_segment(
            mesh_coordinates, image, bboxC, on_eyes=False, on_iris=True, image_depth=image_depth
        )
        full_img_iris_outlined = full_imgs["iris_outlined"]
        cropped_left_iris_outlined, left_iris_depth = self.extract_iris_regions(
            full_img_iris_outlined, mesh_coordinates, self.LEFT_IRIS, image_depth
        )
        cropped_right_iris_outlined, right_iris_depth = self.extract_iris_regions(
            full_img_iris_outlined, mesh_coordinates, self.RIGHT_IRIS, image_depth
        )

        return {
            "full_imgs": {
                "wo_outlines": full_imgs["wo_outlines"],
                "iris_outlined": full_imgs["iris_outlined"],
                "depth": image_depth,
            },
            "face_imgs": {
                "wo_outlines": face_imgs["wo_outlines"],
                "iris_outlined": face_imgs["iris_outlined"],
                "depth": face_depth,
            },
            "left_iris": {
                "wo_outlines": cropped_left_iris,
                "iris_outlined": cropped_left_iris_outlined,
                "segmented_polygon": (segmented_iris_data["left_iris"]["segmented_img"]),
                "segmented_mask_polygon": (segmented_iris_data["left_iris"]["segmented_mask"]),
                "segmented_otsu": (left_iris_segmented_data["segmented_img"]),
                "segmented_mask_otsu": (left_iris_segmented_data["segmented_mask"]),
                "depth": left_iris_depth,
            },
            "right_iris": {
                "wo_outlines": cropped_right_iris,
                "iris_outlined": cropped_right_iris_outlined,
                "segmented_polygon": (segmented_iris_data["right_iris"]["segmented_img"]),
                "segmented_mask_polygon": (segmented_iris_data["right_iris"]["segmented_mask"]),
                "segmented_otsu": (right_iris_segmented_data["segmented_img"]),
                "segmented_mask_otsu": (right_iris_segmented_data["segmented_mask"]),
                "depth": right_iris_depth,
            },
        }
