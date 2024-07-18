import os
import sys
import cv2
import warnings
import os.path as osp

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
sys.path.append(root_path)

from feature_extraction.extractor_mediapipe import ExtractorMediaPipe
from feature_extraction.extractor_facexlib import ExtractorFacexlib

# from feature_extraction.extractor_dlib import ExtractorDlib

warnings.filterwarnings("ignore")


class FeaturesExtractor:

    def __init__(self, extraction_library="mediapipe", blink_detection=True, upscale=1):
        self.blink_detection = blink_detection
        self.extraction_library = extraction_library
        self.upscale = int(upscale)

        if self.extraction_library == "facexlib":
            self.feature_extractor = ExtractorFacexlib(self.upscale)
        elif self.extraction_library == "mediapipe":
            self.feature_extractor = ExtractorMediaPipe(self.upscale)
        # if self.extraction_library == "dlib":
        #     self.feature_extractor = ExtractorDlib(self.upscale)
        else:
            raise Exception(
                f"No such library named: '{extraction_library}' implemented for feature extraction. Must be one of: ['dlib', 'mediapipe', 'facexlib']"
            )

    def __call__(self, image):
        results = {}
        face = self.feature_extractor.extract_face(image)
        if face is None:
            print("No face found. Skipped feature extraction!")
            return None
        else:
            results["img"] = image
            results["face"] = face
            eyes_data = self.feature_extractor.extract_eyes(image, self.blink_detection)
            if eyes_data is None:
                print("No eyes found. Skipped feature extraction!")
                return results
            else:
                results["eyes"] = eyes_data
                if eyes_data["blinked"]:
                    print("Found blinked eyes!")
                    return results
                else:
                    iris_data = self.feature_extractor.extract_iris(image)
                    if iris_data is None:
                        print("No iris found. Skipped feature extraction!")
                        return results
                    else:
                        results["iris"] = iris_data
                        return results


if __name__ == "__main__":

    image = cv2.imread(f"{ROOT_DIR}/data/original/2/2/frame_12.png")
    upscale = 1
    blink_detection = True

    for extraction_library in ["facexlib"]:

        print(f"=================== {extraction_library} ===================")

        features_extractor = FeaturesExtractor(
            extraction_library=extraction_library,
            blink_detection=blink_detection,
            upscale=upscale,
        )
        result_dict = features_extractor(image)

        if result_dict is not None and len(result_dict.keys()) > 0:

            output_folder = (
                f"{ROOT_DIR}/feature_extraction/detections/{extraction_library}"
            )
            os.makedirs(output_folder, exist_ok=True)

            print("result_dict = ", result_dict.keys())
            cv2.imwrite(f"{output_folder}/face.png", result_dict["face"])

            print("result_dict eyes = ", result_dict["eyes"].keys())
            cv2.imwrite(
                f"{output_folder}/left_eye.png", result_dict["eyes"]["left_eye"]
            )
            cv2.imwrite(
                f"{output_folder}/right_eye.png", result_dict["eyes"]["right_eye"]
            )

            print("result_dict iris = ", result_dict["iris"].keys())
            print("result_dict left iris = ", result_dict["iris"]["left_iris"].keys())
            print("result_dict right iris = ", result_dict["iris"]["right_iris"].keys())
            cv2.imwrite(
                f"{output_folder}/left_iris.png",
                result_dict["iris"]["left_iris"]["img"],
            )
            cv2.imwrite(
                f"{output_folder}/right_iris.png",
                result_dict["iris"]["right_iris"]["img"],
            )
            cv2.imwrite(
                f"{output_folder}/left_iris_segmented.png",
                result_dict["iris"]["left_iris"]["segmented_iris"],
            )
            cv2.imwrite(
                f"{output_folder}/right_iris_segmented.png",
                result_dict["iris"]["right_iris"]["segmented_iris"],
            )
            cv2.imwrite(
                f"{output_folder}/left_iris_segmented_mask.png",
                result_dict["iris"]["left_iris"]["segmented_mask"],
            )
            cv2.imwrite(
                f"{output_folder}/right_iris_segmented_mask.png",
                result_dict["iris"]["right_iris"]["segmented_mask"],
            )
