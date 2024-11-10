import sys
import os.path as osp

root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
sys.path.append(root_path)

from data_creation.features_extraction.extractor_mediapipe import ExtractorMediaPipe


class FeaturesExtractor:

    def __init__(self, extraction_library="facexlib", blink_detection=False, upscale=1):
        self.blink_detection = blink_detection
        self.extraction_library = extraction_library
        self.upscale = int(upscale)
        if self.extraction_library == "mediapipe":
            self.feature_extractor = ExtractorMediaPipe(self.upscale)
        else:
            raise Exception(
                f"No such library named: '{extraction_library}' implemented for feature extraction. Must be one of: ['dlib', 'mediapipe', 'facexlib']"
            )

    def __call__(self, image):
        results = {}
        image_depth = self.feature_extractor.get_depth_map(image)
        face, face_depth = self.feature_extractor.extract_face(image, image_depth)
        if face is None:
            print("No face found. Skipped feature extraction!")
            return None
        else:
            results["img"] = image
            results["face"] = face
            results["img_depth"] = image_depth
            results["face_depth"] = face_depth
            eyes_data = self.feature_extractor.extract_eyes(image, self.blink_detection, image_depth)
            if eyes_data is None:
                print("No eyes found. Skipped feature extraction!")
                return results
            else:
                results["eyes"] = eyes_data
                if eyes_data["blinked"]:
                    print("Found blinked eyes!")
                    return results
                else:
                    iris_data = self.feature_extractor.extract_iris(image, image_depth)
                    if iris_data is None:
                        print("No iris found. Skipped feature extraction!")
                        return results
                    else:
                        results["iris"] = iris_data
                        return results
