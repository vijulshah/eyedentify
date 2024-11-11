import os
import sys
import cv2
import os.path as osp

import numpy as np

root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
sys.path.append(root_path)

from data_creation.features_extraction.features_extractor import FeaturesExtractor

if __name__ == "__main__":

    img_path = f"./eyedentify/local/data/EyeDentify/Wo_SR/original/1/1/frame_20.png"
    print("img_path ===> ", img_path)
    image = cv2.imread(img_path)
    upscale = 1
    blink_detection = True
    frame_name = img_path.split("/")[-1]

    extraction_libraries = ["mediapipe"]

    save_features = ["full_imgs", "faces", "eyes", "blinks", "iris"]
    eye_types = ["left_eyes", "right_eyes"]
    iris_types = ["left_iris", "right_iris"]

    features_subdirs = [
        "wo_outlines",
        "eyes_outlined",
        "iris_outlined",
        "eyes_n_iris_outlined",
        "depth",
    ]
    segmentation_polygon_subdirs = [
        "segmented_polygon",
        "segmented_mask_polygon",
    ]
    segmentation_otsu_subdirs = [
        "segmented_otsu",
        "segmented_mask_otsu",
    ]

    for extraction_library in extraction_libraries:

        print(f"=================== {extraction_library} ===================")

        features_extractor = FeaturesExtractor(
            extraction_library=extraction_library,
            blink_detection=blink_detection,
            upscale=upscale,
        )
        result_dict = features_extractor(image)
        print("result_dict = ", result_dict.keys())

        if result_dict is not None and len(result_dict.keys()) > 0:

            blinked = result_dict["eyes"]["blinked"]
            print("blinked = ", blinked)
            print("left EAR = ", result_dict["eyes"]["left_eye"]["EAR"])
            print("right EAR = ", result_dict["eyes"]["right_eye"]["EAR"])
            print("avg_EAR = ", result_dict["eyes"]["avg_EAR"])

            output_folder = f"./eyedentify/data_creation/features_extraction/detections/{extraction_library}"
            os.makedirs(output_folder, exist_ok=True)

            results_list = list(result_dict.keys())

            if blinked:
                print("Results contain blinked eyes. They will not be saved.")
                break

            for feature in save_features:

                if feature == "full_imgs" and "full_imgs" in result_dict["eyes"]:
                    for subdir in features_subdirs:
                        eyes_data = result_dict["eyes"]["full_imgs"][subdir]
                        if eyes_data.ndim == 3:
                            cv2.imwrite(
                                os.path.join(output_folder, f"{feature}_{subdir}.png"),
                                eyes_data,
                            )
                        else:
                            np.save(os.path.join(output_folder, f"{feature}_{subdir}"), eyes_data)
                        if subdir == "eyes_outlined" or subdir == "eyes_n_iris_outlined":
                            continue
                        iris_data = result_dict["iris"]["full_imgs"][subdir]
                        if iris_data.ndim == 3:
                            cv2.imwrite(
                                os.path.join(output_folder, f"{feature}_{subdir}.png"),
                                iris_data,
                            )
                        else:
                            np.save(os.path.join(output_folder, f"{feature}_{subdir}"), iris_data)
                if feature == "faces" and "face_imgs" in result_dict["eyes"]:
                    for subdir in features_subdirs:
                        eyes_data = result_dict["eyes"]["face_imgs"][subdir]
                        if eyes_data.ndim == 3:
                            cv2.imwrite(
                                os.path.join(output_folder, f"{feature}_{subdir}.png"),
                                eyes_data,
                            )
                        else:
                            np.save(os.path.join(output_folder, f"{feature}_{subdir}"), eyes_data)
                        if subdir == "eyes_outlined" or subdir == "eyes_n_iris_outlined":
                            continue
                        iris_data = result_dict["iris"]["face_imgs"][subdir]
                        if iris_data.ndim == 3:
                            cv2.imwrite(
                                os.path.join(output_folder, f"{feature}_{subdir}.png"),
                                iris_data,
                            )
                        else:
                            np.save(os.path.join(output_folder, f"{feature}_{subdir}"), iris_data)

                if feature == "eyes" and "eyes" in results_list:
                    for eye_type in eye_types:
                        for subdir in features_subdirs + segmentation_polygon_subdirs + segmentation_otsu_subdirs:
                            eyes_data = result_dict["eyes"][eye_type.replace("s", "")][subdir]
                            if eyes_data.ndim == 3:
                                cv2.imwrite(
                                    os.path.join(output_folder, f"{eye_type}_{subdir}.png"),
                                    eyes_data,
                                )
                            else:
                                np.save(os.path.join(output_folder, f"{eye_type}_{subdir}"), eyes_data)

                elif feature == "iris" and "iris" in results_list:
                    for iris_type in iris_types:
                        for subdir in (
                            [features_subdirs[0], features_subdirs[2], features_subdirs[-1]]
                            + segmentation_polygon_subdirs
                            + segmentation_otsu_subdirs
                        ):
                            iris_data = result_dict["iris"][iris_type][subdir]
                            if iris_data.ndim == 3:
                                cv2.imwrite(
                                    os.path.join(output_folder, f"{iris_type}_{subdir}.png"),
                                    iris_data,
                                )
                            else:
                                np.save(os.path.join(output_folder, f"{iris_type}_{subdir}"), iris_data)
