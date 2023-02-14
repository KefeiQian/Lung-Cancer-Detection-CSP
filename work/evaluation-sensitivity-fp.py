import multiprocessing
import os

import pandas as pd
from tqdm import tqdm
import json
import math
from multiprocessing import Pool

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def read_detection_result(test_subset_id):
    detection_result_folder = "../output-ss{}/valresults/caltech/hw/off".format(test_subset_id)

    detection_dfs = {}
    for epoch_id in tqdm(os.listdir(detection_result_folder), desc="reading test-{} detections".format(test_subset_id)):
        if int(epoch_id) < 80: continue
        current_folder_path = "{}/{}".format(detection_result_folder, epoch_id)

        detection_dfs.setdefault(epoch_id, {})
        for filename in os.listdir(current_folder_path):
            series_id = filename[0:-4]
            file = os.path.join(current_folder_path, filename)
            try:
                result_df = pd.read_csv(file, delimiter=" ", header=None)
                result_df.columns = ["z-index", "top_left_x",
                                     "top_left_y", "width", "height", "probability"]
                detection_dfs.get(epoch_id).setdefault(series_id, result_df)
            except Exception:
                # print(epoch_id, filename, "skipped")
                empty_df = pd.DataFrame()
                detection_dfs.get(epoch_id).setdefault(series_id, empty_df)

    return detection_dfs


def is_nodule_included(n):
    return n["included"] == 1


def evaluate_froc(test_subset_id, mhd_info_df, annotations, probability_threshold=0.05):
    print("processing ", test_subset_id)

    test_detections = read_detection_result(test_subset_id)

    image_size = (512, 512)

    sensitivity_x = []
    sensitivity_y = []

    fps = []

    for epoch_id in tqdm(test_detections.keys(), desc="test-{} epoch".format(test_subset_id)):
        epoch_candidates = test_detections[epoch_id]

        total_number_of_nodules = 0

        tp = 0
        fp = 0
        fn = 0

        for series_id in epoch_candidates.keys():
            mhd_info = mhd_info_df[mhd_info_df["series_id"] == series_id].iloc[0]
            origins = mhd_info["origins"]
            spacings = mhd_info["spacings"]
            is_flip = mhd_info["is_flip"]

            spacings = json.loads(spacings)
            origins = json.loads(origins)

            [z_spacing, y_spacing, x_spacing] = spacings
            [z_origin, y_origin, x_origin] = origins

            candidates = epoch_candidates[series_id].reset_index()
            candidates = candidates[candidates["probability"] > probability_threshold]
            candidates2 = candidates.copy()

            nodules = annotations[annotations["seriesuid"] == series_id]

            for _, nodule in nodules.iterrows():
                if is_nodule_included(nodule):
                    total_number_of_nodules += 1

                x = nodule["coordX"]
                y = nodule["coordY"]
                z = nodule["coordZ"]
                diameter = nodule["diameter_mm"]

                if diameter <= 0:
                    diameter = 10

                radius_squared = pow((diameter / 2), 2)

                matched_candidates = []

                for candidate_id, candidate in candidates.iterrows():
                    x2 = candidate["top_left_x"] + candidate["width"] / 2
                    y2 = candidate["top_left_y"] + candidate["height"] / 2
                    z2 = candidate["z-index"]

                    if is_flip:
                        x2 = image_size[0] - x2
                        y2 = image_size[1] - y2

                    x2 = x2 * x_spacing + x_origin
                    y2 = y2 * y_spacing + y_origin
                    z2 = z2 * z_spacing + z_origin

                    dist = pow(x - x2, 2) + pow(y - y2, 2) + pow(z - z2, 2)
                    if dist < radius_squared:
                        if is_nodule_included(nodule):
                            matched_candidates.append(candidate)
                        break

                if is_nodule_included(nodule):
                    if len(matched_candidates) > 0:
                        tp += 1
                    else:
                        fn += 1

            fp += len(candidates2)

        sensitivity_x.append(int(epoch_id))
        if total_number_of_nodules == 0:
            sensitivity_y.append(0)
        else:
            sensitivity_y.append(tp / total_number_of_nodules)

        fps.append(fp)

    plt.clf()
    fig, ax1 = plt.subplots(figsize=(25, 5))
    plt.xticks(rotation=45)
    plt.plot(sensitivity_x, sensitivity_y, label='Sensitivity')
    ax1.plot(sensitivity_x, sensitivity_y, color="blue", alpha=0.5, label="sensitivity")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("sensitivity")
    ax2 = ax1.twinx()
    ax2.plot(sensitivity_x, fps, color="red", label="fp")
    ax2.set_ylabel("fp")
    plt.savefig("../output-ss{}/sensitivity-fp-{}.jpg".format(test_subset_id, probability_threshold))
    plt.show()

    result_list = sorted(list(zip(sensitivity_x, sensitivity_y, fps)),
                         key=lambda t: t[1] * 10000)
    result_list_df = pd.DataFrame(result_list)
    result_list_df.to_csv("../output-ss{}/sensitivity-fp-{}.csv".format(test_subset_id, probability_threshold))


def custom_error_callback(error):
    print(error)


if __name__ == '__main__':
    mhd_info_csv_path = "data/mhd_info.csv"
    mhd_info_df = pd.read_csv(mhd_info_csv_path)
    annotations_included_path = "data/annotations.csv"
    annotations_included_df = pd.read_csv(
        annotations_included_path, delimiter=",")
    annotations_included_df["included"] = 1
    annotations_excluded_path = "data/annotations_excluded.csv"
    annotations_excluded_df = pd.read_csv(
        annotations_excluded_path, delimiter=",")
    annotations_excluded_df["included"] = 0
    annotations = pd.concat([annotations_included_df, annotations_excluded_df])

    probability_threshold = 0.02

    pool = Pool(multiprocessing.cpu_count())

    for test_subset_id in range(0, 10):
        pool.apply_async(evaluate_froc, args=(test_subset_id, mhd_info_df, annotations, probability_threshold),
                         error_callback=custom_error_callback)

    pool.close()
    pool.join()
