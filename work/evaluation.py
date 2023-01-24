import os

import pandas as pd
from tqdm import tqdm
import json
import math
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
def read_detection_result():
    # detection_result_folder = "../output/valresults/caltech/h/off"
    detection_result_folder = "../output/valresults/caltech/hw/off"

    detection_dfs = {}
    for epoch_id in tqdm(range(83, 91), desc="test epoch"):
        current_folder_path = "{}/0{}".format(detection_result_folder, epoch_id)

        detection_dfs.setdefault(epoch_id, {})
        for filename in os.listdir(current_folder_path):
            series_id = filename[0:-4]
            file = os.path.join(current_folder_path, filename)
            try:
                result_df = pd.read_csv(file, delimiter=" ", header=None)
                result_df.columns = ["z-index", "top_left_x", "top_left_y", "width", "height", "probability"]
                detection_dfs.get(epoch_id).setdefault(series_id, result_df)
            except Exception:
                # print(epoch_id, filename, "skipped")
                empty_df = pd.DataFrame()
                detection_dfs.get(epoch_id).setdefault(series_id, empty_df)

    return detection_dfs


def is_nodule_included(n):
    return n["included"] == 1


def evaluate_froc(epoch_id, image_size, test_detections, sensitivity_x, sensitivity_y, fps, mhd_info_df, annotations, probability_threshold):
    epoch_candidates = test_detections[epoch_id]

    total_number_of_candidates = 0
    total_number_of_nodules = 0

    tp = 0
    fp = 0
    fn = 0

    series_ids = epoch_candidates.keys()

    for series_id in tqdm(series_ids, desc="{}".format(epoch_id)):
        mhd_info = mhd_info_df[mhd_info_df["series_id"] == series_id].iloc[0]
        origins = mhd_info["origins"]
        spacings = mhd_info["spacings"]
        is_flip = mhd_info["is_flip"]

        spacings = json.loads(spacings)
        origins = json.loads(origins)

        [z_spacing, y_spacing, x_spacing] = spacings
        [z_origin, y_origin, x_origin] = origins

        candidates = epoch_candidates[series_id].reset_index()
        top_z_index = set(candidates[candidates["probability"] > probability_threshold]["z-index"].values)
        candidates = candidates[candidates["z-index"].isin(top_z_index)].sort_values(by=['probability'],
                                                                                     ascending=False)
        candidates2 = candidates.copy()
        total_number_of_candidates = len(candidates)

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
                x2 = candidate["top_left_x"]
                y2 = candidate["top_left_y"]
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
        print("{} sensitivity".format(epoch_id), 0)
        sensitivity_y.append(0)
    else:
        print("{} sensitivity".format(epoch_id), tp / total_number_of_nodules)
        sensitivity_y.append(tp / total_number_of_nodules)

    fps.append(fp)


def custom_error_callback(error):
    print(error)


if __name__ == '__main__':
    image_size = (512, 512)
    mhd_info_csv_path = "data/mhd_info.csv"
    mhd_info_df = pd.read_csv(mhd_info_csv_path)
    annotations_included_path = "data/annotations.csv"
    annotations_included_df = pd.read_csv(annotations_included_path, delimiter=",")
    annotations_included_df["included"] = 1
    annotations_excluded_path = "data/annotations_excluded.csv"
    annotations_excluded_df = pd.read_csv(annotations_excluded_path, delimiter=",")
    annotations_excluded_df["included"] = 0
    annotations = pd.concat([annotations_included_df, annotations_excluded_df])

    test_detections = read_detection_result()

    sensitivity_x = []
    sensitivity_y = []

    fps = []

    probability_threshold = 0.05

    pool = Pool(4)

    for pp in range(83, 91):
        pool.apply_async(evaluate_froc, args=(pp, image_size, test_detections, sensitivity_x, sensitivity_y, fps, mhd_info_df, annotations, probability_threshold), error_callback=custom_error_callback)

    pool.close()
    pool.join()

    figure(figsize=(25, 5))
    plt.plot(sensitivity_x, sensitivity_y, label='Sensitivity')
    figure(figsize=(25, 5))
    plt.plot(sensitivity_x, fps, label='fp')

    qq = sorted(list(zip(sensitivity_x, sensitivity_y, fps)), key=lambda t: t[1] * 10000 - math.log(t[2]))
    print(qq)
