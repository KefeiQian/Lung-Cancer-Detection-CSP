import multiprocessing
import os
import pickle
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def world_to_voxel_coord(world_coord, origin, spacing):
    stretched_voxel_coord = np.absolute(world_coord - origin)
    voxel_coord = stretched_voxel_coord / spacing
    return voxel_coord


def read_ground_truth_data(test_subset_id):
    fid = open("../data/cache/test-ss{}".format(test_subset_id), 'rb')
    test_pickle = pickle.Unpickler(fid, encoding="latin1")
    test_cache = test_pickle.load()

    gt_df = pd.DataFrame(test_cache)
    gt_df["series_id"] = gt_df["filepath"].map(lambda x: x.split("/")[-2])
    gt_df["z-index"] = gt_df["filepath"].map(lambda x: int(x.split("/")[-1][2:-4]))

    return gt_df


def read_detection_result(test_subset_id):
    detection_result_folder = "../output-ss{}/valresults/caltech/hw/off".format(test_subset_id)

    detection_dfs = {}
    for epoch_id in tqdm(os.listdir(detection_result_folder), desc="reading test-{} detections".format(test_subset_id)):
        current_folder_path = "{}/{}".format(detection_result_folder, epoch_id)

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


def compute_overlap_area_ratio(a, b):
    a_top_left_x, a_top_left_y, a_bottom_right_x, a_bottom_right_y = a
    b_top_left_x, b_top_left_y, b_bottom_right_x, b_bottom_right_y = b

    overlap_width = min(a_bottom_right_x, b_bottom_right_x) - max(a_top_left_x, b_top_left_x)
    if overlap_width <= 0:
        return 0

    overlap_height = min(a_bottom_right_y, b_bottom_right_y) - max(a_top_left_y, b_top_left_y)
    if overlap_height <= 0:
        return 0

    a_area = (a_bottom_right_x - a_top_left_x) * (a_bottom_right_y - a_top_left_y)
    b_area = (b_bottom_right_x - b_top_left_x) * (b_bottom_right_y - b_top_left_y)

    overlap_area = overlap_width * overlap_height
    total_area = a_area + b_area - overlap_area

    overlap_ratio = overlap_area / total_area
    return overlap_ratio


def custom_error_callback(error):
    print(error)


def evaluate(test_subset_id, threshold=0.5, probability_threshold=0):
    print("processing ", test_subset_id)
    miss_rate_x = []
    miss_rate_y = []
    test_detections = read_detection_result(test_subset_id)
    annotations = read_ground_truth_data(test_subset_id)

    for epoch_id in tqdm(test_detections.keys(), desc="test-{} epoch".format(test_subset_id)):
        epoch_test_result = test_detections[epoch_id]

        hit_counter = 0

        number_of_test = 0

        for series_id in epoch_test_result.keys():
            image_detections = epoch_test_result[series_id]
            if len(image_detections) == 0:
                continue
            image_detections["z-index"] = image_detections["z-index"].map(lambda x: round(x))
            image_detections["bottom_right_x"] = image_detections["top_left_x"] + image_detections["width"]
            image_detections["bottom_right_y"] = image_detections["top_left_y"] + image_detections["height"]
            image_detections["bbox"] = list(
                image_detections[['top_left_x', 'top_left_y', "bottom_right_x", "bottom_right_y"]].to_records(
                    index=False))

            nodules = annotations[annotations["series_id"] == series_id]

            for i, nodule in nodules.iterrows():
                z_index = nodule["z-index"]

                detections = image_detections[image_detections['z-index'] == z_index].reset_index().query(
                    "probability >= {}".format(probability_threshold))
                matched_detection = set()

                nodule_id = 0
                for bbox in nodule["bboxes"]:
                    max_ratio = threshold
                    best_detection_id = -1
                    number_of_test += 1

                    for detection_id, detection in detections.iterrows():
                        if detection_id in matched_detection:
                            continue

                        ratio = compute_overlap_area_ratio(bbox, detection["bbox"])

                        if ratio > max_ratio:
                            best_detection_id = detection_id
                            break

                    if best_detection_id != -1:
                        hit_counter += 1
                        matched_detection.add(best_detection_id)

                nodule_id += 1

        hit_rate = hit_counter / number_of_test
        miss_rate_y.append(1 - hit_rate)
        miss_rate_x.append(int(epoch_id))

    figure(figsize=(25, 5))
    plt.clf()
    plt.plot(miss_rate_x, miss_rate_y, label='MR')
    plt.savefig("../output-ss{}/mr50.png".format(test_subset_id))
    plt.show()

    result_list = sorted(list(zip(miss_rate_x, miss_rate_y)), key=lambda t: t[1], reverse=False)
    result_list_df = pd.DataFrame(result_list)
    result_list_df.to_csv("../output-ss{}/mr50.csv".format(test_subset_id))


if __name__ == '__main__':
    pool = Pool(multiprocessing.cpu_count())

    for test_subset_id in range(0, 10):
        pool.apply_async(evaluate, args=(test_subset_id,), error_callback=custom_error_callback)

    pool.close()
    pool.join()
