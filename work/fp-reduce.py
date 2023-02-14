import json
import os
from math import sqrt
from tqdm import tqdm
import pandas as pd
import multiprocessing
from multiprocessing import Pool


def read_detection_result(test_subset_id):
    current_folder_path = "../output-ss{}/valresults/fp-reduce".format(test_subset_id)

    detection_dfs = {}

    for filename in os.listdir(current_folder_path):
        series_id = filename[0:-4]
        file = os.path.join(current_folder_path, filename)
        try:
            df = pd.read_csv(file, delimiter=" ", header=None)
            df.columns = ["z-index", "top_left_x", "top_left_y", "width", "height", "probability"]
            df = df.sort_values(by="probability", ascending=False)
            detection_dfs.setdefault(series_id, df)
        except Exception:
            empty_df = pd.DataFrame()
            detection_dfs.setdefault(series_id, empty_df)

    return detection_dfs


def restore_coord(x, y, z, w, h, spacings, origins, is_flip):
    image_size = (512, 512)

    [z_spacing, y_spacing, x_spacing] = spacings
    [z_origin, y_origin, x_origin] = origins

    x = x + w / 2
    y = y + h / 2

    if is_flip:
        x = image_size[0] - x
        y = image_size[1] - y

    x = x * x_spacing + x_origin
    y = y * y_spacing + y_origin
    z = z * z_spacing + z_origin
    w = w * x_spacing
    h = h * y_spacing

    return x, y, z, w, h


def combine_result(mhd_info_df, subset, upper_threshold, lower_threshold, result):
    print("reading subset {} detections".format(subset))
    detections = read_detection_result(subset)

    for series_id in tqdm(detections.keys(), desc="subset {} series".format(subset)):
        mhd_info = mhd_info_df[mhd_info_df["series_id"] == series_id].iloc[0]
        origins = mhd_info["origins"]
        spacings = mhd_info["spacings"]
        is_flip = mhd_info["is_flip"]

        spacings = json.loads(spacings)
        origins = json.loads(origins)

        df = detections[series_id]
        df_top = df[df["probability"] >= upper_threshold]
        df_bottom = df[df["probability"] >= lower_threshold]
        df = df.to_records()

        used = {}
        group = {}

        for i in range(0, len(df_top)):
            if used.get(i, 0) == 1:
                continue
            x1 = df[i][2]
            y1 = df[i][3]
            z1 = df[i][1]
            w1 = df[i][4]
            h1 = df[i][5]

            x1, y1, z1, w1, h1 = restore_coord(x1, y1, z1, w1, h1, spacings, origins, is_flip)

            used[i] = 1

            for j in range(i + 1, len(df_bottom)):
                if used.get(j, 0) == 1:
                    continue

                x2 = df[j][2]
                y2 = df[j][3]
                z2 = df[j][1]
                w2 = df[j][4]
                h2 = df[j][5]

                x2, y2, z2, w2, h2 = restore_coord(x2, y2, z2, w2, h2, spacings, origins, is_flip)

                dist = sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1 - z2, 2))

                if dist <= 0.5 * w1:
                    group.setdefault(i, [])
                    used[j] = 1
                    group[i].append(j)

        for k1 in group.keys():
            z = df[k1][1]
            x = df[k1][2]
            y = df[k1][3]
            w = df[k1][4]
            h = df[k1][5]
            p = df[k1][6]

            # for k2 in group[k1]:
            #     x += df[k2][2]
            #     y += df[k2][3]
            #     z += df[k2][1]
            #     w += df[k2][4]
            #     h += df[k2][5]
            #     p = max(p, df[k2][6])
            #
            # x /= len(group[k1]) + 1
            # y /= len(group[k1]) + 1
            # z /= len(group[k1]) + 1
            # w /= len(group[k1]) + 1
            # h /= len(group[k1]) + 1

            x, y, z, w, h = restore_coord(x, y, z, w, h, spacings, origins, is_flip)

            result.append((series_id, x, y, z, p))


def custom_error_callback(error):
    print(error)


if __name__ == '__main__':
    mhd_info_csv_path = "data/mhd_info.csv"
    mhd_info_df = pd.read_csv(mhd_info_csv_path)

    manager = multiprocessing.Manager()
    result = manager.list()

    pool = Pool(multiprocessing.cpu_count())

    upper_threshold, lower_threshold = 0.1, 0

    for subset in range(0, 10):
        pool.apply_async(combine_result, args=(mhd_info_df, subset, upper_threshold, lower_threshold, result),
                         error_callback=custom_error_callback)

    pool.close()
    pool.join()

    result_df = pd.DataFrame(list(result))
    result_df.columns = ["seriesuid", "coordX", "coordY", "coordZ", "probability"]
    result_df.set_index("seriesuid", inplace=True)
    result_df.to_csv("submission.csv")

    print(len(result))
