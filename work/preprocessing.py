import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

download_data_folder = "../downloads/data"

train_image_path = "../data/caltech/train_3/images"
test_image_path = "../data/caltech/test/images"

annotations_path = "../downloads/annotations.csv"
annotations_df = pd.read_csv(annotations_path, delimiter=",")

output_folder_path = "../output/"
cache_folder_path = "../data/cache/"

gt_cache_path = "../data/cache/caltech/train_gt"
nogt_cache_path = "../data/cache/caltech/train_nogt"
test_cache_path = "../data/cache/caltech/test"

cache_results_folder_path = "../output/cache_results/"
gt_cache_csv_path = "../output/cache_results/gt.csv"
nogt_cache_csv_path = "../output/cache_results/nogt.csv"
test_cache_csv_path = "../output/cache_results/test.csv"

image_width = 512
image_height = 512

test_subset_id = 9


def create_output_folder():
    for p in [train_image_path,
              test_image_path,
              cache_folder_path,
              output_folder_path,
              cache_results_folder_path]:
        if not os.path.exists(p):
            os.makedirs(p)


def configure_matplot():
    px = 1 / plt.rcParams['figure.dpi']
    plt.figure(figsize=(image_width * px, image_height * px))
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)


def world_to_voxel_coord(world_coord, origin, spacing):
    stretched_voxel_coord = np.absolute(world_coord - origin)
    voxel_coord = stretched_voxel_coord / spacing
    return voxel_coord


def extract_annotations(series_id, origins, spacings, is_flip):
    annotations = annotations_df[annotations_df["seriesuid"] == series_id]
    if annotations.size == 0:
        return pd.DataFrame([], columns=["z", "bboxes", "filepath", "ignoreareas"])

    nodules_annotation = []

    for index, row in annotations.iterrows():
        world_x = row["coordX"]
        world_y = row["coordY"]
        world_z = row["coordZ"]
        d = row["diameter_mm"]

        world_coord = np.array([float(world_z), float(world_y), float(world_x)])
        voxel_coord = world_to_voxel_coord(world_coord, origins, spacings).astype(int)

        voxel_z, voxel_y, voxel_x = voxel_coord
        if is_flip:
            voxel_x = 512 - voxel_x
            voxel_y = 512 - voxel_y

        diameter = round(d / spacings[2])

        annotation = {"z": voxel_z, 'bboxes': np.array([
            np.array([
                max(voxel_x, 0),
                max(voxel_y, 0),
                min(voxel_x + diameter, image_width - 1),
                min(voxel_y + diameter, image_height - 1)
            ])
        ]), 'ignoreareas': np.array([])}
        nodules_annotation.append(annotation)

    nodules_df = pd.DataFrame(nodules_annotation)

    return nodules_df


def process_image(series_id, subset_id, generate_picture=True):
    print("subset{}".format(subset_id), series_id)

    mhd_file_path = "{}/subset{}/{}.mhd".format(download_data_folder, subset_id, series_id)

    itk_image = sitk.ReadImage(mhd_file_path, sitk.sitkFloat32)
    ct_scans = sitk.GetArrayFromImage(itk_image)

    origins = np.array(list(reversed(itk_image.GetOrigin())))
    spacings = np.array(list(reversed(itk_image.GetSpacing())))

    direction = itk_image.GetDirection()
    direction = np.array(list(map(lambda x: round(x), direction)))

    if np.any(direction != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
        is_flip = True
    else:
        is_flip = False

    if is_flip:
        ct_scans = ct_scans[:, ::-1, ::-1]

    output_path = train_image_path if subset_id != test_subset_id else test_image_path

    nodules_df = extract_annotations(series_id, origins, spacings, is_flip)

    for i in tqdm(range(0, ct_scans.shape[0])):
        image_file = "{}/{}_{}.jpg".format(output_path, series_id, i)

        if generate_picture:
            plt.gray()
            plt.margins(0, 0)
            plt.axis('off')
            plt.imshow(ct_scans[i])
            plt.savefig(image_file, bbox_inches='tight', pad_inches=0)
            plt.clf()

        nodule_row = nodules_df[nodules_df["z"] == i]

        if subset_id != test_subset_id:
            if nodule_row.size == 0:
                anno = {'bboxes': [], 'ignoreareas': [],
                        'filepath': os.path.join(image_file)}
                image_data_nogt.append(anno)
            else:
                nodule = nodule_row.iloc[0]
                anno = {'bboxes': nodule["bboxes"], 'ignoreareas': nodule["ignoreareas"],
                        'filepath': os.path.join(image_file)}
                image_data_gt.append(anno)
        else:
            if nodule_row.size != 0:
                nodule = nodule_row.iloc[0]
                anno = {'bboxes': nodule["bboxes"], 'ignoreareas': nodule["ignoreareas"],
                        'filepath': os.path.join(image_file)}
                image_data_test.append(anno)

    print()


if __name__ == '__main__':
    create_output_folder()
    configure_matplot()

    image_data_gt, image_data_nogt, image_data_test = [], [], []

    for subset_id in range(0, 10):
        current_folder = os.listdir("{}/subset{}".format(download_data_folder, subset_id))

        for file in current_folder:
            if file.endswith(".mhd"):
                series_id = re.search(r'(.*)\.mhd', file).group(1)
                process_image(series_id, subset_id, generate_picture=False)

    image_data_gt_df = pd.DataFrame(image_data_gt)
    image_data_nogt_df = pd.DataFrame(image_data_nogt)
    image_data_test_df = pd.DataFrame(image_data_test)
    image_data_gt_df.set_index("filepath", inplace=True)
    image_data_nogt_df.set_index("filepath", inplace=True)
    image_data_test_df.set_index("filepath", inplace=True)
    image_data_gt_df.to_csv(gt_cache_csv_path)
    image_data_nogt_df.to_csv(nogt_cache_csv_path)
    image_data_test_df.to_csv(test_cache_csv_path)

    with open(gt_cache_path, 'wb') as fid:
        pickle.dump(image_data_gt, fid, 2)
    with open(nogt_cache_path, 'wb') as fid:
        pickle.dump(image_data_nogt, fid, 2)
    with open(test_cache_path, 'wb') as fid:
        pickle.dump(image_data_test, fid, 2)
