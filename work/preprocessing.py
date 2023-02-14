import os
import pickle
import re
import SimpleITK as sitk
import cv2
import numpy as np
import pandas as pd
import json
import sys
from tqdm import tqdm

test_subset_id = int(sys.argv[1])

download_folder = "../downloads"
download_data_folder = "../downloads/data"
download_mask_folder = "../downloads/seg-lungs-LUNA16"

output_nodules_folder = "output/nodules"

data_image_folder = "../data/images"

mhd_info_csv_path = "data/mhd_info.csv"

for p in [output_nodules_folder, data_image_folder]:
    if not os.path.exists(p):
        os.makedirs(p)

image_size = (512, 512)

annotations_path = "../downloads/annotations.csv"
annotations_df = pd.read_csv(annotations_path, delimiter=",")
annotations_df.head()

candidates_path = "../downloads/candidates_V2.csv"
candidates_df = pd.read_csv(candidates_path, delimiter=",")
gt_candidates_df = candidates_df[candidates_df["class"] == 1]
no_gt_candidates_df = candidates_df[candidates_df["class"] != 1]

gt_candidates_df.head()


def normalize(image):
    image[image > 400] = 400
    image[image < -1000] = -1000

    max_hu = image.max()
    min_hu = image.min()
    image = (image - min_hu) / (max_hu - min_hu) * 255
    image = np.round(image)
    return image


def world_to_voxel_coord(world_coord, origin, spacing):
    stretched_voxel_coord = np.absolute(world_coord - origin)
    voxel_coord = stretched_voxel_coord / spacing
    return voxel_coord


def extract_mhd_file_info(subset_id, series_id):
    mhd_file_path = "{}/subset{}/{}.mhd".format(download_data_folder, subset_id, series_id)
    mask_file_path = "{}/{}.mhd".format(download_mask_folder, series_id)

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

    ct_image_folder = "{}/{}".format(data_image_folder, series_id)

    mhd_info = {
        "mhd_file_path": mhd_file_path,
        "mask_file_path": mask_file_path,
        "subset_id": subset_id,
        "series_id": series_id,
        "total_images": ct_scans.shape[0],
        "origins": origins.tolist(),
        "spacings": spacings.tolist(),
        "is_flip": is_flip,
        "ct_image_folder": ct_image_folder
    }

    return mhd_info


def extract_ct_images(mhd_info, mask=False):
    mhd_file_path = mhd_info["mhd_file_path"]
    mask_file_path = mhd_info["mask_file_path"]
    is_flip = mhd_info["is_flip"]
    total_images = mhd_info["total_images"]
    ct_image_folder = mhd_info["ct_image_folder"]

    ct_scans = sitk.GetArrayFromImage(sitk.ReadImage(mhd_file_path, sitk.sitkFloat32))
    if mask:
        masks = sitk.GetArrayFromImage(sitk.ReadImage(mask_file_path, sitk.sitkFloat32))

    if is_flip:
        ct_scans = ct_scans[:, ::-1, ::-1]
        if mask:
            masks = masks[:, ::-1, ::-1]

    ct_scans = normalize(ct_scans)
    ct_scans = ct_scans.astype(np.uint8)
    if mask:
        masks = masks.astype(np.uint8)

    if not os.path.exists(ct_image_folder):
        os.makedirs(ct_image_folder)

    for z in range(0, total_images):
        if mask:
            masked_image = cv2.bitwise_and(ct_scans[z], ct_scans[z], mask=masks[z])
            colored_image = cv2.cvtColor(masked_image, cv2.COLOR_GRAY2RGB)
        else:
            colored_image = cv2.cvtColor(ct_scans[z], cv2.COLOR_GRAY2RGB)
        cv2.imwrite("{}/z-{}.jpg".format(ct_image_folder, z), colored_image)


def generate_train_test_cache(mhd_info):
    subset_id = int(mhd_info["subset_id"])
    series_id = mhd_info["series_id"]
    origins = mhd_info["origins"]
    spacings = mhd_info["spacings"]
    is_flip = mhd_info["is_flip"]
    total_images = mhd_info["total_images"]
    ct_image_folder = mhd_info["ct_image_folder"]

    spacings = np.array(json.loads(spacings))
    origins = np.array(json.loads(origins))

    annotations = annotations_df[annotations_df["seriesuid"] == series_id]

    nodules_annotation = []
    for _, row in annotations.iterrows():
        world_x = row["coordX"]
        world_y = row["coordY"]
        world_z = row["coordZ"]
        diameter = row["diameter_mm"]

        world_coord = np.array([float(world_z), float(world_y), float(world_x)])
        voxel_coord = world_to_voxel_coord(world_coord, origins, spacings)

        voxel_z, voxel_y, voxel_x = voxel_coord
        if is_flip:
            voxel_x = image_size[0] - voxel_x
            voxel_y = image_size[1] - voxel_y

        radius = diameter / spacings[2] / 2

        annotation = {"z": round(voxel_z), 'bbox':
            np.array([
                max(round(voxel_x - radius), 0),
                max(round(voxel_y - radius), 0),
                min(round(voxel_x + radius), image_size[0] - 1),
                min(round(voxel_y + radius), image_size[1] - 1)
            ]), 'ignoreareas': np.array([])}
        nodules_annotation.append(annotation)
    nodules_df = pd.DataFrame(nodules_annotation)

    no_gt_df = no_gt_candidates_df[no_gt_candidates_df["seriesuid"] == series_id]
    no_gt_z = set()
    for _, row in no_gt_df.iterrows():
        world_x = row["coordX"]
        world_y = row["coordY"]
        world_z = row["coordZ"]

        world_coord = np.array([float(world_z), float(world_y), float(world_x)])
        voxel_coord = world_to_voxel_coord(world_coord, origins, spacings)

        voxel_z, voxel_y, voxel_x = voxel_coord
        no_gt_z.add(round(voxel_z))

    for z in range(0, total_images):
        filepath = "{}/z-{}.jpg".format(ct_image_folder[3:], z)
        bboxes = []
        ignore_areas = np.array([])

        if len(nodules_annotation) != 0:
            for _, nodule in nodules_df[nodules_df["z"] == z].iterrows():
                bboxes.append(nodule["bbox"])

        anno = {'bboxes': np.array(bboxes), 'ignoreareas': ignore_areas,
                'filepath': filepath}

        if len(bboxes) != 0:
            if subset_id != test_subset_id:
                image_data_gt.append(anno)
            else:
                image_data_test.append(anno)
                image_data_test_all.append(anno)
        else:
            if z in no_gt_z:
                image_data_no_gt.append(anno)

            if subset_id == test_subset_id:
                image_data_test_all.append(anno)


if not os.path.exists(data_image_folder):
    os.makedirs(data_image_folder)

image_data_gt, image_data_no_gt, image_data_test, image_data_test_all = [], [], [], []

if os.path.exists(mhd_info_csv_path):
    mhd_info_df = pd.read_csv(mhd_info_csv_path)
else:
    mhd_infos = []

for subset_id in tqdm(range(0, 10), desc="subset"):
    current_folder = os.listdir("{}/subset{}".format(download_data_folder, subset_id))

    for file in current_folder:
        if file.endswith(".mhd"):
            series_id = re.search(r'(.*)\.mhd', file).group(1)
            if os.path.exists(mhd_info_csv_path):
                mhd_info = mhd_info_df[mhd_info_df["series_id"] == series_id].iloc[0]
            else:
                mhd_info = extract_mhd_file_info(subset_id, series_id)
                mhd_infos.append(mhd_info)

            # extract_ct_images(mhd_info, mask=False)
            generate_train_test_cache(mhd_info)

if not os.path.exists(mhd_info_csv_path):
    mhd_info_df = pd.DataFrame(mhd_infos)
    mhd_info_df.to_csv(mhd_info_csv_path, index=None)

gt_cache_path = "../data/cache/train_gt"
no_gt_cache_path = "../data/cache/train_no_gt"
test_cache_path = "../data/cache/test"
test_all_cache_path = "../data/cache/test_all-ss{}".format(test_subset_id)

print("image_data_gt", len(image_data_gt))
print("image_data_no_gt", len(image_data_no_gt))
print("image_data_test", len(image_data_test))
print("image_data_test_all", len(image_data_test_all))

with open(gt_cache_path, 'wb') as fid:
    pickle.dump(image_data_gt, fid, 2)
with open(no_gt_cache_path, 'wb') as fid:
    pickle.dump(image_data_no_gt, fid, 2)
with open(test_cache_path, 'wb') as fid:
    pickle.dump(image_data_test, fid, 2)
with open("{}-ss{}".format(test_cache_path, test_subset_id), 'wb') as fid:
    pickle.dump(image_data_test, fid, 2)
with open(test_all_cache_path, 'wb') as fid:
    pickle.dump(image_data_test_all, fid, 2)