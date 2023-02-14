import os
import pickle

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm
import math
import multiprocessing
from multiprocessing import Pool
import pandas as pd


def generate_augment_gts(random_crop_items, train_gt_len, result, gt, transforms, transform_names, bbox_safe_transform,
                         filepath,
                         series_id, z, output_folder):
    gt_output_path = "../data/gts"

    bboxes = gt["bboxes"]

    image = cv2.imread(filepath)

    class_labels = ['nodule'] * len(bboxes)

    origin_output_path = "{}/{}-ori.jpg".format(output_folder, z)
    cv2.imwrite(origin_output_path, image)
    anno = {'bboxes': bboxes, 'ignoreareas': np.array([]),
            'filepath': origin_output_path[3:]}
    result.append(anno)

    # cv2.rectangle(image, (bboxes[0][0], bboxes[0][1]),
    #               (bboxes[0][2], bboxes[0][3]), color, 1)
    #
    # cv2.rectangle(image, (bboxes[1][0], bboxes[1][1]),
    #               (bboxes[1][2], bboxes[1][3]), color, 1)
    #
    # cv2.imwrite("test/patched.jpg", image)

    for i in range(len(transforms)):
        transformed = transforms[i](image=image, bboxes=bboxes, nodule=class_labels)
        transformed_image = transformed['image']
        transformed_bbox = transformed['bboxes']

        for j in range(len(transformed_bbox)):
            transformed_bbox[j] = list(map(lambda x: math.ceil(x), transformed_bbox[j]))

        output_image_path = "{}/{}/{}-{}.jpg".format(gt_output_path, series_id, z, transform_names[i])
        anno = {'bboxes': transformed_bbox, 'ignoreareas': np.array([]),
                'filepath': output_image_path[3:]}

        # cv2.rectangle(transformed_image, (transformed_bbox[0][0], transformed_bbox[0][1]),
        #               (transformed_bbox[0][2], transformed_bbox[0][3]), color, 1)
        # cv2.rectangle(transformed_image, (transformed_bbox[1][0], transformed_bbox[1][1]),
        #               (transformed_bbox[1][2], transformed_bbox[1][3]), color, 1)
        # cv2.imwrite("test/{}.jpg".format(i), transformed_image)
        cv2.imwrite(output_image_path, transformed_image)
        result.append(anno)

    for i in range(random_crop_items):
        transformed = bbox_safe_transform(image=image, bboxes=bboxes, nodule=class_labels)
        bbox_safe_image = transformed['image']
        bbox_safe_bbox = transformed['bboxes']

        for j in range(len(bbox_safe_bbox)):
            bbox_safe_bbox[j] = list(map(lambda x: math.ceil(x), bbox_safe_bbox[j]))

        # cropped_image = cv2.rectangle(cropped_image, (
        #     bbox_safe_bbox[0][0], bbox_safe_bbox[0][1]), (bbox_safe_bbox[0][2],
        #                                                   bbox_safe_bbox[0][3]),
        #                                                           color, 1)
        # if len(bbox_safe_bbox) > 1:
        #     cropped_image  = cv2.rectangle(cropped_image, (bbox_safe_bbox[1][0], bbox_safe_bbox[1][1]), (bbox_safe_bbox[1][2],bbox_safe_bbox[1][3]), color, 1)
        #     print(bbox_safe_bbox[1])
        #     cv2.imwrite("test/croped-{}.jpg".format(i + 1), cropped_image)

        output_image_path = "{}/{}/{}-cropped-{}.jpg".format(gt_output_path, series_id, z, i + 1)
        anno = {'bboxes': bbox_safe_bbox, 'ignoreareas': np.array([]),
                'filepath': output_image_path[3:]}

        result.append(anno)
        cv2.imwrite(output_image_path, bbox_safe_image)

    print(len(result), "of", train_gt_len)


def custom_error_callback(error):
    print(error)


if __name__ == '__main__':
    fid = open("../data/cache/train_gt", 'rb')
    train_pickle = pickle.Unpickler(fid, encoding="latin1")
    train_gt = train_pickle.load()

    fid = open("../data/cache/train_no_gt", 'rb')
    train_no_gt_pickle = pickle.Unpickler(fid, encoding="latin1")
    train_no_gt = train_no_gt_pickle.load()

    gt_output_path = "../data/gts"
    if not os.path.exists(gt_output_path):
        os.makedirs(gt_output_path)
    manager = multiprocessing.Manager()
    result = manager.list()

    pool = Pool(multiprocessing.cpu_count())


    horizontal_transform = A.Compose([
        A.HorizontalFlip(p=1),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['nodule']))

    vertical_transform = A.Compose([
        A.VerticalFlip(p=1),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['nodule']))

    rotate90_transform = A.Compose([
        A.Rotate(limit=(90, 90), p=1)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['nodule']))

    rotate180_transform = A.Compose([
        A.Rotate(limit=(180, 180), p=1)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['nodule']))

    rotate270_transform = A.Compose([
        A.Rotate(limit=(-90, -90), p=1)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['nodule']))

    transforms = [horizontal_transform, vertical_transform, rotate90_transform, rotate180_transform,
                  rotate270_transform]

    transform_names = ["h", "v", "r90", "r180", "r270"]

    bbox_safe_transform = A.Compose([
        # A.RandomSizedBBoxSafeCrop(image_size[0], image_size[1], p=1),
        A.Rotate(limit=180, p=1),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['nodule']))

    ratio = 1
    target_gts = len(train_no_gt) * ratio
    gt_output_image_count = math.ceil(target_gts / len(train_gt)) - 6

    print("per image count", gt_output_image_count)

    print("processing....")
    for gt in train_gt:
        filepath = gt["filepath"]
        filepath = "../" + filepath

        series_id = filepath.split("/")[-2]
        z = filepath.split("/")[-1].split(".")[-2]

        output_folder = "{}/{}".format(gt_output_path, series_id)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        pool.apply_async(generate_augment_gts, args=(
            gt_output_image_count, target_gts, result, gt, transforms, transform_names, bbox_safe_transform, filepath,
            series_id, z, output_folder),
                         error_callback=custom_error_callback)

    pool.close()
    pool.join()

    print("finished")

    result = list(result)

    df = pd.DataFrame(result)
    df.set_index("filepath", inplace=True)
    df.to_csv("output/gt-augment.csv")

    augment_gt_cache_path = "../data/cache/train_gt_augment"
    f = open(augment_gt_cache_path, 'wb')
    pickle.dump(result, f, 2)
    f.close()

    print("augment_gts: ", len(result))
