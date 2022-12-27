import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk

image_width = 512
image_height = 512

work_data_folder = "data/"
ground_truth_csv_path = "data/ground_truth.csv"

if not os.path.exists(work_data_folder):
    os.makedirs(work_data_folder)


def read_detection_result():
    detection_result_folder = "../output/valresults/caltech/h/off"

    detection_dfs = {}
    for epoch_id in tqdm(range(51, 121), desc="test epoch"):
        folder_name = "0{}".format(epoch_id) if epoch_id < 100 else str(epoch_id)
        current_folder_path = "{}/{}".format(detection_result_folder, folder_name)
        current_folder_files = os.listdir(current_folder_path)

        detection_dfs.setdefault(epoch_id, {})
        for filename in current_folder_files:
            series_id = filename[0:-4]
            file = os.path.join(current_folder_path, filename)
            try:
                result_df = pd.read_csv(file, delimiter=" ", header=None)
                result_df.columns = ["z-index", "top_left_x", "top_left_y", "width", "height", "possibility"]
            except Exception:
                result_df = pd.DataFrame([], columns=["z-index", "top_left_x", "top_left_y", "width", "height", "possibility"])

            detection_dfs.get(epoch_id).setdefault(series_id, result_df)

    return detection_dfs


def world_to_voxel_coord(world_coord, origin, spacing):
    stretched_voxel_coord = np.absolute(world_coord - origin)
    voxel_coord = stretched_voxel_coord / spacing
    return voxel_coord


def read_ground_truth_data():
    if os.path.exists(ground_truth_csv_path):
        annotations_world_coord = pd.read_csv(ground_truth_csv_path)
        return annotations_world_coord

    annotations_path = "../downloads/annotations.csv"
    annotations_df = pd.read_csv(annotations_path, delimiter=",")
    series_uids = set(annotations_df["seriesuid"].values)

    parsed_annotations = []

    for subset_id in tqdm(range(0, 10), desc="subset"):
        download_data_folder = "../downloads/data"
        data_folder = os.listdir("{}/subset{}".format(download_data_folder, subset_id))

        for file in data_folder:
            if file.endswith(".mhd"):
                series_id = re.search(r'(.*)\.mhd', file).group(1)

                if series_id in series_uids:
                    mhd_file_path = "{}/subset{}/{}.mhd".format(download_data_folder, subset_id, series_id)
                    itk_image = sitk.ReadImage(mhd_file_path, sitk.sitkFloat32)

                    origins = np.array(list(reversed(itk_image.GetOrigin())))
                    spacings = np.array(list(reversed(itk_image.GetSpacing())))

                    current_annotations = annotations_df[annotations_df["seriesuid"] == series_id]

                    for index, row in current_annotations.iterrows():
                        world_x = row["coordX"]
                        world_y = row["coordY"]
                        world_z = row["coordZ"]
                        d = row["diameter_mm"]

                        world_coord = np.array([float(world_z), float(world_y), float(world_x)])
                        voxel_coord = world_to_voxel_coord(world_coord, origins, spacings).astype(int)

                        voxel_z, voxel_y, voxel_x = voxel_coord
                        diameter = int(d / spacings[2])

                        annotation = [series_id,
                                      voxel_z,
                                      max(voxel_x, 0),
                                      max(voxel_y, 0),
                                      min(voxel_x + diameter, image_width - 1),
                                      min(voxel_y + diameter, image_height - 1)
                                      ]
                        parsed_annotations.append(annotation)

    annotations_world_coord = pd.DataFrame(parsed_annotations)
    annotations_world_coord.columns = ["series_id", "z-index", "top_left_x", "top_left_y", "width", "height"]
    annotations_world_coord.set_index("series_id", inplace=True)

    annotations_world_coord.to_csv(ground_truth_csv_path)

    return annotations_world_coord


if __name__ == '__main__':
    print("load detections data...")
    detections = read_detection_result()
    print()

    print("load ground truth data...")
    annotations = read_ground_truth_data()
    pass


# function res = evalAlgs( plotName, algs, exps, gts, dts )
# % Evaluate every algorithm on each experiment
# %
# % OUTPUTS
# %  res    - nGt x nDt cell of all evaluations, each with fields
# %   .stra   - string identifying algorithm
# %   .stre   - string identifying experiment
# %   .gtr    - [n x 1] gt result bbs for each frame [x y w h match]
# %   .dtr    - [n x 1] dt result bbs for each frame [x y w h score match]
# fprintf('Evaluating: %s\n',plotName); nGt=length(gts); nDt=length(dts);
# res=repmat(struct('stra',[],'stre',[],'gtr',[],'dtr',[]),nGt,nDt);
# for g=1:nGt
#   for d=1:nDt
#     gt=gts{g}; dt=dts{d}; n=length(gt); assert(length(dt)==n);
#     stra=algs(d).name; stre=exps(g).name;
#     fName = [plotName '/ev-' [stre '-' stra] '.mat'];
#     if(exist(fName,'file')), R=load(fName); res(g,d)=R.R; continue; end
#     fprintf('\tExp %i/%i, Alg %i/%i: %s/%s\n',g,nGt,d,nDt,stre,stra);
#     hr = exps(g).hr.*[1/exps(g).filter exps(g).filter];
#     for f=1:n, bb=dt{f}; dt{f}=bb(bb(:,4)>=hr(1) & bb(:,4)<hr(2),:); end
#     [gtr,dtr] = bbGt('evalRes',gt,dt,exps(g).overlap);
#     R=struct('stra',stra,'stre',stre,'gtr',{gtr},'dtr',{dtr});
#     res(g,d)=R; save(fName,'R');
#   end
# end
# end

# read detections
# calculate alpha0
