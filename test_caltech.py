from __future__ import division
import os
import time
import cPickle
from keras.layers import Input
from keras.models import Model
from keras_csp import config, bbox_process
from keras_csp.utilsfunc import *
from keras_csp import resnet50 as nn
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '1' # 0 if use GPU, since I only have one GPU, change to '1' to use CPU
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
C = config.Config()

cache_path = 'data/cache/test'
with open(cache_path, 'rb') as fid:
    val_data = cPickle.load(fid)
num_imgs = len(val_data)
print 'num of val samples: {}'.format(num_imgs)

input_shape_img = (C.size_test[0], C.size_test[1], 3)

img_input = Input(shape=input_shape_img)

# define the network prediction
preds = nn.nn_p3p4p5(img_input, offset=C.offset, num_scale=C.num_scale, trainable=True)
model = Model(img_input, preds)

if C.offset:
    w_path = 'output/valmodels/caltech/%s/off2' % (C.scale)
    out_path = 'output/valresults/caltech/%s/off' % (C.scale)
else:
    w_path = 'output/valmodels/caltech/%s/nooff' % (C.scale)
    out_path = 'output/valresults/caltech/%s/nooff' % (C.scale)

if not os.path.exists(out_path):
    os.makedirs(out_path)

files = sorted(os.listdir(w_path))
for w_ind in range(1, C.num_epochs + 1):
    for f in files:
        if f.split('_')[0] == 'net' and int(f.split('_')[1][1:]) == w_ind:
            cur_file = f
            break

    weight1 = os.path.join(w_path, cur_file)
    print 'load weights from {}'.format(weight1)
    model.load_weights(weight1, by_name=True)
    res_path = os.path.join(out_path, '%03d'%int(str(w_ind)))
    print "res_path: " + res_path

    if not os.path.exists(res_path):
        os.mkdir(res_path)

    file_lists = set()
    start_time = time.time()
    for f in tqdm(range(num_imgs)):
        # filepath: data/images/1.3.6.1.4.1.14519.5.2.1.6279.6001.765459236550358748053283544075/z-154.jpg
        filepath = val_data[f]['filepath']
        filepath_next = val_data[f + 1]['filepath'] if f < num_imgs - 1 else val_data[f]['filepath']

        series_id = filepath.split("/")[-2]
        z_index = int(filepath.split("/")[-1][2:-4])
        series_id_next = filepath_next.split('/')[-2]

        output_file_path = os.path.join(res_path, "{}.txt".format(series_id))

        if series_id not in file_lists:
            res_all = []
            file_lists.add(series_id)

        img = cv2.imread(filepath)
        x_rcnn = format_img(img, C)
        Y = model.predict(x_rcnn)

        if C.offset:
            boxes = bbox_process.parse_det_offset(Y, C, score=0.01,down=4)
        else:
            boxes = bbox_process.parse_det(Y, C, score=0.01, down=4, scale=C.scale)

        if len(boxes)>0:
            f_res = np.repeat(z_index, len(boxes), axis=0).reshape((-1, 1))
            boxes[:, [2, 3]] -= boxes[:, [0, 1]]
            res_all += np.concatenate((f_res, boxes), axis=-1).tolist()

        if f == num_imgs - 1 or series_id_next != series_id:
            np.savetxt(output_file_path, np.array(res_all), fmt='%6f')

    print time.time() - start_time


