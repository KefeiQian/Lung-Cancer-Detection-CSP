import os

import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np

from scrollview import ScrollView

series_id = "1.3.6.1.4.1.14519.5.2.1.6279.6001.112767175295249119452142211437"
subset_id = 9
file = "../../downloads/data/subset{}/{}.mhd".format(subset_id, series_id)
# file = os.path.join("downloads/seg-lungs-LUNA16/1.3.6.1.4.1.14519.5.2.1.6279.6001.692598144815688523679745963696.mhd")

itk_image = sitk.ReadImage(file, sitk.sitkFloat32)
ct_scans = sitk.GetArrayFromImage(itk_image)

origins = np.array(list(reversed(itk_image.GetOrigin())))
spacings = np.array(list(reversed(itk_image.GetSpacing())))

# print(origins)
# print(spacings)


# def draw_sub_plot():
#     px = 1 / plt.rcParams['figure.dpi']
#     plt.figure(figsize=(512 * px, 512 * px))
#     plt.subplots_adjust(0, 0, 1, 1, 0, 0)
#     # rows = math.ceil(ct_scans.shape[0] / 6)
#     for i in range(ct_scans.shape[0]):
#         # plt.subplot(rows, 6, i + 1)
#         plt.gray()
#         plt.margins(0, 0)
#         plt.axis('off')
#         plt.imshow(ct_scans[i])
#         plt.savefig("output/{}.jpg".format(i), bbox_inches='tight', pad_inches=0)
#         plt.clf()
#     # plt.show(block=True)


plt.gray()
fig, ax = plt.subplots()
ScrollView(ct_scans).plot(ax)
# plt.show(block=True)

# print(ct_scans.shape)
# draw_sub_plot()
