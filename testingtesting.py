import torch
import gc
import cv2
from utils.util import *
import os
from PIL import Image

    # housekeeping

# torch.cuda.empty_cache()
# gc.collect()
# mask_path = os.path.join(root, 'VOCdevkit', 'VOC2007', 'SegmentationClass')
# mask = os.path.join(mask_path, '009841' + '.png')
# # mask_photo = plt.imread(mask)

# mask = np.array(Image.open(mask)) 
# # mask[mask == 0] = 1
# print(np.unique(mask))
# cv2.imshow("a", mask)
# plt.show()

a = np.loadtxt('mask.txt')

print(np.unique(a)*255)