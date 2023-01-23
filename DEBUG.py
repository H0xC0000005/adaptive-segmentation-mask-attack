import numpy as np

from helper_functions import save_image

# arr = np.array([1,2,3], dtype='uint8')
arr = np.array([1,2,3])
arr = arr.astype('uint8')

save_image(arr, "save_test", "/home/peizhu/PycharmProjects/adaptive-segmentation-mask-attack/")
