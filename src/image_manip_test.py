import numpy as np
from PIL import Image


im_as_im = Image.open("../data/image_samples/G_5_R_1.png")

print(im_as_im)
# im_as_im.show()

im_as_np = np.asarray(im_as_im)
print(im_as_np.shape)

im_as_np = im_as_np.transpose((2, 0, 1))
print(im_as_np.shape)

# Crop image
im_as_np = im_as_np[:, 70:70+408, 10:-10]  # 428 x 428 ? 408 x 408
print(im_as_np.shape)

# Pad image
pad_size = 82
im_as_np = np.asarray([np.pad(single_slice, pad_size, mode='edge')
                       for single_slice in im_as_np])
print(im_as_np.shape)
