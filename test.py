import numpy
import numpy as np

from cityscape_dataset import *
from ext_transforms import ExtUnNormalize, ExtNormalize

# # from helper_functions import save_image
#
# n1 = numpy.ones((200, 200, 3), dtype=float)
n1 = numpy.ones((200, 200, 3), dtype='uint8')

n1[100:150, :, 0:2] = 100
n1[0:50, 0:50, :] = 50
n1[:, 100:150, 0:3] += 80
# n1 = n1.transpose((2, 0, 1))
# print(n1.shape)
#
# n2 = numpy.zeros((200, 200, 3), dtype='uint8')
# n2[100:150, :, 0:2] = 100
# n2[0:50, 0:50, 1:3] = 50
# # save_image(n1, 'test', '/home/peizhu/PycharmProjects/adaptive-segmentation-mask-attack/', normalize=False)
# # save_image(n1, 'test2', '/home/peizhu/PycharmProjects/adaptive-segmentation-mask-attack/')
# # save_batch_image(n1, 'test', '/home/peizhu/PycharmProjects/adaptive-segmentation-mask-attack/', save_flag=True,
# #                  normalize=False)
# save_batch_image_difference(n1, n2, 'test', '/home/peizhu/PycharmProjects/adaptive-segmentation-mask-attack/adv_results/cityscape_results/',
#                             normalize=False, enhance_multiplier=100)
# n1 = n1.transpose((2, 0, 1))

print(np.unique(n1))
save_image(n1, 'test1', '/home/peizhu/PycharmProjects/adaptive-segmentation-mask-attack/', normalize=False)
n = CityscapeDataset.val_transform
un = CityscapeDataset.inverse_val_transform
# n = ExtNormalize(mean=[1, 2, 3], std=[1, 2, 3])
# un = ExtUnNormalize(mean=[1, 2, 3], std=[1, 2, 3])
# n1 = n1.transpose((2, 0, 1))
t1 = torch.from_numpy(n1)
t2 = CityscapeDataset.rgb_to_train_image(n1)
n2 = t2.numpy()
print(np.amax(n2), np.amin(n2))
print(np.unique(n2))
n2 = n2.transpose((1, 2, 0))
save_image(n2, 'test2', '/home/peizhu/PycharmProjects/adaptive-segmentation-mask-attack/', normalize=True)
t3= CityscapeDataset.train_image_to_rgb(t2)
n3 = t3.numpy()
n3 = n3.transpose((1, 2, 0))
print(np.unique(n3))
save_image(n3, 'test3', '/home/peizhu/PycharmProjects/adaptive-segmentation-mask-attack/', normalize=True)


