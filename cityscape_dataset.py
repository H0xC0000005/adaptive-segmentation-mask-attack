"""
@author: Utku Ozbulak
@repository: github.com/utkuozbulak/adaptive-segmentation-mask-attack
@article: Impact of Adversarial Examples on Deep Learning Models for Biomedical Image Segmentation
@conference: MICCAI-19
"""
import typing
from collections import namedtuple

import numpy as np
import glob
from PIL import Image
import copy

import torch
from torch.utils.data.dataset import Dataset

from helper_functions import save_binary_prediction_image, save_image


class CityscapeDataset(Dataset):
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    # name id trainid category categoryid has_instances ignore_in_eval colour
    classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    def __init__(self, image_path, mask_path, in_size=572, out_size=388):
        super(CityscapeDataset, self).__init__()
        # paths to all images and masks
        self.mask_path = mask_path
        self.image_list = glob.glob(str(image_path) + str("/*"))
        self.data_len = len(self.image_list)
        self.in_size, self.out_size = in_size, out_size
        print('Dataset size:', self.data_len)

    @classmethod
    def encode_target(cls, target: np.ndarray):
        return cls.id_to_train_id[target]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        # target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    @classmethod
    def encode_full_class_array(cls, array: typing.Iterable) -> typing.Set[int]:
        # since 255 is out-of-training label, need to kick it outside encoding
        # also make sure it's unique classes
        # may also has cls list not in training category that would map to 255
        # just keep them for outside processing
        result = set([cls.id_to_train_id[elem] for elem in array if elem != 255])
        # result = [cls.id_to_train_id[elem] for elem in array if elem != 255]

        # print(f"input array: {array}")
        # print(f"encoded result: {result}")
        return result

    @classmethod
    def normalize_label(cls, array: typing.Iterable[int], normalize_range: float = 1) -> typing.Set[float]:
        label_set = set(array)
        # discard out of training if available without error
        label_set.discard(255)
        norm_set = {elem / 19 * normalize_range for elem in label_set}
        return norm_set

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        # target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index) -> (str, torch.Tensor, torch.Tensor):
        # --- Image operations --- #
        image_path = self.image_list[index]
        image_name = image_path[image_path.rfind('/') + 1:]
        # Read image
        im_as_im = Image.open(image_path)

        im_as_np = np.asarray(im_as_im)
        # transpose tf style colour channel as -1 dim to torch style as 0 dim
        if im_as_np.shape[2] in (1, 2, 3, 4):
            im_as_np = im_as_np.transpose((2, 0, 1))
        """
        # Sanity check
        img1 = Image.fromarray(im_as_np.transpose(1, 2, 0))
        img1.show()
        """
        # Normalize image
        """ is this normalize really necessary or correct?"""
        # im_as_np = im_as_np / 255
        # Convert numpy array to tensor
        im_as_tensor: torch.Tensor
        im_as_tensor = torch.from_numpy(im_as_np).float()

        # --- Mask operations --- #
        # Read mask
        msk_as_im = Image.open(self.mask_path + '/' + image_name)
        msk_as_np = np.asarray(msk_as_im)

        # trick to convert 31 class in cityscapes to repo training scheme with 19 classes
        msk_as_np = self.encode_target(msk_as_np)

        # WARNING: magic number 19: 19 classes used in deeplabv3 cityscape model to normalize mask
        # msk_as_np = msk_as_np / 19

        """
        # Sanity check
        img2 = Image.fromarray(msk_as_np)
        img2.show()
        """

        # keep mask label as the same for cityscape
        msk_as_np = msk_as_np.copy()
        # a trick that torch doesn't support r-only tensor from nparr. deep copy to make it writable
        msk_as_np.setflags(write=True)
        msk_as_tensor = torch.from_numpy(msk_as_np).long()  # Convert numpy array to tensor

        DEBUG = True
        if DEBUG:
            # Save masks
            print(f"< in dataset saving masks to location "
                  f"/home/peizhu/PycharmProjects/adaptive-segmentation-mask-attack/")
            print(f"unique elem in this mask: {np.unique(msk_as_np)}")
            save_image(msk_as_tensor.numpy(), 'mask_debug', "/home/peizhu/PycharmProjects/adaptive"
                                                            "-segmentation-mask-attack/")
        return image_name, im_as_tensor, msk_as_tensor

    def __len__(self):
        return self.data_len
