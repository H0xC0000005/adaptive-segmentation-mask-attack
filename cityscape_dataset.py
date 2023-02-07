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
import ext_transforms as et

class CityscapeDataset(Dataset):
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    # name id trainid category categoryid has_instances ignore_in_eval colour
    classes = [
        CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    # train_transform = et.ExtCompose([
    #     # et.ExtResize( 512 ),
    #     et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
    #     et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    #     et.ExtRandomHorizontalFlip(),
    #     et.ExtToTensor(),
    #     et.ExtNormalize(mean=[0.485, 0.456, 0.406],
    #                     std=[0.229, 0.224, 0.225]),
    # ])

    val_transform = et.ExtCompose([
        # et.ExtResize( 512 ),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    inverse_val_transform = et.ExtCompose([et.ExtUnNormalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])

    def __init__(self, image_path, mask_path, *, transform=None, inv_transform=None):
        super(CityscapeDataset, self).__init__()
        # paths to all images and masks
        self.mask_path = mask_path
        self.image_list = glob.glob(str(image_path) + str("/*"))
        self.data_len = len(self.image_list)
        if transform is not None:
            print(f"specified transform : {transform}")
        else:
            print(f"transform not specified")
        if transform is None:
            self.transform = CityscapeDataset.val_transform
            self.inv_transform = CityscapeDataset.inverse_val_transform
        else:
            if transform is not None and inv_transform is not None:
                self.transform = transform
                self.inv_transform = inv_transform
            else:
                raise RuntimeError(f"transform and inv_transform are not all specified.")

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
    def rgb_to_train_image(cls, img: typing.Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        result, _ = cls.val_transform(img, None)
        return result

    @classmethod
    def train_image_to_rgb(cls, img: torch.Tensor) -> torch.Tensor:
        result, _ = cls.inverse_val_transform(img, None)
        return result

    def __getitem__(self, index) -> (str, torch.Tensor, torch.Tensor):
        DEBUG = True
        """

        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        if self.transform:
            image, target = self.transform(image, target)
        target = self.encode_target(target)
        return image, target

        """
        # --- Image operations --- #
        image_path = self.image_list[index]
        image_name = image_path[image_path.rfind('/') + 1:]
        print(f"fetched image name {image_name}")
        # Read image
        im_as_im = Image.open(image_path)

        # --- Mask operations --- #
        # Read mask
        msk_as_im = Image.open(self.mask_path + '/' + image_name)
        if DEBUG:
            db = np.array(im_as_im)
            print(f"before transfm, max of image: {np.amax(db)}, min of image: {np.amin(db)}")
            # print(f"unique elem of this image: {np.unique(db)}")

        if self.transform is not None:
            # before: 0:255
            # after: -2.1-2.6
            im_as_tensor, msk_as_tensor = self.transform(im_as_im, msk_as_im)
            # print(np.unique(im_as_tensor.numpy()))
        else:
            im_as_tensor = torch.from_numpy(np.array(im_as_im))
            msk_as_tensor = torch.from_numpy(np.array(msk_as_im))
        # print(type(msk_as_tensor))
        msk_as_np = self.encode_target(msk_as_tensor)
        msk_as_tensor = torch.from_numpy(msk_as_np)

        if DEBUG:
            # Save masks
            # print(f"type of im, msk: {type(im_as_tensor), type(msk_as_tensor)}")
            msk_as_np = msk_as_tensor.numpy()
            print(f"< in dataset saving masks to location "
                  f"/home/peizhu/PycharmProjects/adaptive-segmentation-mask-attack/")
            print(f"unique elem in this mask: {np.unique(msk_as_np)}")
            db = im_as_tensor.numpy()
            print(f"max of image: {np.amax(db)}, min of image: {np.amin(db)}")
            save_image(msk_as_tensor.numpy(), 'mask_debug', "/home/peizhu/PycharmProjects/adaptive"
                                                            "-segmentation-mask-attack/")
        return image_name, im_as_tensor, msk_as_tensor

    def __len__(self):
        return self.data_len
