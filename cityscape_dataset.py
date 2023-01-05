"""
@author: Utku Ozbulak
@repository: github.com/utkuozbulak/adaptive-segmentation-mask-attack
@article: Impact of Adversarial Examples on Deep Learning Models for Biomedical Image Segmentation
@conference: MICCAI-19
"""
import numpy as np
import glob
from PIL import Image
import copy

import torch
from torch.utils.data.dataset import Dataset

from helper_functions import save_prediction_image, save_image


class CityscapeDataset(Dataset):
    def __init__(self, image_path, mask_path, in_size=572, out_size=388):
        super(CityscapeDataset, self).__init__()
        # paths to all images and masks
        self.mask_path = mask_path
        self.image_list = glob.glob(str(image_path) + str("/*"))
        self.data_len = len(self.image_list)
        self.in_size, self.out_size = in_size, out_size
        print('Dataset size:', self.data_len)

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
        im_as_np = im_as_np / 255
        # Convert numpy array to tensor
        im_as_tensor = torch.from_numpy(im_as_np).float()

        # --- Mask operations --- #
        # Read mask
        msk_as_im = Image.open(self.mask_path + '/' + image_name)
        msk_as_np = np.asarray(msk_as_im)

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
