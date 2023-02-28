"""
@author: Utku Ozbulak
@repository: github.com/utkuozbulak/adaptive-segmentation-mask-attack
@article: Impact of Adversarial Examples on Deep Learning Models for Biomedical Image Segmentation
@conference: MICCAI-19
"""
from __future__ import annotations

import typing
from math import floor

import numpy as np
from PIL import Image
import os
import copy

import torch


def transpose_torch_to_image(img: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if img.shape[0] in (1, 2, 3):
        # assume first dim = channel, else don't transform
        if isinstance(img, np.ndarray):
            img = img.transpose((1, 2, 0))
        else:
            img = torch.transpose(img, 0, 1)
            img = torch.transpose(img, 1, 2)
    return img


def save_input_image(modified_im, im_name, folder_name='result_images', save_flag=True):
    """
    Discretizes 0-255 (real) image from 0-1 normalized image
    """
    modified_copy = copy.deepcopy(modified_im)[0]
    modified_copy = modified_copy * 255
    # Box constraint
    modified_copy[modified_copy > 255] = 255
    modified_copy[modified_copy < 0] = 0
    modified_copy = modified_copy.transpose(1, 2, 0)
    # modified_copy = modified_copy.astype('uint8')
    if save_flag:
        save_image(modified_copy, im_name, folder_name)
    return modified_copy


def save_batch_image(modified_im: np.ndarray | torch.Tensor,
                     im_name: str,
                     folder_name='result_images',
                     save_flag=True,
                     normalize=True,
                     ):
    """
    Discretizes 0-255 (real) image from 0-1 normalized image
    TODO: ...normalized?

    assume input is in [batch, channel, h, w] shape
    """
    if modified_im.shape[0] == 1:
        # batch image, assume batch dim=1
        modified_copy = copy.deepcopy(modified_im)[0]
    else:
        modified_copy = copy.deepcopy(modified_im)

    if save_flag:
        # from [ch h w] to [h w ch]
        modified_copy = transpose_torch_to_image(modified_copy)
        # modified_copy = modified_copy.astype('uint8')
        save_image(modified_copy, im_name, folder_name, normalize=normalize)
    return modified_copy


def save_binary_prediction_image(pred_out, im_name, folder_name='result_images'):
    """
    Saves the prediction of a segmentation models as a real image
    can only do binary
    """
    # Disc. pred image
    pred_img = copy.deepcopy(pred_out)
    pred_img = pred_img * 255
    pred_img[pred_img > 127] = 255
    pred_img[pred_img <= 127] = 0
    # pred_img = pred_img.astype('uint8')
    save_image(pred_img, im_name, folder_name)


def save_binary_image_difference(org_image, perturbed_image, im_name, folder_name='result_images'):
    """
    Finds the absolute difference between two images in terms of grayscale palette
    """
    # Process images
    im1 = save_input_image(org_image, '', '', save_flag=False)
    im2 = save_input_image(perturbed_image, '', '', save_flag=False)
    # Find difference
    diff = np.abs(im1 - im2)
    # Sum over channel
    diff = np.sum(diff, axis=2)
    # Normalize
    diff_max = np.max(diff)
    diff_min = np.min(diff)
    diff = np.clip((diff - diff_min) / (diff_max - diff_min), 0, 1)
    # Enhance x 120, modify this according to your needs
    diff = diff * 30
    # diff = diff.astype('uint8')
    save_image(diff, im_name, folder_name)


def save_batch_image_difference(org_image: np.ndarray | torch.Tensor,
                                perturbed_image: np.ndarray | torch.Tensor,
                                im_name: str | None,
                                folder_name: str = 'result_images',
                                normalize=True,
                                enhance_multiplier=255,
                                save_flag=True):
    """
    Finds the absolute difference between two images in terms of grayscale palette
    """
    assert not (save_flag and im_name is None), f"attempt to save without name (im_name == None)"
    if type(org_image) != type(perturbed_image):
        raise TypeError(f"in save_batch_image, type of img1 {type(org_image)} doesn't match type of img2:"
                        f" {type(perturbed_image)}")
    if isinstance(org_image, torch.Tensor):
        is_tensor = True
    else:
        is_tensor = False
    # Process images
    im1 = save_batch_image(org_image, '', '', save_flag=False)
    im2 = save_batch_image(perturbed_image, '', '', save_flag=False)
    # Find difference
    # Sum over channel, not needed for colourful saves
    if is_tensor:
        diff = torch.abs(im1 - im2)
        diff_max = torch.max(diff)
        diff_min = torch.min(diff)
    else:
        diff = np.abs(im1 - im2)
        diff_max = np.max(diff)
        diff_min = np.min(diff)

    # Normalize. 0-1 min-max normalization
    if normalize and diff_max != diff_min:
        diff = (diff - diff_min) / (diff_max - diff_min)
    # Enhance x 255 to get full scale image
    if save_flag:
        diff = diff * enhance_multiplier
        diff = transpose_torch_to_image(diff)
        # diff = diff.astype('uint8')
        save_image(diff, im_name, folder_name, normalize=normalize)
    return diff


def save_image(im_as_arr: np.ndarray | torch.Tensor, im_name, folder_name: os.PathLike | str, normalize=True):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    if isinstance(im_as_arr, torch.Tensor):
        im_as_arr = im_as_arr.cpu().detach().numpy()
    # im_as_arr = im_as_arr.copy()
    im_as_arr: np.ndarray
    if len(im_as_arr.shape) > 3:
        if im_as_arr.shape[0] == 1:
            im_as_arr = np.squeeze(im_as_arr, axis=0)
        else:
            raise ValueError(f"received image with 4+ dim but first dim not 1 (thus not squeezable)")

    if im_as_arr.shape[0] in (1, 3):
        # assume first channel is greyscale or 3-ch
        print(im_as_arr.shape)
        im_as_arr = im_as_arr.transpose((1, 2, 0))
    if folder_name[-1] == "/":
        folder_name = folder_name[:-1]
    image_name_with_path = folder_name + '/' + str(im_name) + '.png'
    # astype() returns a copy but not view so it's safe

    if normalize:
        # this should also be valid for colour image as this gets its max channel
        # print(im_as_arr.shape)
        imax = np.amax(im_as_arr)
        imin = np.amin(im_as_arr)
        if imax != imin:
            im_as_arr = im_as_arr / (imax - imin) * 254
    im_as_arr = im_as_arr.astype('uint8')
    pred_img = Image.fromarray(im_as_arr)

    # pred_img.show(im_name)

    pred_img.save(image_name_with_path)


def load_model(path_to_model):
    """
    Loads pytorch models from disk
    """
    model = torch.load(path_to_model)
    print(f"loaded model {path_to_model} with type {type(model)}")
    return model


def calculate_binary_mask_similarity(mask1: np.ndarray, mask2: np.ndarray):
    """
    Calculates IOU and pixel accuracy between two masks
    """
    # Calculate IoU
    intersection = mask1 * mask2
    union = mask1 + mask2
    # Update intersection 2s to 1
    union[union > 1] = 1
    iou = np.sum(intersection) / np.sum(union)

    # Calculate pixel accuracy
    correct_pixels = (mask1 == mask2)
    correct_pixels: np.ndarray
    pixel_acc = np.sum(correct_pixels) / (correct_pixels.shape[0] * correct_pixels.shape[1])
    return iou, pixel_acc


def calculate_multiclass_mask_similarity(mask1_raw: np.ndarray | torch.Tensor,
                                         mask2_raw: np.ndarray | torch.Tensor,
                                         target_classes: list | None = None,
                                         *,
                                         iou_mask: torch.Tensor | np.ndarray = None):
    """
    Calculates IOU and pixel accuracy between two masks
    can optionally provide iou mask, where iou is only computed with pixels within mask
    """
    # print(f">>> calculating IOU of masks")
    if iou_mask is not None:
        if isinstance(iou_mask, torch.Tensor):
            iou_mask = iou_mask.cpu().detach().numpy()
    if isinstance(mask1_raw, np.ndarray):
        mask1 = mask1_raw
    else:
        mask1 = mask1_raw.numpy()
    if isinstance(mask2_raw, np.ndarray):
        mask2 = mask2_raw
    else:
        mask2 = mask2_raw.numpy()
    m1_uniq = np.unique(mask1)
    m2_uniq = np.unique(mask2)
    # print(f"unique m1 elem: {m1_uniq}")
    # print(f"unique m2 elem: {m2_uniq}")
    #
    if target_classes is None:
        # Calculate IoU; calculate as nparr boolean mask
        # for multiclass, this is average of IOU of each class
        if isinstance(m1_uniq, np.ndarray):
            m1_uniq_iter = set(m1_uniq.flatten())
        else:
            m1_uniq_iter = set(m1_uniq)
        if isinstance(m1_uniq, np.ndarray):
            m2_uniq_iter = set(m2_uniq.flatten())
        else:
            m2_uniq_iter = set(m2_uniq)
        # only counts classes that both masks have

        uniq_set = m1_uniq_iter.intersection(m2_uniq_iter)
        # print(f"calculating for all intersected classes of two masks: {uniq_set}")
    else:
        # print(f"calculating for specified unique class list: {target_classes}")
        uniq_set = target_classes
    accumulated_iou = 0

    for elem in uniq_set:
        mask1_single_class: np.ndarray
        mask2_single_class: np.ndarray
        mask1_single_class = (mask1 == elem)
        mask2_single_class = (mask2 == elem)
        mask1_single_class = mask1_single_class.astype(int)
        mask2_single_class = mask2_single_class.astype(int)
        if iou_mask is not None:
            mask2_single_class *= iou_mask
            mask1_single_class *= iou_mask
        mask1_single_class = mask1_single_class.astype(int)
        mask2_single_class = mask2_single_class.astype(int)

        intersection_single = mask1_single_class * mask2_single_class
        union_single = mask1_single_class + mask2_single_class
        union_single[union_single > 1] = 1

        iou_single = np.sum(intersection_single) / np.sum(union_single)
        # print(f"current single iou: {iou_single} with element {elem}")
        accumulated_iou += iou_single
    average_iou = accumulated_iou / len(uniq_set)

    # Calculate pixel accuracy
    if target_classes is not None:
        m1c = copy.deepcopy(mask1)
        m2c = copy.deepcopy(mask2)
        for elem in target_classes:
            m1c[m1c == elem] = 1
            m2c[m2c == elem] = 1
        m1c[m1c != 1] = 0
        m2c[m2c != 1] = 0
    else:
        m1c = mask1
        m2c = mask2
    correct_pixels = (m1c == m2c)
    correct_pixels: np.ndarray
    pixel_acc = np.sum(correct_pixels) / (correct_pixels.shape[0] * correct_pixels.shape[1])
    return average_iou, pixel_acc


def calculate_image_distance(im1, im2):
    """
    Calculates L2 and L_inf distance between two images
    """
    # Calculate L2 distance
    l2_dist = torch.dist(im1, im2, p=2).item()

    # Calculate Linf distance
    diff = torch.abs(im1 - im2)
    diff = torch.max(diff, dim=2)[0]  # 0-> item, 1-> pos
    linf_dist = torch.max(diff).item()
    return l2_dist, linf_dist


def report_image_statistics(img: np.ndarray | torch.Tensor):
    if isinstance(img, torch.Tensor):
        img_as_ndarr = img.cpu().detach().numpy()
    else:
        img_as_ndarr = img
    mean = np.mean(img_as_ndarr)
    median = np.median(img_as_ndarr)
    l1sum = np.sum(img_as_ndarr)

    print(f"max:\t{np.amax(img_as_ndarr)}, min:\t{np.amin(img_as_ndarr)}, mean:\t{mean}, median:\t{median}\n"
          f"type: \t{type(img)}, L1 sum:\t{l1sum}")


"""
utilities for closure. can also be used for other purposes
"""


def compute_conv2d_output_shape(in_size: tuple | list,
                                kernel_size: tuple | list,
                                padding: tuple | list = (0, 0),
                                dilation: tuple | list = (1, 1),
                                stride: tuple | list = (1, 1),
                                ) -> tuple:
    """
    assume (width, height)
    """
    assert len(in_size) == 2 and len(kernel_size) == 2 and len(padding) == 2 and len(dilation) == 2 and \
           len(stride) == 2, \
        f"in compute conv2d out size, get one input without len 2. make sure all inputs have len 2."
    assert dilation[0] > 0 and dilation[1] > 0, f"in compute conv2d out size, get dilation <= 0: {dilation}"
    res_width = floor((in_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    res_height = floor((in_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    return res_width, res_height


def absolute_index_to_tuple_index(absolute_idx: int,
                                  shape: tuple) -> tuple:
    assert len(shape) > 1, f"in absolute idx to tuple idx, received shape with len 1. you dont need this conversion :)"
    cur_divisor = 1
    cur_amount = absolute_idx
    result_size = []
    for idx in range(1, len(shape)):
        # size of each elem of dim[0]
        cur_divisor *= shape[idx]
    for idx in range(1, len(shape)):
        cur_idx = cur_amount // cur_divisor
        cur_amount = cur_amount % cur_divisor
        cur_divisor /= shape[idx]
        result_size.append(cur_idx)
    result_size = tuple(result_size)
    return result_size


"""
closures for various purposes
"""


class SelectRectL1IntenseRegion:
    """
    a closure to select rectangular most intense region for L1 perturbation
    """

    def __init__(self, height: int, width: int, number_of_rec: int):
        self.height = height
        self.width = width
        self.conv = torch.nn.Conv2d(in_channels=3,
                                    out_channels=1,
                                    kernel_size=(width, height),
                                    bias=False)
        self.conv.weight.data = torch.ones(size=(width, height))
        self.number_of_rec = number_of_rec
        self.device = None

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """
        only expect a single mask as args[0]
        return a binary mask on that tensor
        """
        mask: torch.Tensor
        mask = args[0]
        assert isinstance(mask, torch.Tensor), f"in select l1 rect region, only accept torch.Tensor. got" \
                                               f" {type(mask)}"
        # change device for this closure
        self.device = mask.device
        self.conv.to(device=self.device)
        conv_mask: torch.Tensor
        conv_mask = self.conv(mask)
        conv_shape = tuple(conv_mask.shape)
        conv_size = compute_conv2d_output_shape(in_size=(conv_mask.shape[-2], conv_mask.shape[-1]),
                                                kernel_size=(self.width, self.height),
                                                )
        # flatten to 1d
        conv_mask_flattened = conv_mask.flatten()
        conv_topk_indice = torch.topk(conv_mask_flattened,
                                      k=self.number_of_rec)
        result_mask = torch.zeros(size=mask.shape, device=self.device)
        for flt_idx in conv_topk_indice:
            # return linear idx to original idx
            cur_idx = absolute_index_to_tuple_index(flt_idx, conv_shape)
            # mark result mask with corresponding rectangle
            result_mask[cur_idx[-2]: cur_idx[-2] + self.width, cur_idx[-1]: cur_idx[-1] + self.height] += 1
        # clamp all 0+ entries to 1
        result_mask[result_mask != 0] = 1
        return result_mask
