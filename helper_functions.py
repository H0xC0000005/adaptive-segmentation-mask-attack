"""
@author: Utku Ozbulak
@repository: github.com/utkuozbulak/adaptive-segmentation-mask-attack
@article: Impact of Adversarial Examples on Deep Learning Models for Biomedical Image Segmentation
@conference: MICCAI-19
"""
from __future__ import annotations

import random
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
        # print(im_as_arr.shape)
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
            iou_mask = iou_mask.astype(float)
    if isinstance(mask1_raw, np.ndarray):
        mask1 = mask1_raw
    else:
        mask1 = mask1_raw.numpy()
    if isinstance(mask2_raw, np.ndarray):
        mask2 = mask2_raw
    else:
        mask2 = mask2_raw.numpy()

    mask1 = mask1.astype(float)
    mask2 = mask2.astype(float)

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
        mask1_single_class = mask1_single_class.astype(float)
        mask2_single_class = mask2_single_class.astype(float)
        if iou_mask is not None:
            # print(iou_mask.dtype)
            # print(mask2_single_class.dtype)
            # print(iou_mask.shape)
            # print(mask2_single_class.shape)
            mask2_single_class *= iou_mask
            mask1_single_class *= iou_mask
        mask1_single_class = mask1_single_class.astype(float)
        mask2_single_class = mask2_single_class.astype(float)

        intersection_single = mask1_single_class * mask2_single_class
        union_single = mask1_single_class + mask2_single_class
        union_single[union_single > 1] = 1

        iou_single = np.sum(intersection_single) / np.sum(union_single)
        # print(">>", np.sum(union_single))
        # print(iou_single)
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
    assert len(shape) > 1, f"in absolute idx to tuple idx, received shape {shape} " \
                           f"with len 1. you dont need this conversion :)"
    cur_divisor = 1
    cur_amount = absolute_idx
    # print(shape)
    # print(cur_amount, "<<")
    result_size = []
    for idx in range(1, len(shape)):
        # size of each elem of dim[0]
        cur_divisor *= shape[idx]
    for idx in range(1, len(shape)):
        cur_idx = cur_amount // cur_divisor
        cur_amount = cur_amount % cur_divisor
        cur_divisor /= shape[idx]
        # print(f"cur amount: {cur_amount}")
        # print(f"cur div: {cur_divisor}")
        # print(f"cur idx: {cur_idx}")
        result_size.append(int(cur_idx))
    result_size.append(int(cur_amount))
    result_size = tuple(result_size)
    return result_size


def l1d(t1: np.ndarray | torch.Tensor | tuple | list,
        t2: np.ndarray | torch.Tensor | tuple | list) -> float:
    """
    make use of pointwise computation of
    """

    def to_tensor(x) -> torch.Tensor:
        if isinstance(x, list) or isinstance(x, tuple):
            return torch.Tensor(x)
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        elif isinstance(x, torch.Tensor):
            return x
        else:
            raise TypeError(f"in to_tensor, provided type {type(x)} that not supported (supported: ndarr/tensor/list/"
                            f"tuple)")

    t1 = to_tensor(t1)
    t2 = to_tensor(t2)
    diff = t1 - t2
    diff = diff.cpu().detach()
    diff = torch.abs(diff)
    diff_sum = torch.sum(diff)
    return float(diff_sum)


def convert_multiclass_mask_to_binary(mask: torch.Tensor,
                                      target_class: int,
                                      invert_flag: bool = False
                                      ) -> torch.Tensor:
    result = copy.deepcopy(mask)
    class_set = set(np.array(torch.unique(mask)))
    temp_class = None
    while temp_class is None or temp_class in class_set:
        temp_class = random.randint(-65536, 65535)

    result[result == target_class] = temp_class
    if not invert_flag:
        result[result != temp_class] = 0
        result[result == temp_class] = 1
    else:
        result[result != temp_class] = 1
        result[result == temp_class] = 0
    return result


def invert_binary_mask(mask: torch.Tensor) -> torch.Tensor:
    maskc = copy.deepcopy(mask)
    maskc[maskc != 0] = 255
    maskc[maskc == 0] = 1
    maskc[maskc == 255] = 0
    return maskc


"""
closures for various purposes
"""


class SelectL1Method:
    def __init__(self):
        self.device = None

    def check_first_arg(self, mask) -> None:
        assert isinstance(mask, torch.Tensor), f"in {self.__class__}, only accept torch.Tensor. got" \
                                               f" {type(mask)}"
        assert len(mask.shape) in (2, 3), f"in {self.__class__}, get mask with shape {mask.shape}. only accept " \
                                          f"unbached greyscale or color"

    def __call__(self, *args, **kwargs):
        print(f"base class l1 sel call")


class SelectRectL1IntenseRegion(SelectL1Method):
    """
    a closure to select rectangular most intense region for L1 perturbation
    """

    def __init__(self, height: int, width: int, number_of_rec: int, allow_overlap: bool = True,
                 overlap_threshold: float = None):
        super().__init__()
        if not allow_overlap and overlap_threshold is None:
            overlap_threshold = (width + height) / (2 * 20)
        self.overlap_threshold = overlap_threshold
        self.allow_overlap = allow_overlap
        self.width = width
        self.height = height
        self.conv = torch.nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=(width, height),
                                    bias=False)
        self.conv.weight.data = torch.ones(size=(1, 1, width, height))
        self.number_of_rec = number_of_rec

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """
        only expect a single mask as args[0]. mask should be 3-dim (not batched).
        return a binary mask on that tensor
        """
        # print(f"call select rect")

        print(f">> called select rect region: \n"
              f"height: {self.width}, width: {self.height}, num: {self.number_of_rec}\n"
              f"allow?: {self.allow_overlap}, thres: {self.overlap_threshold}\n"
              f"received mask size {args[0].shape}")
        mask: torch.Tensor
        mask = args[0]

        self.check_first_arg(mask)
        # 2d to 3d
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        # sum whatever channel we have and unsqueeze
        mask = torch.sum(mask, dim=0)
        mask = mask.unsqueeze(0)
        # change device for this closure
        self.device = mask.device
        self.conv.to(device=self.device)
        # print(mask.shape)
        conv_mask: torch.Tensor
        conv_mask = self.conv(mask)
        conv_mask = torch.squeeze(conv_mask, 0)
        print(conv_mask.shape)
        # print(conv_mask)
        conv_size = conv_mask.shape
        # flatten to 1d
        conv_mask_flattened = conv_mask.flatten()

        """
        select k most intense regions. by default attempt 3*num_of_rec times 
        and if still can't select over then give up
        """
        result_mask = torch.zeros(size=mask.shape[-2:], device=self.device)
        if self.allow_overlap:
            conv_topk_indices = torch.topk(conv_mask_flattened,
                                           k=self.number_of_rec).indices
            # print(conv_topk_indices)
            # print(f"conv size: {conv_size}")
            for flt_idx in conv_topk_indices:
                # return linear idx to original idx
                cur_idx = absolute_index_to_tuple_index(flt_idx, conv_size)
                # mark result mask with corresponding rectangle
                result_mask[cur_idx[-2]: cur_idx[-2] + self.height, cur_idx[-1]: cur_idx[-1] + self.width] += 1
            # clamp all 0+ entries to 1
            result_mask[result_mask != 0] = 1
            return result_mask
        else:
            result_indices = []
            count = 0
            # attempt for 3 * k times. if exhausted, considered failure
            while len(result_indices) < self.number_of_rec and count < 3 * self.number_of_rec:
                conv_mask_flattened = conv_mask.flatten()
                conv_top_index = torch.topk(conv_mask_flattened,
                                            k=1).indices
                cur_idx = absolute_index_to_tuple_index(conv_top_index, conv_size)
                print(cur_idx)
                result_indices.append(conv_top_index)
                # clear out adjacent area
                conv_mask[cur_idx[-2] - self.overlap_threshold + 1: cur_idx[-2] + self.overlap_threshold,
                cur_idx[-1] - self.overlap_threshold + 1: cur_idx[-1] + self.overlap_threshold] = -1000000
                result_mask[cur_idx[-2]: cur_idx[-2] + self.height, cur_idx[-1]: cur_idx[-1] + self.width] += 1
            result_mask[result_mask != 0] = 1
            return result_mask


class SelectTopKPoints(SelectL1Method):
    def __init__(self, k: int, dot_radius: int = 0, threshold: float = None):
        super().__init__()
        self.k = k
        self.threshold = threshold
        self.dot_radius = dot_radius

    def __call__(self, *args, **kwargs):
        mask: torch.Tensor
        mask = args[0]
        self.check_first_arg(mask)
        self.device = mask.device
        # 2d to 3d padding
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        # sum whatever channel we have and unsqueeze
        mask = torch.sum(mask, dim=0)
        mask = mask.unsqueeze(0)
        mask_flattened = torch.flatten(mask)
        mask_size = mask.shape
        result_mask = torch.zeros(size=mask.shape[-2:], device=self.device)
        topk_indices = torch.topk(mask_flattened,
                                  k=self.k).indices
        for flt_idx in topk_indices:
            # return linear idx to original idx
            cur_idx = absolute_index_to_tuple_index(flt_idx, mask_size)
            # mark result mask with corresponding rectangle
            dr = self.dot_radius
            result_mask[cur_idx[-2] - dr: cur_idx[-2] + dr + 1, cur_idx[-1] - dr: cur_idx[-1] + dr + 1] += 1
        # clamp all 0+ entries to 1
        result_mask[result_mask != 0] = 1
        # print("dev>>>", result_mask.device)
        return result_mask


class L1SelectionPostprocessing:
    def __call__(self, *args, **kwargs):
        print(f"base class l1 postpro call")


class MaskingToOriginalClass(L1SelectionPostprocessing):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        org_mask = copy.deepcopy(args[0])
        l1_mask = copy.deepcopy(args[1])
        assert isinstance(org_mask, torch.Tensor), f"in masking to original class, received org mask as type " \
                                                   f"{type(org_mask)} instead of torch tensor"
        assert isinstance(l1_mask, torch.Tensor), f"in masking to original class, received l1 mask as type " \
                                                  f"{type(l1_mask)} instead of torch tensor"
        org_mask[org_mask != 0] = 1
        l1_mask[l1_mask != 0] = 1
        return org_mask * l1_mask
