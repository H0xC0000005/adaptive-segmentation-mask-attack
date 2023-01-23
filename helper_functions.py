"""
@author: Utku Ozbulak
@repository: github.com/utkuozbulak/adaptive-segmentation-mask-attack
@article: Impact of Adversarial Examples on Deep Learning Models for Biomedical Image Segmentation
@conference: MICCAI-19
"""
import numpy as np
from PIL import Image
import os
import copy

import torch


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
    modified_copy = modified_copy.astype('uint8')
    if save_flag:
        save_image(modified_copy, im_name, folder_name)
    return modified_copy


def save_batch_image(modified_im, im_name, folder_name='result_images', save_flag=True, normalize=True):
    """
    Discretizes 0-255 (real) image from 0-1 normalized image

    assume input is in [batch, channel, h, w] shape
    """
    modified_copy = copy.deepcopy(modified_im)[0]
    # from [ch h w] to [h w ch]
    modified_copy = modified_copy.transpose(1, 2, 0)
    modified_copy = modified_copy.astype('uint8')
    if save_flag:
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
    pred_img = pred_img.astype('uint8')
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
    diff = diff.astype('uint8')
    save_image(diff, im_name, folder_name)


def save_batch_image_difference(org_image, perturbed_image, im_name, folder_name='result_images'):
    """
    Finds the absolute difference between two images in terms of grayscale palette
    """
    # Process images
    im1 = save_batch_image(org_image, '', '', save_flag=False)
    im2 = save_batch_image(perturbed_image, '', '', save_flag=False)
    # Find difference
    diff = np.abs(im1 - im2)
    # Sum over channel, not needed for colourful saves
    # diff = np.sum(diff, axis=2)
    # Normalize
    diff_max = np.max(diff)
    diff_min = np.min(diff)
    diff = np.clip((diff - diff_min) / (diff_max - diff_min), 0, 254)
    # Enhance x 120, modify this according to your needs
    diff = diff * 1
    diff = diff.astype('uint8')
    save_image(diff, im_name, folder_name)


def save_image(im_as_arr: np.ndarray, im_name, folder_name, normalize=True):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # im_as_arr = im_as_arr.copy()
    image_name_with_path = folder_name + '/' + str(im_name) + '.png'
    # astype() returns a copy but not view so it's safe

    if normalize:
        # this should also be valid for colour image as this gets its max channel
        # print(im_as_arr.shape)
        im_as_arr = im_as_arr / (np.amax(im_as_arr) - np.amin(im_as_arr)) * 254
    im_as_arr = im_as_arr.astype('uint8')
    # print(type(im_as_arr))
    # print(np.unique(im_as_arr))
    # print(im_as_arr)
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


def calculate_multiclass_mask_similarity(mask1_raw: [np.ndarray, torch.Tensor], mask2_raw: [np.ndarray, torch.Tensor]):
    """
    Calculates IOU and pixel accuracy between two masks
    """
    print(f">>> calculating IOU of masks")
    if isinstance(mask1_raw, np.ndarray):
        mask1 = mask1_raw
    else:
        mask1 = mask1_raw.numpy()
    if isinstance(mask2_raw, np.ndarray):
        mask2 = mask2_raw
    else:
        mask2 = mask2_raw.numpy()

    # Calculate IoU; calculate as nparr boolean mask
    # for multiclass, this is average of IOU of each class
    m1_uniq = np.unique(mask1)
    m2_uniq = np.unique(mask2)
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
    accumulated_iou = 0

    for elem in uniq_set:
        mask1_single_class: np.ndarray
        mask2_single_class: np.ndarray
        mask1_single_class = (mask1 == elem)
        mask2_single_class = (mask2 == elem)
        mask1_single_class = mask1_single_class.astype(int)
        mask2_single_class = mask2_single_class.astype(int)

        intersection_single = mask1_single_class * mask2_single_class
        union_single = mask1_single_class + mask2_single_class
        union_single[union_single > 1] = 1

        iou_single = np.sum(intersection_single) / np.sum(union_single)
        print(f"current single iou: {iou_single} with element {elem}")
        accumulated_iou += iou_single
    average_iou = accumulated_iou / len(uniq_set)

    # Calculate pixel accuracy
    correct_pixels = (mask1 == mask2)
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
