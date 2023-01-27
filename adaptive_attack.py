"""
@author: Utku Ozbulak
@repository: github.com/utkuozbulak/adaptive-segmentation-mask-attack
@article: Impact of Adversarial Examples on Deep Learning Models for Biomedical Image Segmentation
@conference: MICCAI-19
"""
from __future__ import annotations

import random
import gc
import typing
from os.path import isfile

import numpy as np
import torch
from torch.autograd import Variable

# from torch.autograd import Variable
# In-repo imports
from helper_functions import *
from cityscape_dataset import CityscapeDataset
import os

random.seed(4)


def debug_image_as_arr(arr: np.ndarray, name: str = "test",
                       path: str = '/home/peizhu/PycharmProjects/adaptive-segmentation-mask-attack/'):
    counter = 0
    arr_cp = copy.deepcopy(arr)
    while isfile(path + name + str(counter)):
        counter += 1
    if arr_cp.shape[0] in (1, 2, 3, 4) and len(arr_cp.shape) >= 3:
        arr_cp = arr_cp.transpose([len(arr_cp.shape)] + [x for x in range(0, len(arr_cp.shape))])
    img = Image.fromarray(arr_cp)
    img.save(path + name + str(counter))


class AdaptiveSegmentationMaskAttack:
    def __init__(self, device_id, model, tau, beta, *, use_cpu=False):
        self.temporary_class_id = None
        self.unique_classes = None  # much like a bit hacking that this is quite dynamic
        self.device_id = device_id
        self.model = model
        self.tau = tau
        self.beta = beta
        self.use_cpu = use_cpu
        # print(f"device of self model: {self.model.device}")

    @staticmethod
    def update_perturbation_multiplier(beta, tau, iou):
        return beta * iou + tau

    @staticmethod
    def calculate_l2_loss(x, y):
        loss = (x - y) ** 2
        for a in reversed(range(1, loss.dim())):
            loss = loss.sum(a, keepdim=False)
        loss = loss.sum()
        return loss

    def calculate_pred_loss(self, target_mask, pred_out, model_output, target_class: typing.Iterable[int] = None):
        loss = 0
        if self.unique_classes is None:
            print(f"attack object has unique classes not set (currently None), returning from calculating loss")
            return None
        # iterating_class = self.unique_classes
        if target_class is None:
            iterating_class = self.unique_classes
        else:
            iterating_class = target_class
        print(f">>> calculating pred loss for iterating classes {iterating_class}")
        for single_class in iterating_class:
            # Calculating Equation 2 in the paper

            # g(\theta, \mathbf{X}_n)_c:
            out_channel = model_output[0][single_class]

            # \mathds{1}_{\{\mathbf{Y}^{A} \, =\, c\}}:
            optimization_mask = copy.deepcopy(target_mask)
            optimization_mask[optimization_mask != single_class] = self.temporary_class_id
            optimization_mask[optimization_mask == single_class] = 1
            optimization_mask[optimization_mask == self.temporary_class_id] = 0

            # \mathds{1}_{\{\arg \max_M(g(\theta, \mathbf{X}_n)) \, \neq \, c\}}:
            prediction_mask = copy.deepcopy(pred_out)[0]
            prediction_mask[prediction_mask != single_class] = self.temporary_class_id
            prediction_mask[prediction_mask == single_class] = 0
            prediction_mask[prediction_mask == self.temporary_class_id] = 1

            # Calculate channel loss
            channel_loss = torch.sum(out_channel * optimization_mask * prediction_mask)
            # Add to total loss
            loss = loss + channel_loss
        return loss

    def perform_attack(self,
                       input_image: torch.Tensor,
                       org_mask: torch.Tensor,
                       target_mask: torch.Tensor,
                       unique_class_list: list | set,
                       total_iter=2501,
                       save_samples=True,
                       save_path='/home/peizhu/PycharmProjects/adaptive-segmentation-mask-attack/adv_results'
                                 '/cityscape_results/',
                       verbose=True):
        device = "cpu" if self.use_cpu else self.device_id
        if len(unique_class_list) == 1:
            print(f">>> perform single class attack on device {device} and target class {unique_class_list}")
        elif len(unique_class_list) > 1:
            print(f">>> begin to perform attack on device {device} with unique class list {unique_class_list}.")
        else:
            raise ValueError("in perform attack, unique class list is empty")
        target_classes = unique_class_list.copy()
        try:
            target_classes.remove(0)
        except ValueError:
            pass
        print(f"DEBUG: type of org mask: {type(org_mask)}, size of origin mask: {org_mask.numpy().shape}")
        print(f"DEBUG: type of tar mask: {type(target_mask)}, size of origin mask: {target_mask.numpy().shape}")

        if save_samples:
            # Save masks
            print(f"> saving masks to location {save_path}")
            # TODO: change to dataset specific loading
            # save_image(org_mask.numpy(), 'original_mask', save_path)
            # save_image(target_mask.numpy(), 'target_mask', save_path)
            dec_org_mask = CityscapeDataset.decode_target(org_mask.numpy())
            dec_target_mask = CityscapeDataset.decode_target(target_mask.numpy())
            print(f"saving masks. mask size: {dec_org_mask.shape}")
            save_image(dec_org_mask, 'original_mask', save_path)
            save_image(dec_target_mask, 'target_mask', save_path)

        # Unique classes are needed to simplify prediction loss
        self.unique_classes = unique_class_list
        # Have a look at calculate_pred_loss to see where this is used
        # select an out-of-sample distribution class
        self.temporary_class_id = random.randint(0, 999)
        while self.temporary_class_id in self.unique_classes:
            self.temporary_class_id = random.randint(0, 999)

        # Assume there is no overlapping part for the first iteration
        pert_mul = self.update_perturbation_multiplier(self.beta, self.tau, 0)
        # Get a copy of target mask to use it for stats
        target_mask_numpy = copy.deepcopy(target_mask).numpy()
        # Target mask
        target_mask = target_mask.float()
        target_mask = target_mask.cpu()
        if not self.use_cpu:
            target_mask = target_mask.cuda(self.device_id)

        # Image to perform the attack on
        image_to_optimize: torch.Tensor
        image_to_optimize = input_image.unsqueeze(0)
        # Copied version of image for l2 dist
        org_im_copy = copy.deepcopy(image_to_optimize.cpu())
        if not self.use_cpu:
            org_im_copy = org_im_copy.cuda(self.device_id)
        printed_size = False
        for single_iter in range(total_iter):
            num_garbage = gc.collect()
            print(f">>> collected garbage number {num_garbage}")
            # Put in variable to get grads later on
            # Variable deprecated in later torch: they return tensors instead, and autograd = true as default
            if not self.use_cpu:
                image_to_optimize = Variable(image_to_optimize.cuda(self.device_id), requires_grad=True)
            else:
                image_to_optimize = Variable(image_to_optimize.cpu(), requires_grad=True)
            # image_to_optimize = image_to_optimize.cuda(self.device_id)
            # print(image_to_optimize.device)

            # Forward pass
            out: torch.Tensor
            out = self.model(image_to_optimize)
            if not printed_size:
                print(f"size of model output: {out.shape}")
                printed_size = True
            # Prediction
            pred_out = torch.argmax(out, dim=1).float()

            # L2 Loss
            l2_loss = self.calculate_l2_loss(org_im_copy, image_to_optimize)
            # Prediction loss
            pred_loss = self.calculate_pred_loss(target_mask, pred_out, out, target_class=target_classes)
            # Total loss
            out_grad = torch.sum(pred_loss - l2_loss)
            print(f"OOO out grad dimension: {out_grad.shape}")
            print(f"OOO out grad loss: {out_grad}")
            # Backward pass
            out_grad.backward()

            # Add perturbation to image to optimize
            # print(f">>> type of image to optimize: {type(image_to_optimize)} "
            #       f"and its data type {type(image_to_optimize.data)}")
            # print(f"its grad type: {type(image_to_optimize.grad)} and pert_mul type {type(pert_mul)}")
            perturbed_im: torch.Tensor
            print(f">>> checking gradient of image to optimize:")
            int_grad: torch.Tensor
            int_grad = image_to_optimize.grad
            int_np_grad = int_grad.numpy()
            print(f"max of grad: {np.amax(int_np_grad)}")
            print(f"min of grad: {np.amin(int_np_grad)}")
            perturbed_im = image_to_optimize.data + (image_to_optimize.grad * pert_mul)
            # Do another forward pass to calculate new pert_mul
            perturbed_im_out = self.model(perturbed_im)

            # Discretize perturbed image to calculate stats
            perturbed_im_pred: torch.Tensor
            perturbed_im_pred = torch.argmax(perturbed_im_out, dim=1).float()[0]
            perturbed_im_pred = perturbed_im_pred.detach().cpu().numpy()

            # Calculate performance of the attack
            # Similarities
            # iou, pixel_acc = calculate_multiclass_mask_similarity(perturbed_im_pred, target_mask_numpy,
            #                                                       target_classes=self.unique_classes)
            iou, pixel_acc = calculate_multiclass_mask_similarity(perturbed_im_pred,
                                                                  target_mask_numpy,
                                                                  target_classes=target_classes)

            # iou, pixel_acc = calculate_multiclass_mask_similarity(perturbed_im_pred, target_mask_numpy,)

            # Distances
            l2_dist, linf_dist = calculate_image_distance(org_im_copy, perturbed_im)
            # Update perturbation multiplier
            pert_mul = self.update_perturbation_multiplier(self.beta, self.tau, iou)

            # Update image to optimize and ensure boxt constraint
            print(f">>> checking type of perturbed im and im to op")
            intermediate_tensor = perturbed_im.data
            """
            is this clamp(0,1) really not problemistic for multi class segmentation?
            """
            # image_to_optimize = perturbed_im.data.clamp_(0, 1)
            # image_to_optimize = perturbed_im.data.clamp(0, 255)
            image_to_optimize = copy.deepcopy(perturbed_im)
            # print(type(perturbed_im))
            # print(type(image_to_optimize))
            # print(f"checking type of pert_im.data")
            # print(type(perturbed_im.data))
            print(f">>> checking image to optimize integrity")
            debug_img: np.ndarray
            debug_img = image_to_optimize[0].numpy()
            debug_img = debug_img.transpose((1, 2, 0))
            # print(debug_img)
            # print(debug_img.shape)
            print(np.amax(debug_img))
            print(np.amin(debug_img))
            if single_iter % 2 == 0:
                if save_samples:
                    # print(f">>> saving images sample to location: {save_path}")
                    # pred_out_np = pred_out.cpu().detach().numpy()[0]
                    pred_out_np: np.ndarray
                    pred_out_np = pred_out.cpu().detach().numpy()[0]
                    pred_out_np = pred_out_np.astype('uint8')
                    # pred_out_np = pred_out_np.transpose([1, 2, 0])
                    pred_out_np = CityscapeDataset.decode_target(pred_out_np)
                    # print(pred_out_np.shape)
                    # print(pred_out_np)
                    save_image(pred_out_np, 'iter_' + str(single_iter),
                               save_path + 'prediction')
                    save_batch_image(image_to_optimize.data.cpu().detach().numpy(), 'iter_' + str(single_iter),
                                     save_path + 'modified_image')
                    save_batch_image_difference(image_to_optimize.data.cpu().detach().numpy(),
                                                org_im_copy.data.cpu().detach().numpy(),
                                                'iter_' + str(single_iter), save_path + 'added_perturbation')
                if verbose:
                    print('Iter:', single_iter, '\tIOU Overlap:', iou,
                          '\tPixel Accuracy:', pixel_acc,
                          '\n\t\tL2 Dist:', l2_dist,
                          '\tL_inf dist:', linf_dist)

    # def perform_single_class_attack(self,
    #                                 input_image: torch.Tensor,
    #                                 org_mask: torch.Tensor,
    #                                 target_mask: torch.Tensor,
    #                                 unique_class_list,
    #                                 total_iter=2501,
    #                                 save_samples=True,
    #                                 save_path='/home/peizhu/PycharmProjects/adaptive-segmentation-mask'
    #                                           '-attack/adv_results '
    #                                           '/cityscape_results/single_class/',
    #                                 verbose=True):
    #     device = "cpu" if self.use_cpu else self.device_id
    #     print(f">>> begin to perform single class attack on device {device} with unique class list "
    #           f"{unique_class_list}.")
    #     # print(f"DEBUG: type of org mask: {type(org_mask)}, size of origin mask: {org_mask.numpy().shape}")
    #     # print(f"DEBUG: type of tar mask: {type(target_mask)}, size of origin mask: {target_mask.numpy().shape}")
    #
    #     if save_samples:
    #         # Save masks
    #         print(f"> saving masks to location {save_path}")
    #         # save_image(org_mask.numpy(), 'original_mask', save_path)
    #         # save_image(target_mask.numpy(), 'target_mask', save_path)
    #         dec_org_mask = CityscapeDataset.decode_target(org_mask.numpy())
    #         dec_target_mask = CityscapeDataset.decode_target(target_mask.numpy())
    #         print(f"saving masks. mask size: {dec_org_mask.shape}")
    #         save_image(dec_org_mask, 'original_mask', save_path)
    #         save_image(dec_target_mask, 'target_mask', save_path)
    #
    #     # a single class to attack
    #     if len(unique_class_list) < 2:
    #         self.unique_classes = unique_class_list
    #     else:
    #         raise ValueError(f"in single class attack, received number of classes {len(unique_class_list)} "
    #                          f"instead of a single class")
    #     # Have a look at calculate_pred_loss to see where this is used
    #     # select a out-of-sample distribution class
    #     self.temporary_class_id = random.randint(0, 999)
    #     while self.temporary_class_id in self.unique_classes:
    #         self.temporary_class_id = random.randint(0, 999)
    #
    #     # Assume there is no overlapping part for the first iteration
    #     pert_mul = self.update_perturbation_multiplier(self.beta, self.tau, 0)
    #     # Get a copy of target mask to use it for stats
    #     target_mask_numpy = copy.deepcopy(target_mask).numpy()
    #     # Target mask
    #     target_mask = target_mask.float()
    #     target_mask = target_mask.cpu()
    #     if not self.use_cpu:
    #         target_mask = target_mask.cuda(self.device_id)
    #
    #     # Image to perform the attack on
    #     image_to_optimize: torch.Tensor
    #     image_to_optimize = input_image.unsqueeze(0)
    #     # Copied version of image for l2 dist
    #     org_im_copy = copy.deepcopy(image_to_optimize.cpu())
    #     if not self.use_cpu:
    #         org_im_copy = org_im_copy.cuda(self.device_id)
    #     printed_size = False
    #     for single_iter in range(total_iter):
    #         # Put in variable to get grads later on
    #         # Variable deprecated in later torch: they return tensors instead, and autograd = true as default
    #         if not self.use_cpu:
    #             image_to_optimize = Variable(image_to_optimize.cuda(self.device_id), requires_grad=True)
    #         else:
    #             image_to_optimize = Variable(image_to_optimize.cpu(), requires_grad=True)
    #         # image_to_optimize = image_to_optimize.cuda(self.device_id)
    #
    #         # Forward pass
    #         out: torch.Tensor
    #         out = self.model(image_to_optimize)
    #         pred_out = torch.argmax(out, dim=1).float()
    #         if not printed_size:
    #             # [1,19,1024,2048]
    #             # [1,1024,2048]
    #             print(f"size of model output: {out.shape}")
    #             print(f"size of pred_out: {pred_out.shape}")
    #             printed_size = True
    #         # Prediction
    #
    #         # L2 Loss
    #         l2_loss = self.calculate_l2_loss(org_im_copy, image_to_optimize)
    #         # Prediction loss
    #         pred_loss = self.calculate_pred_loss(target_mask, pred_out, out)
    #         print(f"l2 loss: {l2_loss}")
    #         print(f"pred_loss: {pred_loss}")
    #         # Total loss
    #         out_grad = torch.sum(pred_loss - l2_loss)
    #         # Backward pass
    #         out_grad.backward()
    #
    #         # Add perturbation to image to optimize
    #         # print(f">>> type of image to optimize: {type(image_to_optimize)} "
    #         #       f"and its data type {type(image_to_optimize.data)}")
    #         # print(f"its grad type: {type(image_to_optimize.grad)} and pert_mul type {type(pert_mul)}")
    #         perturbed_im: torch.Tensor
    #         print(f">>> checking gradient of image to optimize:")
    #         int_grad: torch.Tensor
    #         int_grad = image_to_optimize.grad
    #         int_np_grad = int_grad.numpy()
    #         print(f"max of grad: {np.amax(int_np_grad)}")
    #         print(f"min of grad: {np.amin(int_np_grad)}")
    #         perturbed_im = image_to_optimize.data + (image_to_optimize.grad * pert_mul)
    #         # Do another forward pass to calculate new pert_mul
    #         perturbed_im_out = self.model(perturbed_im)
    #
    #         # Discretize perturbed image to calculate stats
    #         perturbed_im_pred: torch.Tensor
    #         perturbed_im_pred = torch.argmax(perturbed_im_out, dim=1).float()[0]
    #         perturbed_im_pred = perturbed_im_pred.detach().cpu().numpy()
    #
    #         # Calculate performance of the attack
    #         # Similarities
    #         iou, pixel_acc = calculate_multiclass_mask_similarity(perturbed_im_pred, target_mask_numpy,
    #                                                               self.unique_classes)
    #         # Distances
    #         l2_dist, linf_dist = calculate_image_distance(org_im_copy, perturbed_im)
    #         # Update perturbation multiplier
    #         pert_mul = self.update_perturbation_multiplier(self.beta, self.tau, iou)
    #
    #         # Update image to optimize and ensure boxt constraint
    #         print(f">>> checking type of perturbed im and im to op")
    #         intermediate_tensor = perturbed_im.data
    #         """
    #         is this clamp(0,1) really not problemistic for multi class segmentation?
    #         """
    #         # image_to_optimize = perturbed_im.data.clamp_(0, 1)
    #         image_to_optimize = perturbed_im.data.clamp(0, 255)
    #         print(type(perturbed_im))
    #         print(type(image_to_optimize))
    #         print(f"checking type of pert_im.data")
    #         print(type(perturbed_im.data))
    #         # print(f">>> checking image to optimize integrity")
    #         # debug_img: np.ndarray
    #         # debug_img = image_to_optimize[0].numpy()
    #         # debug_img = debug_img.transpose((1, 2, 0))
    #         # print(debug_img)
    #         # print(debug_img.shape)
    #         # print(np.amax(debug_img))
    #         # print(np.amin(debug_img))
    #         if single_iter % 2 == 0:
    #             if save_samples:
    #                 # print(f">>> saving images sample to location: {save_path}")
    #                 # pred_out_np = pred_out.cpu().detach().numpy()[0]
    #                 pred_out_np: np.ndarray
    #                 pred_out_np = pred_out.cpu().detach().numpy()[0]
    #                 pred_out_np = pred_out_np.astype('uint8')
    #                 # pred_out_np = pred_out_np.transpose([1, 2, 0])
    #                 pred_out_np = CityscapeDataset.decode_target(pred_out_np)
    #                 print(pred_out_np.shape)
    #                 # print(pred_out_np)
    #                 save_image(pred_out_np, 'iter_' + str(single_iter),
    #                            save_path + 'prediction')
    #                 save_batch_image(image_to_optimize.data.cpu().detach().numpy(), 'iter_' + str(single_iter),
    #                                  save_path + 'modified_image')
    #                 save_batch_image_difference(image_to_optimize.data.cpu().detach().numpy(),
    #                                             org_im_copy.data.cpu().detach().numpy(),
    #                                             'iter_' + str(single_iter), save_path + 'added_perturbation')
    #             if verbose:
    #                 print('Iter:', single_iter, '\tIOU Overlap:', iou,
    #                       '\tPixel Accuracy:', pixel_acc,
    #                       '\n\t\tL2 Dist:', l2_dist,
    #                       '\tL_inf dist:', linf_dist)
