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
    def calculate_loss_facade(x, y, metric: str):
        metric2method_mapp = {"l1": AdaptiveSegmentationMaskAttack.calculate_l1_loss,
                              "l2": AdaptiveSegmentationMaskAttack.calculate_l2_loss,
                              "linf": AdaptiveSegmentationMaskAttack.calculate_linf_loss}
        mtr = metric.lower()
        if mtr not in metric2method_mapp:
            raise NotImplementedError(f"specified metric {metric} not implemented. "
                                      f"valid set: {metric2method_mapp.keys()}")
        else:
            return metric2method_mapp[mtr](x, y)
    @staticmethod
    def calculate_linf_loss(x, y):
        diff = (x - y)
        if isinstance(diff, torch.Tensor):
            diff = diff.numpy()
        if not isinstance(diff, np.ndarray):
            raise RuntimeError(f"diff is supposed to be nparr but get {type(diff)}")
        max_loss = np.abs(np.amax(diff))
        return max_loss

    @staticmethod
    def calculate_l2_loss(x, y):
        loss = (x - y) ** 2
        # print(loss)
        # print(f"min of loss: {np.amin(loss)}, max of loss: {np.amax(loss)}")
        for a in reversed(range(1, loss.dim())):
            loss = loss.sum(a, keepdim=False)
        loss = loss.sum()
        return loss

    @staticmethod
    def calculate_l1_loss(x, y):
        loss = torch.abs(x - y)
        # print(loss)
        # print(f"min of loss: {np.amin(loss)}, max of loss: {np.amax(loss)}")
        for a in reversed(range(1, loss.dim())):
            loss = loss.sum(a, keepdim=False)
        loss = loss.sum()
        return loss

    def calculate_target_pred_loss(self, target_mask, pred_out, model_output,
                                   target_class: typing.Iterable[int] = None):
        # pred_loss = self.calculate_target_pred_loss(target_mask, pred_out, out, target_class=target_classes)
        loss = 0
        if self.unique_classes is None:
            print(f"attack object has unique classes not set (currently None), returning from calculating loss")
            return None
        # iterating_class = self.unique_classes
        if target_class is None:
            iterating_class = self.unique_classes
        else:
            iterating_class = target_class
        # print(f">>> calculating pred loss for iterating classes {iterating_class}")
        for single_class in iterating_class:
            # Calculating Equation 2 in the paper

            # get specific out channel pred
            out_channel = model_output[0][single_class]

            # set all pixels with target class to 1 others to 0
            optimization_mask = copy.deepcopy(target_mask)
            optimization_mask[optimization_mask != single_class] = self.temporary_class_id
            optimization_mask[optimization_mask == single_class] = 1
            optimization_mask[optimization_mask == self.temporary_class_id] = 0

            # same process inversed for pred
            prediction_mask = copy.deepcopy(pred_out)[0]
            prediction_mask[prediction_mask != single_class] = self.temporary_class_id
            prediction_mask[prediction_mask == single_class] = 0
            prediction_mask[prediction_mask == self.temporary_class_id] = 1

            # Calculate channel loss
            channel_loss = torch.sum(out_channel * optimization_mask * prediction_mask)
            # Add to total loss
            loss = loss + channel_loss
        return loss

    def calculate_untargeted_pred_loss(self, pred_out, model_output,
                                       original_class: typing.Iterable[int] = None,
                                       target_class: typing.Iterable[int] = None):
        loss = 0
        if self.unique_classes is None:
            print(f"attack object has unique classes not set (currently None), returning from calculating loss")
            return None
        # iterating_class = self.unique_classes
        if original_class is None:
            iterating_class = self.unique_classes
        else:
            iterating_class = original_class
        # print(f">>> calculating untargeted pred loss for iterating classes {iterating_class}")
        for single_class in iterating_class:
            # directly suppress target classes

            # get specific out channel pred
            out_channel = model_output[0][single_class]
            prediction_mask = copy.deepcopy(pred_out)[0]
            # if target_class is None:
            prediction_mask[prediction_mask != single_class] = self.temporary_class_id
            prediction_mask[prediction_mask == single_class] = 1
            prediction_mask[prediction_mask == self.temporary_class_id] = 0

            # Calculate channel loss
            channel_loss = torch.sum(out_channel * prediction_mask)
            # Add to total loss
            loss = loss + channel_loss
        return loss

    def perform_attack(self,
                       input_image: torch.Tensor,
                       org_mask: torch.Tensor,
                       target_mask: torch.Tensor | None,
                       *,
                       loss_metric: str = "l2",
                       unique_class_list: list | set,
                       total_iter=2501,
                       save_samples=True,
                       save_path='/home/peizhu/PycharmProjects/adaptive-segmentation-mask-attack/adv_results/cityscape_results/',
                       verbose=True):
        print(f">>> performing attack on save path {save_path}.")

        def verbose_print(s):
            if verbose:
                print(s)

        device = "cpu" if self.use_cpu else self.device_id
        if len(unique_class_list) == 1:
            verbose_print(f">>> perform single class attack on device {device} and target class {unique_class_list}")
        elif len(unique_class_list) > 1:
            verbose_print(f">>> begin to perform attack on device {device} with unique class list {unique_class_list}.")
        else:
            raise ValueError("in perform attack, unique class list is empty")
        target_classes = unique_class_list.copy()
        # try:
        #     target_classes.remove(0)
        # except ValueError:
        #     pass
        verbose_print(f"DEBUG: type of org mask: {type(org_mask)}, size of origin mask: {org_mask.numpy().shape}, "
                      f"unique of org mask: {np.unique(org_mask.numpy())}")
        verbose_print(f"DEBUG: type of tar mask: {type(target_mask)}")

        if save_samples:
            # Save masks
            verbose_print(f"> saving masks to location {save_path}")
            # TODO: change to dataset specific loading
            # save_image(org_mask.numpy(), 'original_mask', save_path)
            # save_image(target_mask.numpy(), 'target_mask', save_path)
            dec_org_mask = CityscapeDataset.decode_target(org_mask.numpy())
            verbose_print(f"saving masks. mask size: {dec_org_mask.shape}")
            save_image(dec_org_mask, 'original_mask', save_path)
            if target_mask is not None:
                dec_target_mask = CityscapeDataset.decode_target(target_mask.numpy())

                save_image(dec_target_mask, 'target_mask', save_path)

        # Unique classes are needed to simplify prediction loss
        self.unique_classes = unique_class_list
        # Have a look at calculate_pred_loss to see where this is used
        # select an out-of-sample distribution class
        self.temporary_class_id = random.randint(0, 999)
        while self.temporary_class_id in self.unique_classes or self.temporary_class_id in range(50):
            self.temporary_class_id = random.randint(0, 999)

        # Assume there is no overlapping part for the first iteration
        pert_mul = self.update_perturbation_multiplier(self.beta, self.tau, 0)
        if target_mask is not None:
            # Get a copy of target mask to use it for stats
            target_mask_numpy = copy.deepcopy(target_mask).numpy()
            # Target mask
            target_mask = target_mask.float()
            target_mask = target_mask.cpu()
            if not self.use_cpu:
                target_mask = target_mask.cuda(self.device_id)
        else:
            target_mask_numpy = None

        # Image to perform the attack on
        image_to_optimize: torch.Tensor
        image_to_optimize = input_image.unsqueeze(0)
        print("VVVV max of im to optimize: ", np.amax(image_to_optimize.numpy()), "shape: ", image_to_optimize.shape)
        # add a bit gaussian noise for stabilization
        noise_variance = 1e-8
        image_to_optimize = image_to_optimize + (noise_variance ** 0.5) * torch.randn(image_to_optimize.shape)
        print("VVVV max of im to optimize: ", np.amax(image_to_optimize.numpy()), "shape: ", image_to_optimize.shape)
        # Copied version of image for l2 dist
        org_im_copy = copy.deepcopy(input_image.cpu())
        if not self.use_cpu:
            org_im_copy = org_im_copy.cuda(self.device_id)
        verbose_printed_size = False
        for single_iter in range(total_iter):
            num_garbage = gc.collect()
            verbose_print(f">>> collected garbage number {num_garbage}")
            # Put in variable to get grads later on
            # Variable deprecated in later torch: they return tensors instead, and autograd = true as default
            if not self.use_cpu:
                image_to_optimize = Variable(image_to_optimize.cuda(self.device_id), requires_grad=True)
            else:
                image_to_optimize = Variable(image_to_optimize.cpu(), requires_grad=True)
            # image_to_optimize = image_to_optimize.cuda(self.device_id)
            # verbose_print(image_to_optimize.device)

            # Forward pass
            out: torch.Tensor
            out = self.model(image_to_optimize)
            if not verbose_printed_size:
                verbose_print(f"size of model output: {out.shape}")
                verbose_printed_size = True
            # Prediction
            pred_out = torch.argmax(out, dim=1).float()

            # dist Loss
            dist_loss = self.calculate_loss_facade(org_im_copy, image_to_optimize, loss_metric)
            print(dist_loss)
            if target_mask is not None:
                # Prediction loss
                pred_loss = self.calculate_target_pred_loss(target_mask, pred_out, out, target_class=target_classes)
            else:
                pred_loss = self.calculate_untargeted_pred_loss(pred_out, out, original_class=target_classes)
            # Total loss
            pred_loss_weight = 1
            dist_loss_weight = 4
            if target_mask is not None:
                out_grad = torch.sum(pred_loss_weight * pred_loss - dist_loss_weight * dist_loss)
            else:
                out_grad = torch.sum(- pred_loss_weight * pred_loss - dist_loss_weight * dist_loss)

            # verbose_print(f"OOO out grad dimension: {out_grad.shape}")
            verbose_print(f"OOO out grad : {out_grad}")
            verbose_print(f"OOO out L-x loss: {dist_loss}")
            verbose_print(f"OOO out pred loss: {pred_loss}")
            # Backward pass
            out_grad.backward()

            # Add perturbation to image to optimize
            # verbose_print(f">>> type of image to optimize: {type(image_to_optimize)} "
            #       f"and its data type {type(image_to_optimize.data)}")
            # verbose_print(f"its grad type: {type(image_to_optimize.grad)} and pert_mul type {type(pert_mul)}")
            perturbed_im: torch.Tensor
            verbose_print(f">>> checking gradient of image to optimize:")
            int_grad: torch.Tensor
            int_grad = image_to_optimize.grad
            int_np_grad = int_grad.numpy()
            verbose_print(f"max of grad: {np.amax(int_np_grad)}")
            verbose_print(f"min of grad: {np.amin(int_np_grad)}")
            perturbed_im = image_to_optimize.data + (image_to_optimize.grad * pert_mul)
            # Do another forward pass to calculate new pert_mul
            perturbed_im_out = self.model(perturbed_im)

            # Discretize perturbed image to calculate stats
            perturbed_im_pred_ts: torch.Tensor
            perturbed_im_pred: np.ndarray
            perturbed_im_pred_ts = torch.argmax(perturbed_im_out, dim=1).float()[0]
            perturbed_im_pred = perturbed_im_pred_ts.detach().cpu().numpy()

            # Calculate performance of the attack
            # Similarities
            # iou, pixel_acc = calculate_multiclass_mask_similarity(perturbed_im_pred, target_mask_numpy,
            #                                                       target_classes=self.unique_classes)
            if target_mask is not None:
                iou, pixel_acc = calculate_multiclass_mask_similarity(perturbed_im_pred,
                                                                      target_mask_numpy,
                                                                      target_classes=target_classes)
            else:
                acc_iou = 0
                for elem in target_classes:
                    verbose_print(f"calculating untargeted iou for class {elem}")
                    temp_mask = copy.deepcopy(org_mask).numpy()
                    temp_mask[temp_mask != elem] = self.temporary_class_id
                    temp_mask[temp_mask == elem] = 1
                    temp_mask[temp_mask == self.temporary_class_id] = 0

                    temp_pred_mask = copy.deepcopy(perturbed_im_pred)
                    temp_pred_mask = temp_pred_mask.astype('uint8')
                    # print(np.unique(temp_pred_mask))
                    temp_pred_mask[temp_pred_mask == elem] = self.temporary_class_id
                    temp_pred_mask[temp_pred_mask != self.temporary_class_id] = 1
                    temp_pred_mask[temp_pred_mask == self.temporary_class_id] = 0
                    attacked = np.sum(temp_pred_mask * temp_mask)
                    original_total = np.sum(temp_mask)
                    # print(temp_pred_mask.shape, temp_pred_mask.shape[0] * temp_pred_mask.shape[1])
                    # print(np.sum(temp_pred_mask))
                    # print(attacked, original_total)
                    acc_iou += attacked / original_total
                iou = acc_iou / len(target_classes)
                pixel_acc = "untargeted"

            # iou, pixel_acc = calculate_multiclass_mask_similarity(perturbed_im_pred, target_mask_numpy,)

            # Distances
            l2_dist, linf_dist = calculate_image_distance(org_im_copy, perturbed_im)
            # Update perturbation multiplier
            pert_mul = self.update_perturbation_multiplier(self.beta, self.tau, iou)

            # Update image to optimize and ensure boxt constraint
            verbose_print(f">>> checking type of perturbed im and im to op")
            intermediate_tensor = perturbed_im.data
            """
            is this clamp(0,1) really not problemistic for multi class segmentation?
            """
            # image_to_optimize = perturbed_im.data.clamp_(0, 1)
            # image_to_optimize = perturbed_im.data.clamp(0, 255)
            image_to_optimize = copy.deepcopy(perturbed_im)
            # verbose_print(type(perturbed_im))
            # verbose_print(type(image_to_optimize))
            # verbose_print(f"checking type of pert_im.data")
            # verbose_print(type(perturbed_im.data))
            verbose_print(f">>> checking image to optimize integrity")
            debug_img: np.ndarray
            debug_img = image_to_optimize[0].numpy()
            debug_img = debug_img.transpose((1, 2, 0))
            # verbose_print(debug_img)
            # verbose_print(debug_img.shape)
            verbose_print(np.amax(debug_img))
            verbose_print(np.amin(debug_img))
            if single_iter % 5 == 0:
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
                               save_path + 'prediction', normalize=False)
                    # is this copy really necessary for normalization?
                    unnormalized_imtoptimize: torch.Tensor
                    unnormalized_imtoptimize = copy.deepcopy(image_to_optimize.data.cpu().detach())

                    # db = unnormalized_imtoptimize.numpy()
                    # print(np.amax(db), np.amin(db))

                    normalized_imtoptimize = CityscapeDataset.train_image_to_rgb(unnormalized_imtoptimize)
                    # normalized_imtoptimize = normalized_imtoptimize.clamp(0, 1)
                    # normalized_imtoptimize = unnormalized_imtoptimize

                    # imoc: torch.Tensor
                    # imoc = copy.deepcopy(normalized_imtoptimize)
                    # imoc = torch.squeeze(imoc)
                    # db = imoc.numpy()
                    # print(db.shape)
                    # db *= 254
                    # db = db.transpose((1, 2, 0))
                    # save_image(db, "mod_image_test", save_path, normalize=False)
                    # print(np.amax(db), np.amin(db))

                    save_batch_image(normalized_imtoptimize.numpy(), 'iter_' + str(single_iter),
                                     save_path + 'modified_image', normalize=True)
                    save_batch_image_difference(image_to_optimize.data.cpu().detach().numpy(),
                                                org_im_copy.data.cpu().detach().numpy(),
                                                'iter_' + str(single_iter), save_path + 'added_perturbation',
                                                normalize=True
                                                )
                print('Iter:', single_iter, '\tIOU Overlap:', iou,
                      '\tPixel Accuracy:', pixel_acc,
                      '\n\t\tL2 Dist:', l2_dist,
                      '\tL_inf dist:', linf_dist)
