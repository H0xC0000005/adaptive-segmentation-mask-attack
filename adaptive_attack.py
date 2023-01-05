"""
@author: Utku Ozbulak
@repository: github.com/utkuozbulak/adaptive-segmentation-mask-attack
@article: Impact of Adversarial Examples on Deep Learning Models for Biomedical Image Segmentation
@conference: MICCAI-19
"""
import copy
import random
# Torch imports
import torch
from torch.autograd import Variable

# from torch.autograd import Variable
# In-repo imports
from helper_functions import (save_prediction_image,
                              save_input_image,
                              save_image_difference,
                              calculate_mask_similarity,
                              calculate_image_distance, save_image)
random.seed(4)


class AdaptiveSegmentationMaskAttack:
    def __init__(self, device_id, model, tau, beta):
        self.temporary_class_id = None
        self.unique_classes = None  # much like a bit hacking that this is quite dynamic
        self.device_id = device_id
        self.model = model
        self.tau = tau
        self.beta = beta
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

    def calculate_pred_loss(self, target_mask, pred_out, model_output):
        loss = 0
        if self.unique_classes is None:
            print(f"attack object has unique classes not set (currently None), returning from calculating loss")
            return None
        for single_class in self.unique_classes:
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

    def perform_attack(self, input_image, org_mask: torch.Tensor, target_mask: torch.Tensor, unique_class_list,
                       total_iter=2501, save_samples=True,
                       save_path='/home/peizhu/PycharmProjects/adaptive-segmentation-mask-attack/adv_results'
                                 '/cityscape_results/',
                       verbose=True):
        print(f">>> begin to perform attack.")
        print(f"DEBUG: type of org mask: {type(org_mask)}, size of origin mask: {org_mask.numpy().shape}")
        print(f"DEBUG: type of tar mask: {type(target_mask)}, size of origin mask: {target_mask.numpy().shape}")

        if save_samples:
            # Save masks
            print(f"> saving masks to location {save_path}")
            save_image(org_mask.numpy(), 'original_mask', save_path)
            save_image(target_mask.numpy(), 'target_mask', save_path)
        # Unique classes are needed to simplify prediction loss
        self.unique_classes = unique_class_list
        # Have a look at calculate_pred_loss to see where this is used
        # select a out-of-sample distribution class
        self.temporary_class_id = random.randint(0, 999)
        while self.temporary_class_id in self.unique_classes:
            self.temporary_class_id = random.randint(0, 999)

        # Assume there is no overlapping part for the first iteration
        pert_mul = self.update_perturbation_multiplier(self.beta, self.tau, 0)
        # Get a copy of target mask to use it for stats
        target_mask_numpy = copy.deepcopy(target_mask).numpy()
        # Target mask
        target_mask = target_mask.float().cuda(self.device_id)

        # Image to perform the attack on
        image_to_optimize = input_image.unsqueeze(0)
        # Copied version of image for l2 dist
        org_im_copy = copy.deepcopy(image_to_optimize.cpu()).cuda(self.device_id)
        for single_iter in range(total_iter):
            # Put in variable to get grads later on
            # Variable deprecated in later torch: they return tensors instead, and autograd = true as default
            image_to_optimize = Variable(image_to_optimize.cuda(self.device_id), requires_grad=True)
            # image_to_optimize = image_to_optimize.cuda(self.device_id)

            # Forward pass
            out = self.model(image_to_optimize)
            # Prediction
            pred_out = torch.argmax(out, dim=1).float()

            # L2 Loss
            l2_loss = self.calculate_l2_loss(org_im_copy, image_to_optimize)
            # Prediction loss
            pred_loss = self.calculate_pred_loss(target_mask, pred_out, out)
            # Total loss
            out_grad = torch.sum(pred_loss - l2_loss)
            # Backward pass
            out_grad.backward()

            # Add perturbation to image to optimize
            # print(f">>> type of image to optimize: {type(image_to_optimize)} "
            #       f"and its data type {type(image_to_optimize.data)}")
            # print(f"its grad type: {type(image_to_optimize.grad)} and pert_mul type {type(pert_mul)}")
            perturbed_im = image_to_optimize.data + (image_to_optimize.grad * pert_mul)
            # Do another forward pass to calculate new pert_mul
            perturbed_im_out = self.model(perturbed_im)

            # Discretize perturbed image to calculate stats
            perturbed_im_pred: torch.Tensor
            perturbed_im_pred = torch.argmax(perturbed_im_out, dim=1).float()[0]
            perturbed_im_pred = perturbed_im_pred.detach().cpu().numpy()

            # Calculate performance of the attack
            # Similarities
            iou, pixel_acc = calculate_mask_similarity(perturbed_im_pred, target_mask_numpy)
            # Distances
            l2_dist, linf_dist = calculate_image_distance(org_im_copy, perturbed_im)
            # Update perturbation multiplier
            pert_mul = self.update_perturbation_multiplier(self.beta, self.tau, iou)

            # Update image to optimize and ensure boxt constraint
            image_to_optimize = perturbed_im.data.clamp_(0, 1)
            if single_iter % 50 == 0:
                if save_samples:
                    # print(f">>> saving images sample to location: {save_path}")
                    save_prediction_image(pred_out.cpu().detach().numpy()[0], 'iter_' + str(single_iter),
                                          save_path + 'prediction')
                    save_input_image(image_to_optimize.data.cpu().detach().numpy(), 'iter_' + str(single_iter),
                                     save_path + 'modified_image')
                    save_image_difference(image_to_optimize.data.cpu().detach().numpy(),
                                          org_im_copy.data.cpu().detach().numpy(),
                                          'iter_' + str(single_iter), save_path + 'added_perturbation')
                if verbose:
                    print('Iter:', single_iter, '\tIOU Overlap:', iou,
                          '\tPixel Accuracy:', pixel_acc,
                          '\n\t\tL2 Dist:', l2_dist,
                          '\tL_inf dist:', linf_dist)
