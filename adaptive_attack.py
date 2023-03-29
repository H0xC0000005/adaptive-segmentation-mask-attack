"""
@author: Utku Ozbulak
@repository: github.com/utkuozbulak/adaptive-segmentation-mask-attack
@article: Impact of Adversarial Examples on Deep Learning Models for Biomedical Image Segmentation
@conference: MICCAI-19
"""
from __future__ import annotations

import gc
import random
import typing
from os.path import isfile

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

from cityscape_dataset import CityscapeDataset
from self_defined_loss import *
import re

# from torch.autograd import Variable
# In-repo imports


# random.seed(4)
from stats_logger import StatsLogger


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
    def __init__(self, device_id: int, model: torch.nn.Module, tau: float, beta: float, *, use_cpu=False):
        self.temporary_class_id = 65535
        self.unique_classes = None  # much like a bit hacking that this is quite dynamic
        self.device_id = device_id
        self.model = model
        self.tau = tau
        self.beta = beta
        self.use_cpu = use_cpu
        # print(f"device of self model: {self.model.device}")

    @staticmethod
    def update_perturbation_multiplier(option: str, beta, tau, iou) -> float:
        if option is None:
            return beta / 2
        if "incr" in option:
            return AdaptiveSegmentationMaskAttack.update_perturbation_multiplier_incrLR(beta, tau, iou)
        if "decr" in option:
            return AdaptiveSegmentationMaskAttack.update_perturbation_multiplier_decrLR(beta, tau, iou)
        raise ValueError(f"in function update_perturbation_multiplier, get option {option} that is invalid. "
                         f"it should be None, containing 'incr' or 'decr'.")

    @staticmethod
    def update_perturbation_multiplier_incrLR(beta, tau, iou):
        return beta * iou + tau

    @staticmethod
    def update_perturbation_multiplier_decrLR(beta, tau, iou):
        return beta * (1 - iou) + tau

    @staticmethod
    def calculate_loss_facade(x, y, metric: str):
        metric2method_mapp = {"l1": AdaptiveSegmentationMaskAttack.calculate_l1_loss,
                              "l2": AdaptiveSegmentationMaskAttack.calculate_l2_loss,
                              "linf": AdaptiveSegmentationMaskAttack.calculate_linf_loss,
                              "l0": AdaptiveSegmentationMaskAttack.calculate_l0_loss,
                              "l4": AdaptiveSegmentationMaskAttack.calculate_l4_loss,
                              }
        mtr = metric.lower()
        if mtr not in metric2method_mapp:
            raise NotImplementedError(f"specified metric {metric} not implemented. "
                                      f"valid set: {metric2method_mapp.keys()}")
        else:
            return metric2method_mapp[mtr](x, y)

    @staticmethod
    def calculate_l0_loss(x, y, *, threshold=1e-6):
        # diff = (x - y)
        # diff = torch.abs(diff)
        # diff[diff > threshold] = 1
        # diff[diff <= threshold] = 0
        # return torch.sum(diff)
        loss = torch.dist(x, y, p=0).item()
        return loss

    @staticmethod
    def calculate_linf_loss(x, y):
        diff = (x - y)
        max_loss = torch.max(diff)
        return max_loss

    @staticmethod
    def calculate_l4_loss(x, y):
        return torch.dist(x, y, p=4).item()

    @staticmethod
    def calculate_l2_loss(x, y):
        # loss = (x - y) ** 2
        # # print(loss)
        # # print(f"min of loss: {np.amin(loss)}, max of loss: {np.amax(loss)}")
        # for a in reversed(range(1, loss.dim())):
        #     loss = loss.sum(a, keepdim=False)
        # loss = loss.sum()
        loss = torch.dist(x, y, p=2).item()
        return loss

    @staticmethod
    def calculate_l1_loss(x, y):
        # loss = torch.abs(x - y)
        # # print(loss)
        # # print(f"min of loss: {np.amin(loss)}, max of loss: {np.amax(loss)}")
        # for a in reversed(range(1, loss.dim())):
        #     loss = loss.sum(a, keepdim=False)
        # loss = loss.sum()
        loss = torch.dist(x, y, p=1).item()
        return loss

    @staticmethod
    def calculate_l1_dense_mask(perturbation: torch.Tensor) -> torch.Tensor:
        """
        calculate the most dense parts of a perturbation and return as a 0-1 mask as tensor
        """
        pass

    def do_model_prediction(self, input_image: torch.Tensor) -> torch.Tensor:

        # Image to perform the attack on
        image_to_optimize: torch.Tensor
        image_to_optimize = input_image.unsqueeze(0)
        # Forward pass
        out: torch.Tensor
        out = self.model(image_to_optimize)
        pred_out = torch.argmax(out, dim=1).float()
        return pred_out

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

    # def perform_attack(self,
    #                    input_image: torch.Tensor,
    #                    org_mask: torch.Tensor,
    #                    target_mask: torch.Tensor | None,
    #                    *,
    #                    target_class_list: typing.Collection,
    #
    #                    kwargs_for_metrics: dict[str, typing.Any] = None,
    #                    perturbation_mask: torch.Tensor = None,
    #                    initial_perturbation: torch.Tensor = None,
    #                    loss_metric: str = "l2",
    #                    additional_loss_metric: typing.Collection[typing.Callable] = None,
    #                    additional_loss_weights: typing.Collection[float] = None,
    #                    total_iter=500,
    #                    classification_vs_norm_ratio: float = 1 / 16,
    #                    step_update_multiplier: float = 4,
    #                    save_samples: bool = True,
    #                    save_path: str | None = None,
    #                    verbose: bool = False,
    #                    report_stats: bool = True,
    #                    report_stat_interval: int = 10,
    #                    early_stopping_accuracy_threshold: float | None = 1e-4,
    #                    dynamic_LR_option: str = None,
    #
    #                    logger_agent: StatsLogger = None,
    #                    logging_variables: list | tuple = None,
    #                    ) -> torch.Tensor:

    def perform_static_universal_attack(self,
                                        segmentation_dataset: CityscapeDataset,
                                        target_mask: torch.Tensor,
                                        *,
                                        dataset_loop_count: int = 2,
                                        batch_size: int = 4,
                                        loss_metric: str | list[str] = "l2",
                                        additional_loss_metric: list = None,
                                        additional_loss_weights: list = None,
                                        each_step_iter: int = 100,
                                        save_sample: bool = True,
                                        save_path: str = './adv_results/cityscapes_universal_results/',
                                        verbose: bool = True,
                                        classification_vs_norm_ratio: float = 1 / 16,
                                        perturbation_learning_rate: float = 1e-3,
                                        attack_learning_multiplier: float = 1,
                                        report_stat_interval: int = 10,
                                        early_stopping_accuracy_threshold=1e-3,
                                        dynamic_LR_option: str = None,
                                        logger_agent: StatsLogger = None,

                                        eval_dataset: Dataset = None,
                                        eval_model: torch.nn.Module = None,
                                        ) -> torch.Tensor:
        save_suffix = ""
        if save_path is not None:
            save_suffix += "TS_"
            save_suffix += f"step{each_step_iter}_aLR{attack_learning_multiplier}" \
                           f"_pLR{perturbation_learning_rate}_{loss_metric}_"
            save_suffix += "LW{:.2f}_".format(1 / classification_vs_norm_ratio)
            if additional_loss_metric is not None:
                save_suffix += f"metc_"
                if additional_loss_weights is not None:
                    save_suffix += f"w{additional_loss_weights}_"
            if dynamic_LR_option is not None:
                save_suffix += f"{dynamic_LR_option}_"
            save_path += save_suffix + "/"

        mask2: torch.Tensor
        mask2 = target_mask
        m2set = set(np.unique(mask2.numpy(force=True)))
        try:
            m2set.remove(255)
        except KeyError:
            pass
        print(f">>> untarget atk against {m2set}.")

        if save_sample:
            m2s: torch.Tensor
            m2s = copy.deepcopy(mask2)
            m2s = m2s.cpu().detach()
            m2s = CityscapeDataset.decode_target(m2s)
            save_image(m2s, "target_mask", save_path, normalize=False)

        # hardcoded logging variables
        global_perturbation: torch.Tensor | None
        global_perturbation = None
        counter = 1
        batch_pert = None
        for cnt in range(dataset_loop_count):
            for sample_tuple in segmentation_dataset:
                if len(sample_tuple) == 2:
                    # image, mask
                    img = sample_tuple[0]
                    mask = sample_tuple[1]
                elif len(sample_tuple) == 3:
                    # name, image, mask
                    name = sample_tuple[0]
                    img = sample_tuple[1]
                    mask = sample_tuple[2]
                else:
                    raise ValueError(f"sample tuple returned by your dataset is {len(sample_tuple)} instead of 2 or 3")

                # report_image_statistics(mask)
                # report_image_statistics(img)
                # Perform attack
                if global_perturbation is None:
                    global_perturbation = torch.zeros(img.shape, device="cpu")
                    if not self.use_cpu:
                        global_perturbation = global_perturbation.cuda(self.device_id)
                # if counter == 1:
                #     save_image(mask, "mask_test1", save_path, normalize=False)
                #     save_image(mask2, "mask_test2", save_path, normalize=False)

                current_pert = self.perform_attack(img,
                                                   mask,
                                                   mask2,
                                                   loss_metric=loss_metric,
                                                   initial_perturbation=global_perturbation,
                                                   save_samples=False,
                                                   target_class_list=m2set,
                                                   total_iter=each_step_iter,
                                                   report_stat_interval=report_stat_interval,
                                                   verbose=verbose,
                                                   report_stats=verbose,
                                                   early_stopping_accuracy_threshold=early_stopping_accuracy_threshold,
                                                   classification_vs_norm_ratio=classification_vs_norm_ratio,
                                                   perturbation_mask=None,
                                                   step_update_multiplier=attack_learning_multiplier,
                                                   additional_loss_metric=additional_loss_metric,
                                                   additional_loss_weights=additional_loss_weights,

                                                   )
                # update towards difference, or absolute value?
                # global_perturbation += (current_pert - global_perturbation) * perturbation_learning_rate
                if counter % batch_size == 0:
                    global_perturbation += current_pert * perturbation_learning_rate / batch_size
                    batch_pert = None
                else:
                    if batch_pert is None:
                        batch_pert = copy.deepcopy(current_pert)
                    else:
                        batch_pert += current_pert

                if counter % report_stat_interval == 0:
                    if save_sample:
                        global_perturb_np: np.ndarray
                        global_perturb_np = global_perturbation.cpu().detach().numpy()[0]
                        save_image(global_perturb_np, 'iter_' + str(counter),
                                   save_path + 'global_ptb', normalize=True)
                        mask_dec = CityscapeDataset.decode_target(mask)
                        mask2_dec = CityscapeDataset.decode_target(mask2)
                        save_image(mask_dec, f"mask_{counter}", save_path + "masks", normalize=False)
                        save_image(mask2_dec, f"mask_{counter}", save_path + "masks2", normalize=False)
                    linf = torch.max(global_perturbation)
                    l2 = torch.sum(global_perturbation * global_perturbation)
                    # print(f"Iter: {counter}\t Linf: {linf}\t L2: {l2}")
                    if logger_agent is not None:
                        logger_agent.log_variable("Iteration", counter)
                        logger_agent.log_variable("Linf", linf)
                        logger_agent.log_variable("L2", l2)
                counter += 1
        save_image(global_perturbation, "global_pert", save_path, normalize=True)
        if eval_dataset is None:
            return global_perturbation

        counter = 1
        for eval_tuple in eval_dataset:
            img_eval: torch.Tensor
            mask_eval: torch.Tensor
            name, img_eval, mask_eval = eval_tuple[0], eval_tuple[1], eval_tuple[2]
            if not self.use_cpu:
                img_eval = img_eval.cuda(self.device_id)
                mask_eval = mask_eval.cuda(self.device_id)
            img_eval = img_eval.unsqueeze(0)
            img_eval_pert = img_eval + global_perturbation
            pred_out: torch.Tensor
            pred_out_pert: torch.Tensor
            pred_out = eval_model(img_eval)
            pred_out_pert = eval_model(img_eval_pert)
            pred_out = pred_out.cpu().detach()
            pred_out_pert = pred_out_pert.cpu().detach()
            pred_out = torch.argmax(pred_out, dim=1)
            pred_out_pert = torch.argmax(pred_out_pert, dim=1)
            pred_out = CityscapeDataset.decode_target(pred_out)
            pred_out_pert = CityscapeDataset.decode_target(pred_out_pert)

            save_image(pred_out, f"eval_{counter}", save_path + "eval")
            save_image(pred_out_pert, f"eval_{counter}_pert", save_path + "eval")

            counter += 1
        if logger_agent is not None:
            print(f"saving variables to {save_path + save_suffix}.csv")
            logger_agent.save_variables(("Iteration", "Linf", "L2"), save_path + f"{save_suffix}.csv")

        return global_perturbation

    def perform_targeted_universal_attack(self,
                                          segmentation_dataset: CityscapeDataset,
                                          original_class: int,
                                          target_class: int,
                                          *,
                                          dataset_loop_count: int = 2,
                                          batch_size: int = 4,
                                          loss_metric: str | list[str] = "l2",
                                          additional_loss_metric: list = None,
                                          additional_loss_weights: list = None,
                                          each_step_iter: int = 100,
                                          save_sample: bool = True,
                                          save_path: str = './adv_results/cityscapes_universal_results/',
                                          verbose: bool = True,
                                          classification_vs_norm_ratio: float = 1 / 16,
                                          perturbation_learning_rate: float = 1e-3,
                                          attack_learning_multiplier: float = 1,
                                          report_stat_interval: int = 10,
                                          early_stopping_accuracy_threshold=1e-3,
                                          limit_perturbation_to_target: bool = True,
                                          dynamic_LR_option: str = None,
                                          logger_agent: StatsLogger = None,

                                          eval_dataset: Dataset = None,
                                          eval_model: torch.nn.Module = None,
                                          ) -> torch.Tensor:
        save_suffix = ""
        if save_path is not None:
            save_suffix += "T_"
            save_suffix += f"{original_class}-to-{target_class}_"
            save_suffix += f"step{each_step_iter}_aLR{attack_learning_multiplier}" \
                           f"_pLR{perturbation_learning_rate}_{loss_metric}_"
            save_suffix += "LW{:.2f}_".format(1 / classification_vs_norm_ratio)
            if limit_perturbation_to_target:
                save_suffix += f"ptmask_"
            if additional_loss_metric is not None:
                save_suffix += f"metc_"
                if additional_loss_weights is not None:
                    save_suffix += f"w{additional_loss_weights}_"
            if dynamic_LR_option is not None:
                save_suffix += f"{dynamic_LR_option}_"
            save_path += save_suffix + "/"

        # hardcoded logging variables
        global_perturbation: torch.Tensor | None
        global_perturbation = None
        counter = 1
        batch_pert = None
        for cnt in range(dataset_loop_count):
            for sample_tuple in segmentation_dataset:
                if len(sample_tuple) == 2:
                    # image, mask
                    img = sample_tuple[0]
                    mask = sample_tuple[1]
                elif len(sample_tuple) == 3:
                    # name, image, mask
                    name = sample_tuple[0]
                    img = sample_tuple[1]
                    mask = sample_tuple[2]
                else:
                    raise ValueError(f"sample tuple returned by your dataset is {len(sample_tuple)} instead of 2 or 3")

                # report_image_statistics(mask)
                # report_image_statistics(img)
                # Perform attack
                if global_perturbation is None:
                    global_perturbation = torch.zeros(img.shape, device="cpu")
                    if not self.use_cpu:
                        global_perturbation = global_perturbation.cuda(self.device_id)
                mask2 = copy.deepcopy(mask)
                mask2[mask2 == original_class] = target_class
                if limit_perturbation_to_target:
                    pert_mask = copy.deepcopy(mask)
                    pert_mask[pert_mask == original_class] = self.temporary_class_id
                    pert_mask[pert_mask != self.temporary_class_id] = 0
                    pert_mask[pert_mask == self.temporary_class_id] = 1
                else:
                    pert_mask = None

                # if counter == 1:
                #     save_image(mask, "mask_test1", save_path, normalize=False)
                #     save_image(mask2, "mask_test2", save_path, normalize=False)

                current_pert = self.perform_attack(img,
                                                   mask,
                                                   mask2,
                                                   loss_metric=loss_metric,
                                                   initial_perturbation=global_perturbation,
                                                   save_samples=False,
                                                   target_class_list=[target_class],
                                                   total_iter=each_step_iter,
                                                   report_stat_interval=report_stat_interval,
                                                   verbose=verbose,
                                                   report_stats=verbose,
                                                   early_stopping_accuracy_threshold=early_stopping_accuracy_threshold,
                                                   classification_vs_norm_ratio=classification_vs_norm_ratio,
                                                   perturbation_mask=pert_mask,
                                                   step_update_multiplier=attack_learning_multiplier,
                                                   additional_loss_metric=additional_loss_metric,
                                                   additional_loss_weights=additional_loss_weights,

                                                   )
                # update towards difference, or absolute value?
                # global_perturbation += (current_pert - global_perturbation) * perturbation_learning_rate
                if counter % batch_size == 0:
                    global_perturbation += current_pert * perturbation_learning_rate / batch_size
                    batch_pert = None
                else:
                    if batch_pert is None:
                        batch_pert = copy.deepcopy(current_pert)
                    else:
                        batch_pert += current_pert

                if counter % report_stat_interval == 0:
                    if save_sample:
                        global_perturb_np: np.ndarray
                        global_perturb_np = global_perturbation.cpu().detach().numpy()[0]
                        save_image(global_perturb_np, 'iter_' + str(counter),
                                   save_path + 'global_ptb', normalize=True)
                        mask_dec = CityscapeDataset.decode_target(mask)
                        mask2_dec = CityscapeDataset.decode_target(mask2)
                        save_image(mask_dec, f"mask_{counter}", save_path + "masks", normalize=False)
                        save_image(mask2_dec, f"mask_{counter}", save_path + "masks2", normalize=False)
                    linf = torch.max(global_perturbation)
                    l2 = torch.sum(global_perturbation * global_perturbation)
                    # print(f"Iter: {counter}\t Linf: {linf}\t L2: {l2}")
                    if logger_agent is not None:
                        logger_agent.log_variable("Iteration", counter)
                        logger_agent.log_variable("Linf", linf)
                        logger_agent.log_variable("L2", l2)
                counter += 1
        save_image(global_perturbation, "global_pert", save_path, normalize=True)
        if eval_dataset is None:
            return global_perturbation

        counter = 1
        for eval_tuple in eval_dataset:
            img_eval: torch.Tensor
            mask_eval: torch.Tensor
            name, img_eval, mask_eval = eval_tuple[0], eval_tuple[1], eval_tuple[2]
            if not self.use_cpu:
                img_eval = img_eval.cuda(self.device_id)
                mask_eval = mask_eval.cuda(self.device_id)
            img_eval = img_eval.unsqueeze(0)
            img_eval_pert = img_eval + global_perturbation
            pred_out: torch.Tensor
            pred_out_pert: torch.Tensor
            pred_out = eval_model(img_eval)
            pred_out_pert = eval_model(img_eval_pert)
            pred_out = pred_out.cpu().detach()
            pred_out_pert = pred_out_pert.cpu().detach()
            pred_out = torch.argmax(pred_out, dim=1)
            pred_out_pert = torch.argmax(pred_out_pert, dim=1)
            pred_out = CityscapeDataset.decode_target(pred_out)
            pred_out_pert = CityscapeDataset.decode_target(pred_out_pert)

            save_image(pred_out, f"eval_{counter}", save_path + "eval")
            save_image(pred_out_pert, f"eval_{counter}_pert", save_path + "eval")

            counter += 1
        if logger_agent is not None:
            print(f"saving variables to {save_path + save_suffix}.csv")
            logger_agent.save_variables(("Iteration", "Linf", "L2"), save_path + f"{save_suffix}.csv")

        return global_perturbation

    def perform_L1plus_second_attack(self,
                                     input_image: torch.Tensor,
                                     org_mask: torch.Tensor,
                                     target_mask: torch.Tensor,
                                     *,
                                     select_l1_method: SelectL1Method = None,
                                     additional_select_postprocessing: typing.Collection[
                                         L1SelectionPostprocessing] = None,
                                     kwargs_for_metrics: dict[str, typing.Any] = None,
                                     initial_perturbation: torch.Tensor = None,
                                     loss_metric: str = "l2",
                                     additional_loss_metric: typing.Collection = None,
                                     additional_loss_weights: typing.Collection = None,
                                     target_class_list: typing.Collection[int],
                                     l1_total_iter: int = 50,
                                     atk_total_iter: int = 500,
                                     classification_vs_norm_ratio: float = 1 / 16,
                                     step_update_multiplier: float = 4,
                                     report_stat: bool = True,
                                     report_stat_interval: int = 25,
                                     save_l1_samples: bool = False,
                                     save_l1_path: str = None,
                                     save_attack_samples: bool = False,
                                     save_attack_path: str = None,
                                     save_mask_path: str = None,
                                     save_mask_sample: bool = False,
                                     early_stopping_threshold: float = None,

                                     logger_agent: StatsLogger = None,
                                     logging_variables: typing.Collection = None,

                                     ) -> torch.Tensor:
        """
        loss metric, c_vs_norm_ratio are for step L2

        select_l1_method should be a callable that is a closure. a closure is defined here with default args
        """

        if save_attack_samples:
            save_attack_path += f"/l1{l1_total_iter}_ln{atk_total_iter}"
        if save_mask_sample:
            save_mask_path += f"/l1{l1_total_iter}_ln{atk_total_iter}"
        if save_l1_samples:
            save_l1_path += f"/l1{l1_total_iter}_ln{atk_total_iter}"

        if not self.use_cpu:
            org_mask = org_mask.cuda(self.device_id)
        else:
            org_mask = org_mask.cpu()
        print(">>>>>", org_mask.device, self.device_id)
        width, height = input_image.shape[-2], input_image.shape[-1]
        if select_l1_method is None:
            select_l1_method = SelectRectL1IntenseRegion(width=width // 16,
                                                         height=height // 16,
                                                         number_of_rec=4,
                                                         allow_overlap=False,
                                                         overlap_threshold=(height + width) // 64)
        print(input_image.shape, org_mask.shape, target_mask.shape)

        first_l1_pert = self.perform_attack(input_image, org_mask, target_mask,
                                            target_class_list=target_class_list,
                                            loss_metric="l1",
                                            total_iter=l1_total_iter,
                                            classification_vs_norm_ratio=classification_vs_norm_ratio / 2,
                                            save_samples=save_l1_samples,
                                            verbose=False,
                                            report_stats=report_stat,
                                            report_stat_interval=report_stat_interval,
                                            save_path=save_l1_path,
                                            early_stopping_accuracy_threshold=early_stopping_threshold,

                                            )
        selected_l1_mask: torch.Tensor
        print(f"L1 perturbation shape: {first_l1_pert.shape}")
        selected_l1_mask = select_l1_method(first_l1_pert)
        # print(selected_l1_mask.device)
        if additional_select_postprocessing is not None:
            for pros_meth in additional_select_postprocessing:
                selected_l1_mask = pros_meth(org_mask, selected_l1_mask)
        if save_mask_sample:
            save_image(selected_l1_mask, "selected_mask", save_mask_path)
        print(f"-------------------------------- l1 completed")
        print(type(selected_l1_mask))
        print(input_image.shape, org_mask.shape, target_mask.shape)
        second_pert = self.perform_attack(input_image, org_mask, target_mask,
                                          target_class_list=target_class_list,
                                          loss_metric=loss_metric,
                                          initial_perturbation=initial_perturbation,
                                          perturbation_mask=selected_l1_mask,
                                          kwargs_for_metrics=kwargs_for_metrics,
                                          additional_loss_metric=additional_loss_metric,
                                          additional_loss_weights=additional_loss_weights,
                                          total_iter=atk_total_iter,
                                          classification_vs_norm_ratio=classification_vs_norm_ratio,
                                          step_update_multiplier=step_update_multiplier,
                                          save_samples=save_attack_samples,
                                          save_path=save_attack_path,
                                          report_stats=report_stat,
                                          report_stat_interval=report_stat_interval,
                                          early_stopping_accuracy_threshold=early_stopping_threshold,

                                          logger_agent=logger_agent,
                                          logging_variables=logging_variables
                                          )
        print(input_image.device)
        print(second_pert.device)
        if self.use_cpu:
            input_image = input_image.cpu()
            org_mask = org_mask.cpu()
        else:
            input_image = input_image.cuda(self.device_id)
            org_mask = org_mask.cuda(self.device_id)
        pred_mask = self.do_model_prediction(input_image=input_image + second_pert)
        externality = 0
        for target_class in target_class_list:
            externality += evaluate_externality(selected_l1_mask, org_mask, pred_mask, target_class)
        externality /= len(target_class_list)

        # Create a file object in write mode
        with open(save_mask_path + "/externality.txt", "w") as file:
            # Write the float variable to the file
            file.write(str(externality))

        # Close the file
        file.close()

        return second_pert

    def perform_attack(self,
                       input_image: torch.Tensor,
                       org_mask: torch.Tensor,
                       target_mask: torch.Tensor | None,
                       *,
                       target_class_list: typing.Collection,

                       kwargs_for_metrics: dict[str, typing.Any] = None,
                       perturbation_mask: torch.Tensor = None,
                       initial_perturbation: torch.Tensor = None,
                       loss_metric: str = "l2",
                       additional_loss_metric: typing.Collection[typing.Callable] = None,
                       additional_loss_weights: typing.Collection[float] = None,
                       total_iter=500,
                       classification_vs_norm_ratio: float = 1 / 16,
                       step_update_multiplier: float = 4,
                       save_samples: bool = True,
                       save_path: str | None = None,
                       verbose: bool = False,
                       report_stats: bool = True,
                       report_stat_interval: int = 10,
                       early_stopping_accuracy_threshold: float | None = 1e-4,
                       dynamic_LR_option: str = None,

                       logger_agent: StatsLogger = None,
                       logging_variables: list | tuple = None,
                       ) -> torch.Tensor:

        assert not (save_path is None and save_samples), f"in perform_attack, " \
                                                         f"attempt to save samples without save path specified."
        save_suffix = ""
        if save_path is not None:
            save_samples = True
            if target_mask is not None:
                save_suffix += "T_"
            else:
                save_suffix += "U_"
                save_suffix += f"{target_class_list}_"
            save_suffix += f"{total_iter}_LR{step_update_multiplier}_{loss_metric}_"
            save_suffix += "LW{:.2f}_".format(1 / classification_vs_norm_ratio)
            if perturbation_mask is not None:
                save_suffix += f"ptmask_"
            if additional_loss_metric is not None:
                save_suffix += f"metc_"
                if additional_loss_weights is not None:
                    save_suffix += f"w{additional_loss_weights}_"
            if dynamic_LR_option is not None:
                save_suffix += f"{dynamic_LR_option}_"
            save_path += save_suffix + "/"

        if kwargs_for_metrics is None:
            kwargs_for_metrics = dict()
        if additional_loss_metric is not None:
            if additional_loss_weights is None:
                additional_loss_weights = [1] * len(additional_loss_metric)
        if perturbation_mask is not None:
            perturbation_mask_numpy = perturbation_mask.cpu().numpy()
            """
            work by feature: tensor pointwise mul
            if ts1 and ts2 doesn't have same dim, they are matched from tail to head
            eg. [1, 2, 3, 4] can multiply with [3, 4]
            hence here require pert mask to have same dim as mask
            """
            assert perturbation_mask.shape == org_mask.shape, f"in perform_attack, provided perturbation mask " \
                                                              f"with shape " \
                                                              f"{perturbation_mask.shape} that is inconsistent " \
                                                              f"with input shape {org_mask.shape}"
            if not self.use_cpu:
                perturbation_mask = perturbation_mask.cuda(self.device_id)
            else:
                perturbation_mask = perturbation_mask.cpu()
        else:
            perturbation_mask_numpy = None

        def verbose_print(s):
            if verbose:
                print(s)

        def logv(name: str, var):
            if logger_agent is not None and name in logging_variables:
                logger_agent.log_variable(name, var)

        verbose_print(f">>> performing attack on save path {save_path}.")

        device = "cpu" if self.use_cpu else self.device_id
        if len(target_class_list) == 1:
            verbose_print(f">>> perform single class attack on device {device} and target class {target_class_list}")
        elif len(target_class_list) > 1:
            verbose_print(f">>> begin to perform attack on device {device} with unique class list {target_class_list}.")
        else:
            raise ValueError("in perform attack, unique class list is empty")
        target_classes = target_class_list.copy()
        # try:
        #     target_classes.remove(0)
        # except ValueError:
        #     pass
        if save_samples:
            # Save masks
            verbose_print(f"> saving masks to location {save_path}")
            dec_org_mask = CityscapeDataset.decode_target(org_mask.cpu().detach().numpy())
            verbose_print(f"saving masks. mask size: {dec_org_mask.shape}")
            save_image(dec_org_mask, 'original_mask', save_path)
            if target_mask is not None:
                dec_target_mask = CityscapeDataset.decode_target(target_mask.numpy())

                save_image(dec_target_mask, 'target_mask', save_path)

        # Unique classes are needed to simplify prediction loss
        self.unique_classes = target_class_list
        # Have a look at calculate_pred_loss to see where this is used
        # select an out-of-sample distribution class
        self.temporary_class_id = random.randint(0, 999)
        while self.temporary_class_id in self.unique_classes or self.temporary_class_id in range(50):
            self.temporary_class_id = random.randint(0, 999)

        # Assume there is no overlapping part for the first iteration
        pert_mul = self.update_perturbation_multiplier(dynamic_LR_option, self.beta, self.tau, 0)
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
        if not self.use_cpu:
            image_to_optimize = Variable(image_to_optimize.cuda(self.device_id), requires_grad=True)
        # add a bit gaussian noise for stabilization
        noise_variance = 1e-8
        # image_to_optimize = image_to_optimize + (noise_variance ** 0.5) * torch.randn(image_to_optimize.shape)
        if initial_perturbation is not None:
            image_to_optimize = image_to_optimize + initial_perturbation
        # print("VVVV max of im to optimize: ", torch.max(image_to_optimize), "shape: ", image_to_optimize.shape)
        # Copied version of image for l2 dist
        org_im_copy = copy.deepcopy(input_image.cpu())
        if not self.use_cpu:
            org_im_copy = org_im_copy.cuda(self.device_id)
        verbose_printed_size = False

        prev_iou = None
        for single_iter in range(total_iter):
            num_garbage = gc.collect()
            verbose_print(f">>> collected garbage number {num_garbage}")
            # Put in variable to get grads later on
            # Variable deprecated in later torch: they return tensors instead, and autograd = true as default
            if not self.use_cpu:
                image_to_optimize = Variable(image_to_optimize.cuda(self.device_id), requires_grad=True)
            else:
                image_to_optimize = Variable(image_to_optimize.cpu(), requires_grad=True)

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
            # print(dist_loss)
            if target_mask is not None:
                # Prediction loss
                pred_loss = self.calculate_target_pred_loss(target_mask, pred_out, out, target_class=target_classes)
            else:
                pred_loss = self.calculate_untargeted_pred_loss(pred_out, out, original_class=target_classes)
            # Total loss
            pred_loss_weight = 1
            dist_loss_weight = pred_loss_weight / classification_vs_norm_ratio

            """
            update loss, targeted & untargeted have different loss.
            normalized to have total weight of 1
            """
            total_weight = 1 + dist_loss_weight
            if target_mask is not None:
                # TODO: refactor +- signs of both losses, - sign for weight loss really correct?
                # positive dist loss, positive pred loss
                out_grad = pred_loss_weight * pred_loss + dist_loss_weight * dist_loss
            else:
                # positive dist loss, negative untargeted loss
                out_grad = - pred_loss_weight * pred_loss + dist_loss_weight * dist_loss
            if additional_loss_metric is not None:
                kwargs_for_metrics["tensor1"] = pred_out
                kwargs_for_metrics["tensor2"] = target_mask
                for loss_name, cur_weight in zip(additional_loss_metric, additional_loss_weights):
                    kwargs_for_metrics["weight"] = cur_weight
                    if not isinstance(loss_name, typing.Callable):
                        raise NotImplementedError(f"in perform_attack, additional loss metric not callable: currently"
                                                  f" only support callables")
                    # by default loss is positive
                    out_grad += loss_name(kwargs_for_metrics)
                    total_weight += cur_weight
            # normalize
            out_grad /= (total_weight / step_update_multiplier)

            out_grad = torch.sum(out_grad)
            """
            remember to torch.sum() for outgrad
            """
            verbose_print(f"OOO out grad : {out_grad}")
            verbose_print(f"OOO out L-x loss: {dist_loss}")
            verbose_print(f"OOO out pred loss: {pred_loss}")

            # Backward pass
            out_grad.backward()
            perturbed_im: torch.Tensor
            verbose_print(f">>> checking gradient of image to optimize:")
            int_grad: torch.Tensor
            int_grad = image_to_optimize.grad
            verbose_print(f"max of grad: {torch.max(int_grad)}")
            verbose_print(f"min of grad: {torch.max(int_grad)}")

            # apply hardcoded mask on perturbation. by default perturbation "bumps into" mask
            # and clamped (as backprop first and update).

            if perturbation_mask is None:
                cur_pert = image_to_optimize.grad * pert_mul
            else:
                cur_pert = image_to_optimize.grad * perturbation_mask * pert_mul
            perturbed_im = image_to_optimize.data + cur_pert

            # Do another forward pass to calculate new pert_mul
            perturbed_im_out = self.model(perturbed_im)

            # Discretize perturbed image to calculate stats
            perturbed_im_pred_ts: torch.Tensor
            perturbed_im_pred: np.ndarray
            perturbed_im_pred_ts = torch.argmax(perturbed_im_out, dim=1).float()[0]
            perturbed_im_pred = perturbed_im_pred_ts.detach().cpu().numpy()

            # Calculate performance of the attack
            # Similarities
            if target_mask is not None:
                iou, pixel_acc = calculate_multiclass_mask_similarity(perturbed_im_pred,
                                                                      target_mask_numpy,
                                                                      target_classes=target_classes,
                                                                      iou_mask=perturbation_mask_numpy)
            else:
                acc_iou = 0
                for elem in target_classes:
                    verbose_print(f"calculating untargeted iou for class {elem}")
                    temp_mask = copy.deepcopy(org_mask).numpy(force=True).astype(int)
                    temp_mask[temp_mask != elem] = self.temporary_class_id
                    temp_mask[temp_mask == elem] = 1
                    temp_mask[temp_mask == self.temporary_class_id] = 0

                    temp_pred_mask = copy.deepcopy(perturbed_im_pred)
                    temp_pred_mask = temp_pred_mask.astype(int)
                    # print(np.unique(temp_pred_mask))
                    temp_pred_mask[temp_pred_mask == elem] = self.temporary_class_id
                    temp_pred_mask[temp_pred_mask != self.temporary_class_id] = 1
                    temp_pred_mask[temp_pred_mask == self.temporary_class_id] = 0
                    attacked = np.sum(temp_pred_mask * temp_mask)
                    original_total = np.sum(temp_mask)
                    acc_iou += attacked / original_total
                iou = acc_iou / len(target_classes)
                pixel_acc = "untargeted"

            # Distances
            l2_dist, linf_dist = calculate_image_distance(org_im_copy, perturbed_im)
            # Update perturbation multiplier
            pert_mul = self.update_perturbation_multiplier(dynamic_LR_option, self.beta, self.tau, iou)

            # Update image to optimize and ensure boxt constraint
            verbose_print(f">>> checking type of perturbed im and im to op")
            """
            is this clamp(0,1) really not problemistic for multi class segmentation?
            """
            image_to_optimize = copy.deepcopy(perturbed_im)
            verbose_print(f">>> checking image to optimize integrity")
            verbose_print(f"max: {torch.max(image_to_optimize[0])}")
            verbose_print(f"min: {torch.min(image_to_optimize[0])}")
            if single_iter % report_stat_interval == 0 or single_iter == total_iter - 1:
                if save_samples:
                    pred_out_np: np.ndarray
                    pred_out_np = pred_out.cpu().detach().numpy()[0]
                    pred_out_np = pred_out_np.astype('uint8')
                    pred_out_np = CityscapeDataset.decode_target(pred_out_np)
                    save_image(pred_out_np, 'iter_' + str(single_iter),
                               save_path + 'prediction', normalize=False)
                    # is this copy really necessary for normalization?
                    unnormalized_imtoptimize: torch.Tensor
                    unnormalized_imtoptimize = copy.deepcopy(image_to_optimize.data.cpu().detach())
                    normalized_imtoptimize = CityscapeDataset.train_image_to_rgb(unnormalized_imtoptimize)
                    save_batch_image(normalized_imtoptimize.numpy(), 'iter_' + str(single_iter),
                                     save_path + 'modified_image', normalize=True, save_flag=True)
                    save_batch_image_difference(image_to_optimize.data.cpu().detach().numpy(),
                                                org_im_copy.data.cpu().detach().numpy(),
                                                'iter_' + str(single_iter), save_path + 'added_perturbation',
                                                normalize=True, save_flag=True
                                                )
                if report_stats:
                    print('Iter:', single_iter, '\tIOU Overlap:', iou,
                          '\tPixel Accuracy:', pixel_acc,
                          '\n\t\tL2 Dist:', l2_dist,
                          '\tL_inf dist:', linf_dist)
                logv("iteration", single_iter)
                logv("iou", iou)
                logv("pixelwise accuracy", pixel_acc)
                logv("L2 norm", l2_dist),
                logv("Linf norm", linf_dist)
                if isinstance(dist_loss, torch.Tensor):
                    logv("selected distance", dist_loss.cpu().detach())
                else:
                    logv("selected distance", dist_loss)

                # early stopping via iou, a simple control
                if early_stopping_accuracy_threshold is not None and \
                        prev_iou is not None and \
                        iou - prev_iou <= early_stopping_accuracy_threshold and single_iter >= total_iter / 2:
                    if report_stats:
                        print(f"IOU diff less than threshold. returning")
                    print(f"/// prev iou: {prev_iou}, iou: {iou}")
                    break
                else:
                    prev_iou = iou

        if save_samples and logger_agent is not None:
            logger_agent.save_variables(logging_variables, save_path + f"{save_suffix}.csv")
        # unormalized final diff as perturbation. throw it back
        final_diff = save_batch_image_difference(image_to_optimize.data,
                                                 org_im_copy.data,
                                                 None,
                                                 normalize=False,
                                                 save_flag=False,
                                                 )
        return final_diff
