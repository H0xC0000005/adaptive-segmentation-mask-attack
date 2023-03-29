import collections
import copy

import numpy as np
import pandas as pd
import setuptools.wheel
import torch
from PIL import Image
from stats_logger import StatsLogger

import network._deeplab
# In-repo imports
from cityscape_dataset import CityscapeDataset
from helper_functions import load_model, save_image
from adaptive_attack import AdaptiveSegmentationMaskAttack
from self_defined_loss import *
import torch.nn as nn
import time

# USE_CPU = True
USE_CPU = False

root = "/home/peizhu/Desktop/proj/segmentation-atk-pipeline/"
# torch.cuda.set_enabled_lms(True)

if __name__ == '__main__':

    # Glaucoma dataset
    cityscape_dataset = CityscapeDataset(
        image_path=root + 'data/cityscape/image',
        mask_path=root + '/data/cityscape/mask'
    )
    # cityscape_dataset = CityscapeDataset(
    #     image_path= '/home/peizhu/PycharmProjects/adaptive-segmentation-mask-attack/data/cityscape_single_eg2/image',
    #     mask_path='/home/peizhu/PycharmProjects/adaptive-segmentation-mask-attack/data/cityscape_single_eg2/mask'
    # )

    # GPU parameters
    DEVICE_ID = 0

    # Load models, change it to where you download the models to
    model: nn.Module
    model = network.modeling.__dict__["deeplabv3plus_mobilenet"](num_classes=19, output_stride=8)

    model_dict = load_model(root + 'models/best_deeplabv3plus_mobilenet_cityscapes_os16.pth')

    """
    argv[1]: path to model to load
    """
    # print(f"type of read model: {type(model)}")
    if isinstance(model_dict, dict):
        # print(f"len of model: {len(model_dict)}")
        layerwise_len = ""
        for k in model_dict.keys():
            elem = model_dict[k]
            try:
                # print(f"got type with len(): {type(elem)} with key {k}")
                # layerwise_len += str(len(elem)) + ", "
                if isinstance(elem, np.float64) or isinstance(elem, int):
                    # print(elem)
                    layerwise_len += "1, "
                elif isinstance(elem, dict) or isinstance(elem, collections.OrderedDict):
                    cur_stream = "["
                    for key, value in elem.items():
                        try:
                            cur_stream += f"{len(value)}, "
                        except TypeError:
                            cur_stream += "1n, "
                    cur_stream += "]"
                    layerwise_len += cur_stream
                else:
                    raise RuntimeError(f"unexpected type in model dict elem: {type(elem)}")
            except TypeError:
                # print(f"got type without len() : {type(elem)}")
                layerwise_len += "1, "
        # print(f"key of model dict: {model_dict.keys()}")
        # print(f"len of each layer: {layerwise_len}")
        model.load_state_dict(model_dict['model_state'])
        print(f"loaded model weights to model.")

    else:
        print(f"get model type {type(model_dict)}")
        if isinstance(model_dict, nn.Module):
            model = model_dict
    # model = load_model(sys.argv[1])
    model.eval()
    model.cpu()
    if not USE_CPU:
        model.cuda(DEVICE_ID)
    print(f"device of net (by first layer parameter): {next(model.parameters()).device}")

    # Attack parameters
    tau = 2e-7
    beta = 2e-6
    # vanishing_class = 13
    vanishing_class = None

    # Read images
    """hardcoded static attack"""
    im_name1, im1, mask1 = cityscape_dataset[0]
    im_name2, im2, mask2 = cityscape_dataset[1]
    # print(mask2)
    if vanishing_class is not None:
        mask1[mask1 != vanishing_class] = 0
        mask1[mask1 == vanishing_class] = vanishing_class
        mask2[mask2 != vanishing_class] = 0
        mask2[mask2 == vanishing_class] = vanishing_class

    # Perform attack
    adaptive_attack = AdaptiveSegmentationMaskAttack(DEVICE_ID, model, tau, beta, use_cpu=USE_CPU)
    """
        def perform_attack(self, input_image, org_mask, target_mask, unique_class_list,
                       total_iter=2501, save_samples=True, save_path='../adv_results/', verbose=True):

    """

    # cannot normalize: need index to access model hierarchy
    # TODO: normalize to [0,1]and then decode to [0,19] is also an option
    m1list = np.unique(mask1.numpy())
    print(f"mask1 unique check in main: {np.unique(m1list)}")
    # already encoded in dataset. I am a fool. don't encode again :(
    # m1set = CityscapeDataset.encode_full_class_array(m1list)
    m1set = set(m1list)
    # print(f"mask1 set after dataset encoding: {m1set}")
    # kick out valid but out of training 255 label
    try:
        m1set.remove(255)
    except KeyError:
        pass
    m2list = np.unique(mask2.numpy())
    print(f"mask2 unique check in main: {np.unique(m2list)}")
    m2set = CityscapeDataset.encode_full_class_array(m2list)
    m2set = set(m2list)
    try:
        m2set.remove(255)
    except KeyError:
        pass
    set_union = set.union(m1set, m2set)
    set_intersection = set.intersection(m1set, m2set)
    print(f"union elements of m1 m2: {set_union}")
    print(f"intersection elements of m1 m2: {set_intersection}")

    """
    attack sandbox
    """
    target = 8
    original = 13

    start_time = time.time()
    # adaptive_attack.perform_attack(im2,
    #                                mask2,
    #                                mask1,
    #                                unique_class_list=list(set().union(m1set, m2set)),
    #                                total_iter=200)
    lg_agt = StatsLogger()
    logging_var = ("iteration",
                   "iou",
                   "pixelwise accuracy",
                   "L2 norm",
                   "Linf norm",
                   "selected distance",)
    save_path = root + "adv_results/cityscapes_UO_results/"

    # hiding attack
    mask1 = copy.deepcopy(mask2)
    mask1[mask1 == original] = target

    pert_mask = copy.deepcopy(mask2)
    pert_mask[pert_mask == original] = -1
    pert_mask[pert_mask != -1] = 0
    pert_mask[pert_mask == -1] = 1
    additional_loss = [total_variation]
    dynamic_LR_option = "incr"
    l1m_1 = SelectRectL1IntenseRegion(width=50,
                                      height=50,
                                      number_of_rec=8,
                                      allow_overlap=False,
                                      overlap_threshold=50,
                                      )

    # adaptive_attack.perform_attack(im2,
    #                                mask2,
    #                                None,
    #                                loss_metric="l1",
    #                                save_path=save_path,
    #                                target_class_list=[5, 7, 11, 13],
    #                                total_iter=400,
    #                                report_stat_interval=20,
    #                                verbose=False,
    #                                report_stats=False,
    #                                perturbation_mask=None,
    #                                classification_vs_norm_ratio=1 / 4,
    #                                early_stopping_accuracy_threshold=None,
    #                                additional_loss_metric=None,
    #                                additional_loss_weights=[16],
    #                                logger_agent=lg_agt,
    #                                logging_variables=logging_var,
    #                                step_update_multiplier=1,
    #                                dynamic_LR_option=None,
    #                                )

    # mask1 = np.ones(mask1.shape, dtype='uint8')
    # print(f"mask1 created with shape {mask1.shape}")
    # mask1 = torch.from_numpy(mask1)
    # adaptive_attack.perform_attack(im2,
    #                                mask2,
    #                                mask1,
    #                                unique_class_list=[0, 1, 13],
    #                                total_iter=200,
    #                                verbose=False)
    # mask_2cp = copy.deepcopy(mask2)
    # mask_2cp[mask_2cp == 13] = 1
    # # mask_2cp[0:2, 0:2] = 13
    l1m_2 = SelectTopKPoints(16, dot_radius=1, threshold=None)

    # adaptive_attack.perform_attack(im2,
    #                                mask2,
    #                                None,
    #                                loss_metric="l1",
    #                                save_path=save_path,
    #                                target_class_list=[5, 7, 11, 13],
    #                                total_iter=400,
    #                                report_stat_interval=20,
    #                                verbose=False,
    #                                report_stats=False,
    #                                perturbation_mask=None,
    #                                classification_vs_norm_ratio=1 / 4,
    #                                early_stopping_accuracy_threshold=None,
    #                                additional_loss_metric=None,
    #                                additional_loss_weights=[16],
    #                                logger_agent=lg_agt,
    #                                logging_variables=logging_var,
    #                                step_update_multiplier=1,
    #                                dynamic_LR_option=None,
    #                                )
    # def perform_L1plus_second_attack(self,
    #                                  input_image: torch.Tensor,
    #                                  org_mask: torch.Tensor,
    #                                  target_mask: torch.Tensor,
    #                                  *,
    #                                  select_l1_method: SelectL1Method = None,
    #                                  additional_select_postprocessing: typing.Collection[L1SelectionPostprocessing] =
    #                                  None,
    #                                  kwargs_for_metrics: dict[str, typing.Any] = None,
    #                                  initial_perturbation: torch.Tensor = None,
    #                                  loss_metric: str = "l2",
    #                                  additional_loss_metric: typing.Collection = None,
    #                                  additional_loss_weights: typing.Collection = None,
    #                                  target_class_list: typing.Collection,
    #                                  total_iter: int = 500,
    #                                  classification_vs_norm_ratio: float = 1 / 16,
    #                                  step_update_multiplier: float = 4,
    #                                  report_stat: bool = True,
    #                                  report_stat_interval: int = 25,
    #                                  save_l1_samples: bool = False,
    #                                  save_l1_path: str = None,
    #                                  save_attack_samples: bool = False,
    #                                  save_attack_path: str = None,
    #
    #                                  ) -> torch.Tensor:

    postpro = [MaskingToOriginalClass()]
    # save_image(CityscapeDataset.decode_target(mask2.cpu()), "test_mask", root + "adv_results/test/")
    adaptive_attack.perform_L1plus_second_attack(im2,
                                                 mask2,
                                                 mask1,
                                                 loss_metric="l1",
                                                 select_l1_method=l1m_1,
                                                 additional_select_postprocessing=postpro,
                                                 save_attack_samples=True,
                                                 save_attack_path=root + "adv_results/l1lnrect50n8/attack/",
                                                 save_l1_samples=True,
                                                 save_l1_path=root + "adv_results/l1lnrect50n8/l1mask/",
                                                 save_mask_sample=True,
                                                 save_mask_path=root + "adv_results/l1lnrect50n8/selected_mask",
                                                 target_class_list=[target],
                                                 l1_total_iter=75,
                                                 atk_total_iter=200,
                                                 report_stat_interval=20,
                                                 report_stat=True,
                                                 classification_vs_norm_ratio=1 / 16,
                                                 additional_loss_metric=additional_loss,
                                                 additional_loss_weights=[8],
                                                 step_update_multiplier=256,

                                                 logger_agent=lg_agt,
                                                 logging_variables=logging_var,

                                                 )

    ldf: pd.DataFrame
    ldf = lg_agt.export_dataframe(logging_var)
    # ldf.to_csv(f"{save_path}single_atk.csv")

    # ldf_t = ldf.T
    # # ldf_t.plot(kind="line", x=ldf_t.index)
    # ldf_t.plot(kind="line")
    end_time = time.time()
    print(f">>> attack ended. time elapsed: {end_time - start_time}")
