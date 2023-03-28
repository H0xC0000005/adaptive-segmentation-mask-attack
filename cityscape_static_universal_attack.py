import collections
import copy

import numpy as np
import torch
from PIL import Image

import network._deeplab
# In-repo imports
from cityscape_dataset import CityscapeDataset
from helper_functions import *
from adaptive_attack import AdaptiveSegmentationMaskAttack
import torch.nn as nn
import time

from stats_logger import StatsLogger

# USE_CPU = True
USE_CPU = False

root = "/home/peizhu/Desktop/proj/segmentation-atk-pipeline/"
# torch.cuda.set_enabled_lms(True)

if __name__ == '__main__':

    # Glaucoma dataset
    cityscape_dataset = CityscapeDataset(
        image_path=root + 'data/hamburg_set_downsampled/image',
        mask_path=root + 'data/hamburg_set_downsampled/mask'
    )
    cityscape_dataset_eval = CityscapeDataset(
        image_path=root + 'data/hamburg_set_small/image',
        mask_path=root + 'data/hamburg_set_small/mask'
    )
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
    # tau = 1e-7
    # beta = 1e-6
    tau = 2e-6
    beta = 2e-5
    original_class = 8
    target_class = 2

    # Perform attack
    adaptive_attack = AdaptiveSegmentationMaskAttack(DEVICE_ID, model, tau, beta, use_cpu=USE_CPU)
    """
    def perform_targeted_universal_attack(self,
                                          segmentation_dataset: Dataset,
                                          original_class: int,
                                          target_class: int,
                                          *,
                                          loss_metric: str | list[str] = "l2",
                                          each_step_iter: int = 100,
                                          save_sample: bool = True,
                                          save_path: str = './adv_results/cityscapes_results/',
                                          verbose: bool = True,
                                          perturbation_learning_rate: float = 1e-3,
                                          report_stat_interval: int = 10
                                          ):
    """
    im_name1, im1, mask1 = cityscape_dataset[0]

    start_time = time.time()

    pert = adaptive_attack.perform_static_universal_attack(cityscape_dataset,
                                                           target_mask=mask1,
                                                           loss_metric="l1",
                                                           each_step_iter=32,
                                                           save_sample=True,
                                                           verbose=False,
                                                           save_path='./adv_results/cityscapes_TSV_results/',
                                                           report_stat_interval=5,
                                                           early_stopping_accuracy_threshold=None,
                                                           perturbation_learning_rate=60e-3,
                                                           attack_learning_multiplier=1024,
                                                           eval_dataset=cityscape_dataset_eval,
                                                           eval_model=model,
                                                           logger_agent=StatsLogger()

                                                           )

    report_image_statistics(pert)
    # counter = 1
    # for eval_tuple in cityscape_dataset_eval:
    #     img_eval: torch.Tensor
    #     mask_eval: torch.Tensor
    #     name, img_eval, mask_eval = eval_tuple[0], eval_tuple[1], eval_tuple[2]
    #     if not USE_CPU:
    #         img_eval = img_eval.cuda(DEVICE_ID)
    #         mask_eval = mask_eval.cuda(DEVICE_ID)
    #     img_eval = img_eval.unsqueeze(0)
    #     img_eval_pert = img_eval + pert
    #     pred_out: torch.Tensor
    #     pred_out_pert: torch.Tensor
    #     pred_out = model(img_eval)
    #     pred_out_pert = model(img_eval_pert)
    #     pred_out = pred_out.cpu().detach()
    #     pred_out_pert = pred_out_pert.cpu().detach()
    #     pred_out = torch.argmax(pred_out, dim=1)
    #     pred_out_pert = torch.argmax(pred_out_pert, dim=1)
    #     pred_out = CityscapeDataset.decode_target(pred_out)
    #     pred_out_pert = CityscapeDataset.decode_target(pred_out_pert)
    #
    #     save_image(pred_out, f"eval_{counter}", root + f"adv_results/cityscapes_universal_results/eval/")
    #     save_image(pred_out_pert, f"eval_{counter}_pert", root + f"adv_results/cityscapes_universal_results/eval/")
    #
    #
    #     counter += 1

    end_time = time.time()
    print(f">>> attack ended. time elapsed: {end_time - start_time}")
