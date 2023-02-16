
import collections
import copy

import numpy as np
import torch
from PIL import Image

import network._deeplab
# In-repo imports
from cityscape_dataset import CityscapeDataset
from helper_functions import load_model, save_image
from adaptive_attack import AdaptiveSegmentationMaskAttack
import torch.nn as nn
import time

# USE_CPU = True
USE_CPU = False


root = "/home/peizhu/Desktop/proj/segmentation-atk-pipeline/"
# torch.cuda.set_enabled_lms(True)

if __name__ == '__main__':

    # Glaucoma dataset
    cityscape_dataset = CityscapeDataset(
        image_path=root+'data/small_berlin_set/image',
        mask_path=root+'data/small_berlin_set/mask'
    )
    # GPU parameters
    DEVICE_ID = 0

    # Load models, change it to where you download the models to
    model: nn.Module
    model = network.modeling.__dict__["deeplabv3plus_mobilenet"](num_classes=19, output_stride=8)

    model_dict = load_model(root+'models/best_deeplabv3plus_mobilenet_cityscapes_os16.pth')

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
    tau = 1e-7
    beta = 1e-6
    original_class = 13
    target_class = 8

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
    start_time = time.time()

    pert = adaptive_attack.perform_targeted_universal_attack(cityscape_dataset,
                                                             original_class=original_class,
                                                             target_class=target_class,
                                                             loss_metric="l1",
                                                             each_step_iter=500,
                                                             save_sample=True,
                                                             verbose=False,
                                                             report_stat_interval=1)

    end_time = time.time()
    print(f">>> attack ended. time elapsed: {end_time - start_time}")

