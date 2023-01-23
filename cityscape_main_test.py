"""
@author: Utku Ozbulak
@repository: github.com/utkuozbulak/adaptive-segmentation-mask-attack
@article: Impact of Adversarial Examples on Deep Learning Models for Biomedical Image Segmentation
@conference: MICCAI-19
"""
import collections

import numpy as np

import network._deeplab
# In-repo imports
from cityscape_dataset import CityscapeDataset
from helper_functions import load_model
from adaptive_attack import AdaptiveSegmentationMaskAttack
import torch.nn as nn

USE_CPU = True

if __name__ == '__main__':


    # Glaucoma dataset
    cityscape_dataset = CityscapeDataset(
        '/home/peizhu/PycharmProjects/adaptive-segmentation-mask-attack/data/cityscape/image',
        '/home/peizhu/PycharmProjects/adaptive-segmentation-mask-attack/data/cityscape/mask'
    )
    # GPU parameters
    DEVICE_ID = 0

    # Load models, change it to where you download the models to
    model: nn.Module
    model = network.modeling.__dict__["deeplabv3plus_mobilenet"](num_classes=19, output_stride=8)

    model_dict = load_model('/home/peizhu/PycharmProjects/adaptive-segmentation-mask-attack/models'
                            '/best_deeplabv3plus_mobilenet_cityscapes_os16.pth')

    """
    argv[1]: path to model to load
    """
    print(f"type of read model: {type(model)}")
    if isinstance(model_dict, dict):
        print(f"len of model: {len(model_dict)}")
        layerwise_len = ""
        for k in model_dict.keys():
            elem = model_dict[k]
            try:
                print(f"got type with len(): {type(elem)} with key {k}")
                # layerwise_len += str(len(elem)) + ", "
                if isinstance(elem, np.float64) or isinstance(elem, int):
                    print(elem)
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
                print(f"got type without len() : {type(elem)}")
                layerwise_len += "1, "
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

    # Read images
    """hardcoded static attack"""
    im_name1, im1, mask1 = cityscape_dataset[0]
    im_name2, im2, mask2 = cityscape_dataset[1]
    # print(mask2)

    # Perform attack
    adaptive_attack = AdaptiveSegmentationMaskAttack(DEVICE_ID, model, tau, beta, use_cpu=USE_CPU)
    """
        def perform_attack(self, input_image, org_mask, target_mask, unique_class_list,
                       total_iter=2501, save_samples=True, save_path='../adv_results/', verbose=True):

    """

    # cannot normalize: need index to access model hierarchy
    # TODO: normalize to [0,1]and then decode to [0,19] is also an option
    m1list = np.unique(mask1.numpy())
    m1set = CityscapeDataset.encode_full_class_array(m1list)
    # kick out valid but out of training 255 label
    m1set.remove(255)
    # m1set = CityscapeDataset.normalize_label(m1set)
    m2list = np.unique(mask2.numpy())
    m2set = CityscapeDataset.encode_full_class_array(m2list)
    # m2set = CityscapeDataset.normalize_label(m2set)
    m2set.remove(255)

    adaptive_attack.perform_attack(im2,
                                   mask2,
                                   mask1,
                                   unique_class_list=list(set().union(m1set, m2set)),
                                   total_iter=200)
