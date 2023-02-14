"""
@author: Utku Ozbulak
@repository: github.com/utkuozbulak/adaptive-segmentation-mask-attack
@article: Impact of Adversarial Examples on Deep Learning Models for Biomedical Image Segmentation
@conference: MICCAI-19
"""
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
        image_path=root+'data/cityscape/image',
        mask_path=root+'/data/cityscape/mask'
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
    # vanishing_class = 13
    vanishing_class = None

    # Read images
    """hardcoded static attack"""
    im_name1, im1, mask1 = cityscape_dataset[0]
    im_name2, im2, mask2 = cityscape_dataset[1]

    # print(f"mmmmm model out test:")
    # im1_ts = im1
    # im1_ts = torch.unsqueeze(im1_ts, 0)
    # out = model(im1_ts)
    # pred_out = torch.argmax(out, dim=1)
    # pred_out = torch.squeeze(pred_out)
    # pred_out = pred_out.numpy()
    # pred_out = CityscapeDataset.decode_target(pred_out)
    # save_image(pred_out, "model_out_test", root)
    # # out_img = Image.fromarray(pred_out)
    # # out_img.save("/des")
    # print(f"mmmmm model out saved.")

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
    # m1set = CityscapeDataset.normalize_label(m1set)
    m2list = np.unique(mask2.numpy())
    print(f"mask2 unique check in main: {np.unique(m2list)}")
    # m2set = CityscapeDataset.encode_full_class_array(m2list)
    m2set = set(m2list)
    # m2set = CityscapeDataset.normalize_label(m2set)
    try:
        m2set.remove(255)
    except KeyError:
        pass
    """
    attack sandbox
    """
    start_time = time.time()
    # adaptive_attack.perform_attack(im2,
    #                                mask2,
    #                                mask1,
    #                                unique_class_list=list(set().union(m1set, m2set)),
    #                                total_iter=200)
    mask1 = copy.deepcopy(mask2)
    mask1[mask1 == 13] = 8
    adaptive_attack.perform_attack(im2,
                                   mask2,
                                   mask1,
                                   loss_metric="l0",
                                   save_path=root + "adv_results/cityscapes_results/",
                                   unique_class_list=[8],
                                   total_iter=200,
                                   report_stat_interval=50,
                                   verbose=False)
    # adaptive_attack.perform_attack(im2,
    #                                mask2,
    #                                mask1,
    #                                unique_class_list=[0, vanishing_class],
    #                                total_iter=200)
    # adaptive_attack.perform_attack(im2,
    #                                mask2,
    #                                mask1,
    #                                unique_class_list=[0, 7, 11, 13],
    #                                total_iter=200,
    #                                verbose=True)
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
    # adaptive_attack.perform_attack(im2,
    #                                mask2,
    #                                None,
    #                                unique_class_list=[0, 13],
    #                                loss_metric="l1",
    #                                total_iter=200,
    #                                verbose=False)

    end_time = time.time()
    print(f">>> attack ended. time elapsed: {end_time - start_time}")

