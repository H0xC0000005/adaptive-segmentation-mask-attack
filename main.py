"""
@author: Utku Ozbulak
@repository: github.com/utkuozbulak/adaptive-segmentation-mask-attack
@article: Impact of Adversarial Examples on Deep Learning Models for Biomedical Image Segmentation
@conference: MICCAI-19
"""
# In-repo imports
from eye_dataset import EyeDatasetTest
from helper_functions import load_model
from adaptive_attack import AdaptiveSegmentationMaskAttack
import torch.nn as nn
import sys


if __name__ == '__main__':
    # Glaucoma dataset
    eye_dataset = EyeDatasetTest('/home/peizhu/PycharmProjects/adaptive-segmentation-mask-attack/data/image_samples',
                                 '/home/peizhu/PycharmProjects/adaptive-segmentation-mask-attack/data/mask_samples')
    # GPU parameters
    DEVICE_ID = 0

    # Load models, change it to where you download the models to
    model: nn.Module
    model = load_model('/home/peizhu/PycharmProjects/adaptive-segmentation-mask-attack/models/eye_pretrained_model.pt')
    """
    argv[1]: path to model to load
    """
    # model = load_model(sys.argv[1])
    model.eval()
    model.cpu()
    model.cuda(DEVICE_ID)
    print(f"device of net (by first layer parameter): {next(model.parameters()).device}")

    # Attack parameters
    tau = 1e-7
    beta = 1e-6

    # Read images
    im_name1, im1, mask1 = eye_dataset[0]
    im_name2, im2, mask2 = eye_dataset[1]
    # print(mask2)

    # Perform attack
    adaptive_attack = AdaptiveSegmentationMaskAttack(DEVICE_ID, model, tau, beta)
    """
        def perform_attack(self, input_image, org_mask, target_mask, unique_class_list,
                       total_iter=2501, save_samples=True, save_path='../adv_results/', verbose=True):

    """
    adaptive_attack.perform_attack(im2,
                                   mask2,
                                   mask1,
                                   unique_class_list=[0, 1],
                                   total_iter=1500)
