import numpy as np
import torch
from PIL import Image
from torch import nn

import network
from helper_functions import load_model

model: nn.Module
model = network.modeling.__dict__["deeplabv3plus_mobilenet"](num_classes=19, output_stride=8)

model_dict = load_model('/home/peizhu/PycharmProjects/adaptive-segmentation-mask-attack/models'
                        '/best_deeplabv3plus_mobilenet_cityscapes_os16.pth')

im_as_im = Image.open("adv_results/cityscape_results/modified_image/iter_50.png")

im_as_np = np.asarray(im_as_im)
print(im_as_np.shape)
if im_as_np.shape[2] in (1, 2, 3, 4):
    im_as_np = im_as_np.transpose((2, 0, 1))

im_as_np = np.array([im_as_np])
print(im_as_np.shape)
im_as_tensor: torch.Tensor
im_as_tensor = torch.from_numpy(im_as_np).float()

out: torch.Tensor
out = model(im_as_tensor)

out_image = torch.argmax(out, dim=1).float()
print(type(out_image))



