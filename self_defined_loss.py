from abc import ABC

import torch

from helper_functions import *
import typing

"""
self defined losses. some naming conventions:
tensor1 (original)
tensor2 (target)
weight (weight of this loss)
others should be loss-specific.

each loss must can be derived since loss is used within backprop.
"""


def get_param_from_kwargs(kwargs: dict[str, typing.Any], key: str):
    try:
        return kwargs[key]
    except KeyError:
        return None


class SelfDefinedLoss(object):

    @staticmethod
    def __call__(**kwargs):
        raise NotImplementedError(f"base class of self defined loss is not implemented.")

    @staticmethod
    def get_weight(kwargs: dict[str, typing.Any]) -> float:
        weight = get_param_from_kwargs(kwargs, "weight")
        if weight is None:
            return 1.0
        else:
            return weight


class TotalVariation(SelfDefinedLoss):
    """
    only support channeled images (i.e. no greyscale)
    """

    @staticmethod
    def __call__(**kwargs: dict[str, typing.Any]) -> float:
        t1: torch.Tensor
        weight: float
        t1 = get_param_from_kwargs(kwargs, "tensor1")
        weight = SelfDefinedLoss.get_weight(kwargs)
        if t1 is None:
            raise ValueError(f"in loss TotalVariation, tensor 1 is None. specify with keyword tensor1=xx.")

        if len(t1.shape) > 3:
            dim_of_size = 2
        else:
            dim_of_size = 1
        t1_roll_down = torch.roll(t1, shifts=1, dims=dim_of_size)
        t1_roll_right = torch.roll(t1, shifts=1, dims=dim_of_size + 1)
        t1_down_square_sum = torch.sum((t1 - t1_roll_down) ** 2)
        t1_right_square_sum = torch.sum((t1 - t1_roll_right) ** 2)

        return (t1_down_square_sum + t1_right_square_sum) * weight


class NonPrintabilityScore(SelfDefinedLoss):
    """
    only support channeled images (i.e. no greyscale)

    can cause problem across normalized tensors (grad vanishing since squared)
    or grad explosion for raw images (same reason)
    take caution.
    """

    @staticmethod
    def __call__(**kwargs):
        t1: torch.Tensor
        weight: float
        printable_tuples: typing.Iterable[tuple[float]]
        t1 = get_param_from_kwargs(kwargs, "tensor1")
        if t1 is None:
            raise ValueError(f"in loss NonPrintabilityScore, tensor 1 is None. specify with keyword tensor1=xx.")
        # torch style colouring
        if len(t1.shape) > 3:
            has_batch = True
        else:
            has_batch = False
        weight = SelfDefinedLoss.get_weight(kwargs)
        printable_tuples = get_param_from_kwargs(kwargs, "printable_tuples")
        global_loss = None
        for elem in printable_tuples:
            cur_color = copy.deepcopy(t1)
            if has_batch:
                for channel in (0, 1, 2):
                    cur_color[:][channel][:][:] = elem[channel]
            else:
                for channel in (0, 1, 2):
                    cur_color[channel][:][:] = elem[channel]
            cur_square_diff = (t1 - cur_color) ** 2
            if global_loss is None:
                global_loss = cur_square_diff
            else:
                # pixel-wise mul
                global_loss *= cur_square_diff
        return torch.sum(global_loss)
