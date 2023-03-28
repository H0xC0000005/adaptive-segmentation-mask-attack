import numpy as np
import torch

import torch


m1 = np.array([[0, 0, 1, 1, ],
               [0, 0, 1, 1, ],
               [0, 0, 1, 1, ],
               [1, 1, 1, 1, ]])
m2 = np.array([[0, 0, 1, 1, ],
               [0, 0, 1, 1, ],
               [1, 0, 0, 0, ],
               [1, 1, 1, 1, ]])

t1 = torch.from_numpy(m1)
t2 = torch.from_numpy(m2)
print(t1 * t2)
