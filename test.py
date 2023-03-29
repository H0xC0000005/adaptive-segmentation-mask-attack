from helper_functions import *

import torch


m1 = np.array([[0, 0, 1, 1, ],
               [0, 0, 1, 1, ],
               [0, 0, 1, 1, ],
               [0, 0, 1, 1, ]], dtype=float)
m2 = np.array([[0, 7, 1, 1, ],
               [0, 5, 1, 1, ],
               [0, 4, 0, 0, ],
               [0, 1, 1, 1, ]], dtype=float)
mo = np.array([[1, 7, 1, 1, ],
               [0, 5, 1, 1, ],
               [0, 4, 0, 0, ],
               [0, 1, 1, 1, ]], dtype=float)

t1 = torch.from_numpy(m1)
t2 = torch.from_numpy(m2)
to = torch.from_numpy(mo)
ext = evaluate_externality(t1, to, t2, 0, evaluation_mask=None)
print(ext)
