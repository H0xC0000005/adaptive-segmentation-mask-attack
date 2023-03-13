import torch
from helper_functions import absolute_index_to_tuple_index, SelectRectL1IntenseRegion, l1d

from stats_logger import *

lg = StatsLogger()

sir = SelectRectL1IntenseRegion(height=2, width=2, number_of_rec=3, in_channels=1)

test_lst3 = [[[8, 8, 1, 1, 5,],
              [8, 8, 0, 0, 8,],
              [0, 1, 10, 0, 9,],
              [0, 1, 1, 1, 1,],
              [32, 1, 6, 0, 12,]]]
ts = torch.Tensor(test_lst3)
conv = torch.nn.Conv2d(kernel_size=(2, 2), in_channels=1, out_channels=1, bias=False)
conv.weight.data = torch.tensor([[[[1.0, 1.0]] * 2]])

tsc = conv(ts)

name1 = "t1"
name2 = "t2"
lg.log_variable(name1, 1)
lg.log_variable(name1, 2)
lg.log_variable(name1, 3)
lg.log_variable(name2, "a")
lg.log_variable(name2, "b")
lg.log_variable(name2, "lambda")
df1 = lg.export_dataframe((name1, name2))
# print(df1)
lg.save_variables((name1, name2), "adv_results/1test.csv")

print(1/12)


# res = sir(ts)
# print(tsc)
# print(res)

# test_lst = [[[[1, 0, 3],
#              [4, 5, 6],
#              [7, 1, 1]]] * 2]
# test_list2 = [[1,1,2]] * 3
# tts = torch.tensor(test_lst)
# print(tts.shape)
# tts2 = torch.tensor(test_list2)
# print(tts2.shape)
#
# print(tts * tts2)