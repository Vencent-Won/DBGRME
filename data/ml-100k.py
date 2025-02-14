import os.path
import torch
import numpy as np
import pandas as pd
# Process data


# user_item = {}
# user_rating = {}
# item_list = []
# file_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), "ml-100k/u.data")
# with open(file_path, "r") as f:
#     for ui in f.readlines():
#         # ui == 1::1193::5::978300760
#         # ['1', '1193', '5', '978300760\n']   ->   ['user_id', 'item_id', 'rating', 'rating_time\n']
#         line_split_data = ui.split("\t")
#         if (int(line_split_data[0])-1) not in user_item:
#             user_item.update({int(line_split_data[0])-1: [[int(line_split_data[3]), int(line_split_data[1])-1]]})  # {0: [[978300760, 1192]]}
#         else:
#             user_item[int(line_split_data[0])-1].append([int(line_split_data[3]), int(line_split_data[1])-1])
#         if line_split_data[1] not in item_list:
#             item_list.append(line_split_data[1])
#     f.close()
#
# # print(user_item)
#
# user_item_dict = {}
# item_user_dict = {}
# for key, value in user_item.items():
#     user_item[key].sort()
#     user_item_dict.update({key: np.array(value)[:, 1]})
#
#
# num_users = 0
# num_items = 0
# trainItem, trainUser = [], []
# valItem, valUser = [], []
# testItem,  testUser = [], []
# sum = 0
# iid = 0
# uid = 0
# print(user_item[0])
# for user, value in user_item_dict.items():
#     num_train = int(len(value) * 0.8)
#     num_test = int(len(value) * 0.1)
#     num_val = int(len(value) * 0.1)
#     trainUser.extend([user] *  num_train)
#     trainItem.extend(value[: num_train])
#     testUser.extend([user] * num_test)
#     testItem.extend(value[num_train: num_train + num_test])
#     valUser.extend([user] * num_val)
#     valItem.extend(value[num_train + num_test: num_train + num_test + num_val])
#
#
# df_train = pd.DataFrame(
#     {'user': trainUser,
#      'item': trainItem
#      }
# )
#
# df_test = pd.DataFrame(
#     {'user': testUser,
#      'item': testItem
#      }
# )
#
# df_val = pd.DataFrame(
#     {'user': valUser,
#      'item': valItem
#      }
# )
#
# df_train.to_csv('ml-100k/train_sparse.csv')
# df_test.to_csv('ml-100k/test_sparse.csv')
# df_val.to_csv('ml-100k/val_sparse.csv')

import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
import time
from utils import *
dataset = ''
df_train = pd.read_csv(dataset + r'train_sparse.csv')
df_test = pd.read_csv(dataset + r'test_sparse.csv')
df_combined = pd.concat([df_train, df_test], ignore_index=False)

max_row1 = 0
max_row2 = 0
users = []
items = []
for row in df_train.itertuples():
    if row[1] >= max_row1:
        max_row1 = row[1]
    if row[2] >= max_row2:
        max_row2 = row[2]
    users.append(row[1])
    items.append(row[2])


for row in df_test.itertuples():
    if row[1] >= max_row1:
        max_row1 = row[1]
    if row[2] >= max_row2:
        max_row2 = row[2]

for row in df_val.itertuples():
    if row[1] >= max_row1:
        max_row1 = row[1]
    if row[2] >= max_row2:
        max_row2 = row[2]

print(max_row1)
print(max_row2)

u = set(list(range(max_row1 + 1)))
i = set(list(range(max_row2 + 1)))

n_u = (set(users) - u) | (u - set(users))
n_i = (set(items) - i) | (i - set(items))

print(n_u)
print(n_i)

train_items = {}
test_items = {}
val_items = {}

# import numpy as np
#
# # 创建一个 NumPy 数组
# arr = [[1,3],[2,3],[3,3],[1,3]]
# arr = np.array(arr)
# # 打乱数组的元素
# np.random.shuffle(arr)
#
# # 输出打乱后的数组
# print(arr)
