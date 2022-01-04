import dataclasses
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass, make_dataclass
from dataclasses_json import dataclass_json
import pickle
import json

Ped_sample = make_dataclass("Hircus_Sample", [('Actions', float), ('Ground_Truth', float), ('Image_name', str)])
# df = pd.read_csv('Herdr_data.csv', names=["Actions", "Ground_Truth", "Image_Name"])
a = torch.zeros((50, 20, 2))
v = np.ones((50, 20, 1))
s = np.ones((50, 20, 1))
gt = torch.zeros((5, 2, 1))
imn = 'one'
ls = Ped_sample(a, gt, imn)

# df = pd.read_pickle("test.txt")
# df = df.append(df, ignore_index=True)
df = pd.DataFrame([ls], columns=["Actions", "Ground_Truth", "Image_Name"])
df.to_pickle("test.txt")

df = pd.read_pickle("test.txt")
print(df["Actions"].values[0].shape)

# with open("test.txt", "ab") as f:
#     pickle.dump(ls, f)
#
# with open("test.txt", "rb") as f:
#     ls = pickle.load(f)
# print(len(ls))
# t = np.array([v.flatten(), s.flatten(), gt.flatten()]).T
# df1 = pd.DataFrame(t, columns=["Velocity", "Steering", "Gt"])
# df2 = pd.DataFrame([imn], columns=["Image_name"])
# result = pd.concat([df1, df2], axis=1)
# result.to_csv('test.csv', mode='a', header=False)
#
# test_in = pd.read_csv('test.csv', names=["Velocity", "steering", "Ground_Truth", "Image_Name"])



