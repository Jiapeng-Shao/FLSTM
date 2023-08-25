import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

import torch.optim as optim
from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

all_sample = 2819 # 69680 17420 SH 34039 BJ 50376 SUN 2819 MAX 3650

X = np.linspace(1, all_sample, all_sample)

data = pd.read_csv('C:\\data\\SUN_1.csv')
# 将data变成为tensor类型
# 当表中元素都为数值时，这个.values会将csv的表中的第一行（属性），第一列（序号）去除
data = data.values.astype(float)  # numpy强制类型转换
data = torch.tensor(data,device="cpu") #'cpu' "cuda:0"

data = data.view(1, all_sample)
data = data.view(all_sample,1)
datayuan = data.cpu().numpy()
print(datayuan.shape)
scaler = MinMaxScaler(feature_range=(0, 1)).fit(data.cpu().numpy())
datafit= scaler.transform(data.cpu())
datafit= datafit.reshape(all_sample)

#plt.plot(X,datafit)
plt.plot(X,data)
plt.legend(["ETTm2"])
plt.show()