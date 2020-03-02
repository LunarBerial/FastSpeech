#coding:utf-8

import re, codecs
import torch
import torch.nn as nn
import numpy as np
x = torch.randn(10, 3)
y = torch.randn(10, 2).view((2, 5, 2))
print(y.data, y.shape)
linear = nn.Linear(3, 2)