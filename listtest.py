#coding : utf-8

import torch.nn as nn

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(2):
            sublinear = nn.ModuleList()
            for j in range(2):
                sublinear.append(nn.Linear(10, 10))
            self.linears.append(sublinear)

    def forward(self,x):
        for i in range(len(self.linears)):
            x = self.linears[i][0](x)

        return x


if __name__ == "__main__":
    n = net()
    print(n)