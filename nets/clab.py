import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

class CLAB_BLOCK(nn.Module):
    def __init__(self,in_channels,p_block,k_size,size):
        super(CLAB_BLOCK, self).__init__()
        self.size = size
        self.layers = nn.ModuleList([nn.Conv2d(in_channels, k_size, kernel_size=1,
                                          stride=1, padding=0, bias=True) for i in range(p_block)])
        self.conv_sig = nn.Sequential(
            nn.Conv2d(p_block*k_size ,in_channels , kernel_size=1,stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        self.ln = Linear(p_block,p_block)
        self.norm = nn.BatchNorm2d(num_features = in_channels)

    def forward(self,x):
        print('x',x.shape)
        x_temp = []
        for layer in self.layers:
            print('2', layer(x).shape)
            branch = layer(x)

            branch = self.relu(branch)
            branch = torch.mean(branch,dim=1, keepdim=True)
            # b,h,w = branch.size()
            # branch = branch.view(b,1,h,w)
            branch = self.norm(branch)
            x_temp.append(branch)
        output = torch.cat(x_temp, dim=1)
        output = nn.AvgPool2d(kernel_size=self.size, stride=1, padding=0)(output)
        b,c,h,w = output.size()
        output = output.view(b,c)
        output = self.ln(output)
        output = output.view(b,1,1,p_block)
        output = self.conv_sig(output)
        output = output * x
        return output