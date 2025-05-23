 # -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.clab import CLAB_BLOCK

'''pixel-level module'''


class PixLevelModule(nn.Module):
    def __init__(self, in_channels):
        super(PixLevelModule, self).__init__()
        self.middle_layer_size_ratio = 2
        self.conv_avg = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.relu_avg = nn.ReLU(inplace=True)
        self.conv_max = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.relu_max = nn.ReLU(inplace=True)
        self.bottleneck = nn.Sequential(
            nn.Linear(3, 3 * self.middle_layer_size_ratio),  # 2, 2*self.
            nn.ReLU(inplace=True),
            nn.Linear(3 * self.middle_layer_size_ratio, 1)
        )
        self.conv_sig = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.channel_attention = CLAB_BLOCK(in_channels, p_block=3, k_size=in_channels, size=3)

    '''forward'''

    def forward(self, x):
        x_avg = self.conv_avg(x)
        x_avg = self.relu_avg(x_avg)
        x_avg = torch.mean(x_avg, dim=1)
        x_avg = x_avg.unsqueeze(dim=1)
        x_max = self.conv_max(x)
        x_max = self.relu_max(x_max)
        x_max = torch.max(x_max, dim=1).values
        x_max = x_max.unsqueeze(dim=1)

        x_avg_p = F.max_pool2d(x_avg, kernel_size=(x_avg.size()[2], x_avg.size()[3]))
        x_max_p = F.max_pool2d(x_max, kernel_size=(x_max.size()[2], x_max.size()[3]))
        x_out = x_max_p+x_avg_p
        # x_out = x_max+x_avg

        x_out_p = F.max_pool2d(x_out, kernel_size=(x_out.size()[2], x_out.size()[3]))

        x_output = torch.cat((x_avg_p, x_max_p, x_out_p), dim=1)

        x_output = x_output.transpose(1, 3)
        x_output = self.bottleneck(x_output)
        x_output = x_output.transpose(1, 3)
        y = x_output * x
        return y
