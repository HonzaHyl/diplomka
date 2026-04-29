import torch
import torch.nn as nn
import torch.nn.functional as F

class MyResidualBlock(nn.Module):
    def __init__(self, downsample):
        super(MyResidualBlock, self).__init__()
        self.downsample = downsample
        self.stride = 2 if self.downsample else 1
        K = 9
        P = (K - 1) // 2
        
        # Names and shapes remain EXACTLY the same for the state_dict
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, K),
                               stride=(1, self.stride), padding=(0, P), bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, K),
                               padding=(0, P), bias=False)
        self.bn2 = nn.BatchNorm2d(256)

        if self.downsample:
            self.idfunc_0 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
            self.idfunc_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), bias=False)

    def forward(self, x):
        identity = x
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        if self.downsample:
            identity = self.idfunc_0(identity)
            identity = self.idfunc_1(identity)

        x = x+identity
        return x


class NN(nn.Module):
    def __init__(self, nOUT):
        super(NN, self).__init__()
        # Names and shapes remain EXACTLY the same
        self.conv = nn.Conv2d(in_channels=12, out_channels=256, kernel_size=(1, 15),
                              padding=(0, 7), stride=(1, 2), bias=False)
        self.bn = nn.BatchNorm2d(256)
        
        self.rb_0 = MyResidualBlock(downsample=True)
        self.rb_1 = MyResidualBlock(downsample=True)
        self.rb_2 = MyResidualBlock(downsample=True)
        self.rb_3 = MyResidualBlock(downsample=True)
        self.rb_4 = MyResidualBlock(downsample=True)

        self.pool_max = nn.AdaptiveMaxPool1d(output_size=1)
        self.pool_avg = nn.AdaptiveAvgPool1d(output_size=1)

        
        # 256 (max) + 256 (avg) + 12 (leads) = 524
        self.fc_1 = nn.Linear(256 + 256 + 12, nOUT)

    def forward(self, x, l):
        x = F.leaky_relu(self.bn(self.conv(x)))

        x = self.rb_0(x)
        x = self.rb_1(x)
        x = self.rb_2(x)
        x = self.rb_3(x)
        x = self.rb_4(x)

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.squeeze(2)
        
        # Apply both poolings to capture peak and global activations
        x_max = self.pool_max(x).squeeze(2) # Shape: [Batch, 256]
        x_avg = self.pool_avg(x).squeeze(2) # Shape: [Batch, 256]
        
        # Concatenate: peak features + global features + lead mask
        x = torch.cat((x_max, x_avg, l), dim=1)  # [Batch, 524]
        x = F.dropout(x, p=0.5, training=self.training)  # Regularise the classification head
        x = self.fc_1(x)
        
        return x