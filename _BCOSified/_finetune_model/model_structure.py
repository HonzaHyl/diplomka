import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest

class MyResidualBlock(nn.Module):
    def __init__(self,downsample):
        super(MyResidualBlock,self).__init__()
        self.downsample = downsample
        self.stride = 2 if self.downsample else 1
        K = 9
        P = (K-1)//2
        self.conv1 = nn.Conv2d(in_channels=256,
                               out_channels=256,
                               kernel_size=(1,K),
                               stride=(1,self.stride),
                               padding=(0,P),
                               bias=True)

        self.conv2 = nn.Conv2d(in_channels=256,
                               out_channels=256,
                               kernel_size=(1,K),
                               padding=(0,P),
                               bias=True)

        if self.downsample:
            self.idfunc_0 = nn.AvgPool2d(kernel_size=(1,2),stride=(1,2))
            self.idfunc_1 = nn.Conv2d(in_channels=256,
                                      out_channels=256,
                                      kernel_size=(1,1),
                                      bias=True)





    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.downsample:
            identity = self.idfunc_0(identity)
            identity = self.idfunc_1(identity)

        x = x+identity
        return x






class NN(nn.Module):
    def __init__(self, nOUT, dropout_rate=0.5):
        super(NN, self).__init__()
        self.dropout_rate = dropout_rate
        # Names and shapes remain EXACTLY the same
        self.conv = nn.Conv2d(in_channels=12, out_channels=256, kernel_size=(1, 15),
                              padding=(0, 7), stride=(1, 2), bias=True)
        
        self.rb_0 = MyResidualBlock(downsample=True)
        self.rb_1 = MyResidualBlock(downsample=True)
        self.rb_2 = MyResidualBlock(downsample=True)
        self.rb_3 = MyResidualBlock(downsample=True)
        self.rb_4 = MyResidualBlock(downsample=True)

        # Temporal fully convolutional head: 256 (latent) + 12 (leads) + 2 (rhythm)
        self.head = nn.Conv1d(256 + 12 + 2, nOUT, kernel_size=1)

    def forward(self, x, l, r):
        x = self.conv(x)

        x = self.rb_0(x)
        x = self.rb_1(x)
        x = self.rb_2(x)
        x = self.rb_3(x)
        x = self.rb_4(x)

        # Spatial Dropout (Dropout2d drops entire 2D feature maps/channels)
        x = F.dropout2d(x, p=self.dropout_rate, training=self.training)
        x = x.squeeze(2) # Shape: [Batch, 256, Time]
        
        # Expand context vectors along time dimension
        time_steps = x.size(-1)
        l_expanded = l.unsqueeze(-1).expand(-1, -1, time_steps) # [Batch, 12, Time]
        r_expanded = r.unsqueeze(-1).expand(-1, -1, time_steps) # [Batch, 2, Time]

        # Concatenate: latent features + lead mask + rhythm vector
        x = torch.cat((x, l_expanded, r_expanded), dim=1)  # [Batch, 270, Time]
        
        # Get predictions for every time step
        step_predictions = self.head(x) # [Batch, nOUT, Time]
        
        # Apply temporal aggregation to predictions
        x = F.adaptive_avg_pool1d(step_predictions, 1).squeeze(2) # [Batch, nOUT]
        
        return x

class EnsembleNN(nn.Module):
    def __init__(self, nOUT, num_models=4, dropout_rate=0.5):
        super(EnsembleNN, self).__init__()
        self.models = nn.ModuleList([NN(nOUT=nOUT, dropout_rate=dropout_rate) for _ in range(num_models)])
        
    def forward(self, x, l, r):
        # Get probabilities from all models
        probs = [torch.softmax(model(x, l, r), dim=1) for model in self.models]
        # Average probabilities
        avg_probs = torch.stack(probs, dim=0).mean(dim=0)
        
        # Return log probabilities so that applying softmax later 
        # (as in test_model.py) recovers the averaged probabilities: softmax(log(p)) = p
        epsilon = 1e-8
        log_probs = torch.log(avg_probs + epsilon)
        return log_probs

class test(unittest.TestCase):
    def setUp(self) -> None:
        pass
    def test_0(self):
        x = torch.rand(64,12,1,8192)
        l = torch.ones(64,12)
        r = torch.ones(64,2)
        mdl = NN(24)
        y  = mdl(x,l,r)
    def test_1(self):
        x = torch.rand(1,12,1,8192)
        l = torch.ones(1,12)
        r = torch.ones(1,2)
        mdl = NN(24)
        mdl.eval()
        y  = mdl(x,l,r)