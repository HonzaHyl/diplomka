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
    def __init__(self, nOUT, dropout_rate=0.5):
        super(NN, self).__init__()
        self.dropout_rate = dropout_rate
        # Names and shapes remain EXACTLY the same
        self.conv = nn.Conv2d(in_channels=12, out_channels=256, kernel_size=(1, 15),
                              padding=(0, 7), stride=(1, 2), bias=False)
        self.bn = nn.BatchNorm2d(256)
        
        self.rb_0 = MyResidualBlock(downsample=True)
        self.rb_1 = MyResidualBlock(downsample=True)
        self.rb_2 = MyResidualBlock(downsample=True)
        self.rb_3 = MyResidualBlock(downsample=True)
        self.rb_4 = MyResidualBlock(downsample=True)

        self.pool_avg = nn.AdaptiveAvgPool1d(output_size=1)
        self.pool_max = nn.AdaptiveMaxPool1d(output_size=1)

        # Simple linear classification head: 256 (avg) + 256 (max) + 12 (leads)
        self.head = nn.Linear(256 * 2 + 12, nOUT)

    def forward(self, x, l, r=None):
        x = F.leaky_relu(self.bn(self.conv(x)))

        x = self.rb_0(x)
        x = self.rb_1(x)
        x = self.rb_2(x)
        x = self.rb_3(x)
        x = self.rb_4(x)

        # Spatial Dropout (Dropout2d drops entire 2D feature maps/channels)
        x = F.dropout2d(x, p=getattr(self, 'dropout_rate', 0.5), training=self.training)
        x = x.squeeze(2)
        
        # Apply average pooling to capture global statistics (general rhythm, RMS)
        x_avg = self.pool_avg(x).squeeze(2) # Shape: [Batch, 256]

        # Apply max pooling to capture presence of specific morphological features (like P-waves)
        x_max = self.pool_max(x).squeeze(2) # Shape: [Batch, 256]

        # Concatenate: global features + max features + lead mask + rhythm (if available)
        if r is not None:
            x = torch.cat((x_avg, x_max, l, r), dim=1)  # [Batch, 526]
        else:
            x = torch.cat((x_avg, x_max, l), dim=1)  # [Batch, 524]
        x = self.head(x)
        
        return x

class EnsembleNN(nn.Module):
    def __init__(self, nOUT, num_models=4, use_rhythm=True):
        super(EnsembleNN, self).__init__()
        self.models = nn.ModuleList([NN(nOUT=nOUT) for _ in range(num_models)])
        if use_rhythm:
            for m in self.models:
                in_features = m.head.in_features + 2
                m.head = nn.Linear(in_features, nOUT)
        
    def forward(self, x, l, r=None):
        # Get probabilities from all models
        probs = [torch.softmax(model(x, l, r), dim=1) for model in self.models]
        # Average probabilities
        avg_probs = torch.stack(probs, dim=0).mean(dim=0)
        
        # Return log probabilities so that applying softmax later 
        # (as in test_model.py) recovers the averaged probabilities: softmax(log(p)) = p
        epsilon = 1e-8
        log_probs = torch.log(avg_probs + epsilon)
        return log_probs