import torch
import torch.nn as nn
import torch.nn.functional as F

class BcosifiedConv2d(nn.Module):
    def __init__(self, conv2d_module):
        super().__init__()
        self.conv = conv2d_module
        # Biases are kept and utilized in self.conv(x)
        
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.kssq = self.kernel_size[0] * self.kernel_size[1]
        self.in_channels = self.conv.in_channels

    def forward(self, x):
        # 1. Compute standard linear activation
        out = self.conv(x)
        
        # 2. Compute norm of weights (since they are unnormalized)
        w = self.conv.weight
        w_norm = w.view(w.size(0), -1).norm(p=2, dim=1).view(-1, 1, 1) + 1e-6
        
        # 3. Compute norm of input patches 
        x_norm_sq = (x ** 2).sum(dim=1, keepdim=True)
        # avg_pool2d computes sum / kssq, so we multiply by kssq to get sum of squares
        x_patch_norm = (F.avg_pool2d(x_norm_sq, self.kernel_size, padding=self.padding, stride=self.stride) * self.kssq + 1e-6).sqrt_()
        
        # 4. Compute B=2 output multiplier: 
        # Standard B=1 linear activation is: out. For B=2, we need: norm(x) * norm(W) * cos^2(theta) * sign(cos_theta)
        # cos_theta = out / (x_patch_norm * w_norm)
        # Output = (norm(x) * norm(W)) * (out / (x_patch_norm * w_norm)) * |out / (x_patch_norm * w_norm)|
        # Output = out * |out| / (x_patch_norm * w_norm)
        
        bcos_out = out * out.abs() / (x_patch_norm * w_norm)
        
        return bcos_out

class BcosifiedLinear(nn.Module):
    def __init__(self, linear_module):
        super().__init__()
        self.linear = linear_module
        # Biases are kept and utilized in self.linear(x)
        
    def forward(self, x):
        # 1. Compute standard linear activation
        out = self.linear(x)
        
        # 2. Compute norm of weights
        w = self.linear.weight
        w_norm = w.norm(p=2, dim=1).unsqueeze(0) + 1e-6
        
        # 3. Compute norm of inputs
        x_norm = x.norm(p=2, dim=1, keepdim=True) + 1e-6
        
        # 4. Compute B=2 logic
        bcos_out = out * out.abs() / (x_norm * w_norm)
        
        return bcos_out

def bcosify_model(model):
    """
    Recursively replaces all Conv2d and Linear layers in the model with their Bcosified equivalents.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            setattr(model, name, BcosifiedConv2d(module))
        elif isinstance(module, nn.Linear):
            setattr(model, name, BcosifiedLinear(module))
        else:
            bcosify_model(module)
