import torch
import torch.nn as nn
from model_structure import NN
from helper_code import finetune_model_prep
import copy

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def test_conv_equiv():
    model = NN(24).to(DEVICE)
    model.eval()
    
    old_conv = model.conv
    
    # 1. Adapt conv
    new_conv = nn.Conv2d(in_channels=24,
                         out_channels=old_conv.out_channels,
                         kernel_size=old_conv.kernel_size,
                         stride=old_conv.stride,
                         padding=old_conv.padding,
                         bias=False).to(DEVICE)
                         
    with torch.no_grad():
        new_conv.weight[:, :12, :, :] = old_conv.weight
        new_conv.weight[:, 12:, :, :] = 0.0
        
    # 2. Mock input
    x = torch.randn(2, 12, 1, 1000).to(DEVICE)
    x_bcos = torch.cat([x, 1.0 - x], dim=1)
    
    # 3. Compare outputs
    with torch.no_grad():
        out_orig = old_conv(x)
        out_new = new_conv(x_bcos)
        
    diff = torch.max(torch.abs(out_orig - out_new)).item()
    print(f"Max difference between old conv and adapted conv: {diff}")
    assert diff < 1e-5, "Conv conversion failed!"
    print("Conv conversion successful! Output is identical.")
    
def test_full_prep():
    model = NN(24).to(DEVICE)
    model.eval()
    
    bcos_model = copy.deepcopy(model)
    bcos_model = finetune_model_prep(bcos_model)
    bcos_model.eval()
    
    x = torch.randn(2, 12, 1, 1000).to(DEVICE)
    l = torch.ones(2, 12).to(DEVICE)
    
    x_bcos = torch.cat([x, 1.0 - x], dim=1)
    l_bcos = torch.cat([l, l], dim=1)
    
    with torch.no_grad():
        out_bcos = bcos_model(x_bcos, l_bcos)
        
    print(f"B-cos model output shape: {out_bcos.shape}")
    print("Full prep run successfully.")
    
if __name__ == "__main__":
    test_conv_equiv()
    test_full_prep()
