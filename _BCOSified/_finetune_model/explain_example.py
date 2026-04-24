import torch
import matplotlib.pyplot as plt

def generate_explanation(model, input_tensor, target_class=0):
    """
    Generates a B-cos textual/visual explanation for the target class.
    
    Args:
        model: Your B-cosified NN model
        input_tensor: A batch of inputs, shape (B, 24, 1, L)
        target_class: The output index you want to explain (e.g., 0 or 1)
        
    Returns:
        explanation_map: The absolute contribution of each input feature to the target class
    """
    # 1. We must turn on `requires_grad` for the input to track the gradients
    input_tensor.requires_grad_(True)
    
    # Optional: ensure we are using the 'explanation mode' which detaches the dynamic scaling
    # We didn't explicitly add `explanation_mode()` to our wrappers, but in original B-cos,
    # the scaling factor is detached during backpropagation to only explain the linear part.
    # For a simplified explanation map, standard gradients are often sufficient.
    
    model.eval()
    
    # 2. Forward pass: l is just an example scalar tensor from your architecture
    l = torch.ones(input_tensor.shape[0], 12).to(input_tensor.device)
    output = model(input_tensor, l)
    
    # 3. isolate the target class score
    score = output[:, target_class].sum()
    
    # 4. Backward pass to get the linear mapping (gradients)
    model.zero_grad()
    score.backward()
    
    # 5. The explanation is exactly the gradient * input for a linear model!
    # Because B-cos networks form a single dynamic linear transformation: output = W_dynamic(x) * x
    # The gradient is precisely W_dynamic(x).
    explanations = input_tensor.grad * input_tensor
    
    # 6. Sum across the channel dimension to get the contribution of each time step
    # We sum across all 24 channels to see which time-steps drove the prediction
    explanation_map = explanations.sum(dim=1).squeeze()
    
    return explanation_map

def plot_explanation(original_signal, explanation_map):
    """
    Plots the original ECG signal colored by the explanation map.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Plot original signal (e.g., Lead I)
    time_steps = torch.arange(original_signal.shape[-1])
    ax1.plot(time_steps, original_signal[0].cpu().detach().numpy())
    ax1.set_title("Original ECG Signal (Lead I)")
    
    # Plot explanation weight
    ax2.fill_between(time_steps, 0, explanation_map.cpu().detach().numpy(), color='red', alpha=0.5)
    ax2.set_title("B-cos Explanation Map (Contribution to Class)")
    
    plt.tight_width()
    plt.show()

# Example Usage:
# expl_map = generate_explanation(model, x_bcos, target_class=1)
# plot_explanation(x_orig[0, 0], expl_map[0])
