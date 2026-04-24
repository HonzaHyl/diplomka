import sys
import os
import torch

# Ensure the script can import from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_structure import NN

def main():
    # Instantiate the model. Using nOUT=24 as seen in your model_structure.py tests.
    model = NN(nOUT=24)
    
    print("========== ALL LAYER NAMES AND THEIR TYPES ==========\n")
    # named_modules() recursively yields all modules (layers) in the network
    for name, module in model.named_modules():
        if name == "":
            continue # Skip the root model container itself
        
        # Formatting the output for readability
        print(f"{name:<20} -> {module.__class__.__name__}")
        
    print("\n\n========== DETAILED MODEL STRUCTURE ==========\n")
    # Printing the model directly also gives a nice hierarchical overview
    print(model)

if __name__ == "__main__":
    main()
