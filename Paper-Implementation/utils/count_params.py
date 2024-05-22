# Count the number of parameters of the model
import torch

def count_parameters(model):

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params
    
