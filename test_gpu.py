import torch

print(torch.cuda.is_available()) ####### SHOULD RETURN TRUE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device) ### should be something like "cuda:0"