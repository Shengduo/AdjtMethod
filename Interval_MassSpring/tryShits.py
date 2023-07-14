import torch
a = torch.tensor([torch.tensor(2.0), torch.tensor(1.0), torch.tensor(-1.0), torch.tensor(4.0)])
sorted, indices = torch.sort(a, descending=True)
print(a)
print(sorted)
print(indices)