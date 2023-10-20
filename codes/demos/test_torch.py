import torch
t = [[1, 2, 3], [4, 5, 6]]
x = torch.tensor(t)
print(x)
print(x[:, -10:])
print(x[:, -2:])

## tensor([[1, 2, 3],
##         [4, 5, 6]])
## tensor([[1, 2, 3],
##         [4, 5, 6]])
## tensor([[2, 3],
##         [5, 6]])
