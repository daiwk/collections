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

import tensorflow as tf

# TensorFlow中的inf
inf_tf = tf.constant(float('inf'))
x_inf_tf = tf.constant(float('-inf'))

# PyTorch中的inf
inf_torch = torch.tensor(float('inf'))
x_inf_torch = torch.tensor(float('-inf'))

print(inf_tf, x_inf_tf, inf_torch, x_inf_torch)
