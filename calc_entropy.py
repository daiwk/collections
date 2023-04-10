import tensorflow as tf
import math

sigmoid_logits = tf.constant([1., -1., 0.])
softmax_logits = tf.stack([sigmoid_logits, tf.zeros_like(sigmoid_logits)],
                          axis=-1)
soft_binary_labels = tf.constant([1., 1., 0.])
soft_multiclass_labels = tf.stack(
    [soft_binary_labels, 1. - soft_binary_labels], axis=-1)
hard_labels = tf.constant([0, 0, 1])

l1 = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=hard_labels, logits=softmax_logits).numpy()

l2 = tf.nn.softmax_cross_entropy_with_logits(
    labels=soft_multiclass_labels, logits=softmax_logits).numpy()

l3 = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=soft_binary_labels, logits=sigmoid_logits).numpy()

print(l1)
print(l2)
print(l3)

def sigmoid(x):
  return 1. / (1 + math.exp(-x))

def sigmoid_cross_entropy_with_logits(labels, logits):
  loss = 0.  
  for idx in range(0, len(logits)):
    logit = logits[idx]
    x = sigmoid(logit)
    label = labels[idx]
    xloss = -(label * math.log(x) + (1- label) * math.log(1-x))
    print(xloss)

  return loss

logits = [1, -1, 0]
labels = [1, 1, 0]
print("cross_entropy")

lxx = sigmoid_cross_entropy_with_logits(labels, logits)
#print(lxx)

print("softmax....")

def softmax_cross_entropy_with_logits(labels, logits):
  loss = 0. 
  alst = [] 
  for idx in range(0, len(logits)):
    logit = logits[idx]
    x = sigmoid(logit)
    x = logit
    label = labels[idx]
    x_exp = math.exp(x)
    alst.append(x_exp)
  fenmu = sum(alst)
  for idx in range(0, len(alst)):
    xx = alst[idx]
    label = labels[idx]
    x = xx/fenmu
    xloss = -(label * math.log(x))
    print(xloss)
  
logits = [4., 2., 1.]
labels = [1, 0, 0]

lxx = softmax_cross_entropy_with_logits(labels, logits)
print("xxxxxx")

logits = [0., 5., 1.]
labels = [0, 0.8, 0.2]

lxx = softmax_cross_entropy_with_logits(labels, logits)
