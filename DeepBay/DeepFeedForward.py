import tensorflow as tf
from DeepBay.DenseBlock import DenseBlock

class DeepFeedForward(tf.keras.Model):
  """
  Multi-Layer Perceptron

  Args:
    layers (int): Rank-1 array-like object describing the hiden units of each layer in the Model
                  Example: [4, 2] Creates an MLP that have 4 unit on the first layer and 2 unit on the last layer
    dropout (float): Dropout rate of all the layers
    l2 (float): L2 regularization strenght of all the layers

  """

  def __init__(self, layers, dropout=0.1, l2=0.001):
    super(DeepFeedForward, self).__init__()

    self.block_list = list()
    for layer in layers:
      self.block_list.append(DenseBlock(layer, "relu", dropout, l2))
    
  def call(self, inputs):

    X = inputs
    for layer in self.block_list:
      X = layer(X)
    return X

a = DeepFeedForward([1])