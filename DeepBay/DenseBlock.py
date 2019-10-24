import tensorflow as tf

class DenseBlock(tf.keras.Model):
  """
  Dense Block

  Features Batch Normalization, Dropout and Dense layers, in that order
  Created for convenient building of Complex Networks

  Args:
    units (int): Number of units on Dense Layer
		activation (str): Activation of Dense layer, default="relu"
    dropout (float): % of inputs to drop from Batch Normalization Layer
    l2 (float): Strenght of L2 regularization on Dense Layer

  Call:
    inputs (float): Rank-2 Tensor of shape [Batch_Size, Input_Size]

  Return:
    Rank-2 Tensor of shape [Batch_Size, units]

  """

  def __init__(self, units, activation="relu", dropout=0.1, l2=1e-6):
    super(DenseBlock, self).__init__()

    self.bn = tf.keras.layers.BatchNormalization()
    self.drop = tf.keras.layers.Dropout(dropout)
    self.dense = tf.keras.layers.Dense(units, 
                                      activation, 
                                      kernel_regularizer=tf.keras.regularizers.L1L2(l2=l2), 
                                      kernel_constraint=tf.keras.constraints.UnitNorm())
    
  def call(self, inputs):
    """
		Call Model Feed Forward Run
		
		Args:
			inputs (float): Rank-2 Tensor of shape [Batch_Size, Input_Size]

		Return:
			Rank-2 Tensor of shape [Batch_Size, units]

		"""
    X = self.bn(inputs)
    X = self.drop(X)
    X = self.dense(X)
    return X