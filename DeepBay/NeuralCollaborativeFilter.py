import tensorflow as tf
from DeepBay.DenseBlock import DenseBlock
from DeepBay.DeepFeedForward import DeepFeedForward

class NeuralCollaborativeFilter(tf.keras.Model):
  """
  Neural Collaborative Filtering, Extending Matrix Factorization to a DNN Context 
  taking adventage of non-linearities and Deep Dense Structures

  Args:
    alpha_dim (int): Number of rows of alpha Embedding Matrix, on movie recommend, can be the max number of users
    beta_dim (int): Number of rows of beta Embedding Matrix, on movie recommend, can be the max number of movies
    latent_dim (int): Dimension of the latent space representation of both Embedding Layers
    layers (int): Rank-1 array-like object describing the hiden units in each layer of the MLP
    output_activation (str): Activation function to be used on the last layer of the model, default="sigmoid"
    dropout (float): % of inputs to drop from Batch Normalization Layer
    l2 (float): Strenght of L2 regularization on Dense Layer
    use_bias (bool): If allow the model to use bias, default=False
    alpha_key (str): Key name in the call() input dictionary assigned to alpha, default="alpha"
    beta_key (str): Key name in the call() input dictionary assigned to beta, default="beta"

  """

  def __init__(self, 
               alpha_dim, 
               beta_dim, 
               latent_dim, 
               layers, 
               output_activation="sigmoid",
               dropout=0.1, 
               l2=0.001, 
               use_bias=False,
               alpha_key="alpha", 
               beta_key="beta"):
    
    super(NeuralCollaborativeFilter, self).__init__()

    self.alpha_key = alpha_key
    self.beta_key = beta_key
    self.use_bias = use_bias

    self.alpha_emb = tf.keras.layers.Embedding(alpha_dim, latent_dim, embeddings_regularizer=tf.keras.regularizers.L1L2(l2=l2))
    self.beta_emb = tf.keras.layers.Embedding(beta_dim, latent_dim, embeddings_regularizer=tf.keras.regularizers.L1L2(l2=l2))

    if(use_bias):
      self.alpha_bias = tf.keras.layers.Embedding(alpha_dim, 1)
      self.beta_bias = tf.keras.layers.Embedding(beta_dim, 1)
    
    self.flat = tf.keras.layers.Flatten()
    self.feedforward = DeepFeedForward(layers[:-1], dropout, l2)
    self.dense_out = DenseBlock(layers[-1], output_activation, dropout, l2)

  def call(self, inputs):
    """Model Call

    Args:
      inputs (dict): Python dictionary with two keys, one for alpha and one for beta

    Return:
      Model output using current weights

    """

    alpha_emb = self.flat(self.alpha_emb(inputs[self.alpha_key]))
    beta_emb = self.flat(self.beta_emb(inputs[self.beta_key]))
    X = tf.concat([alpha_emb, beta_emb], axis=-1)

    if(self.use_bias):
      alpha_bias = self.flat(self.alpha_bias(inputs[self.alpha_key]))
      beta_bias = self.flat(self.beta_bias(inputs[self.beta_key]))
      bias = tf.add(alpha_bias, beta_bias)
      X = tf.add(X, bias)

    X = self.feedforward(X)
    X = self.dense_out(X)
    return X