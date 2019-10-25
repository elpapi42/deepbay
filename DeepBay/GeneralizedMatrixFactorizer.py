import tensorflow as tf
from DeepBay.DenseBlock import DenseBlock

class GeneralizedMatrixFactorizer(tf.keras.Model):
  """
  Generalized Matrix Factorization Model 

  Element-wise Product of Embeddings instead of Dot Product
  Non-Linear Activation Capability

  Args:
    alpha_dim (int): Number of rows of alpha Embedding Matrix, on movie recommend, can be the max number of users
    beta_dim (int): Number of rows of beta Embedding Matrix, on movie recommend, can be the max number of movies
    latent_dim (int): Dimension of the latent space representation of both Embedding Layers
    output_dim (int): Dimension of the model output
    output_activation (str): Activation function to be used on the last layer of the model, default="sigmoid"
    dropout (float): % of inputs to drop before output dense layer
    l2 (float): Strenght of L2 regularization on embeddings and output dense layer
    use_bias (bool): If allow the model to use bias for embeddings, default=False
    alpha_key (str): Key name in the call() input dictionary assigned to alpha, default="alpha"
    beta_key (str): Key name in the call() input dictionary assigned to beta, default="beta"

  """

  def __init__(self, 
               alpha_dim, 
               beta_dim, 
               latent_dim,
               output_dim,
               output_activation="sigmoid",
               dropout=0.1, 
               l2=1e-5, 
               use_bias=False,
               alpha_key="alpha", 
               beta_key="beta"):
    
    super(GeneralizedMatrixFactorizer, self).__init__()

    self.use_bias = use_bias
    self.alpha_key = alpha_key
    self.beta_key = beta_key

    self.alpha_emb = tf.keras.layers.Embedding(alpha_dim, latent_dim, 
                                               embeddings_regularizer=tf.keras.regularizers.L1L2(l2=l2))
    
    self.beta_emb = tf.keras.layers.Embedding(beta_dim, latent_dim, 
                                              embeddings_regularizer=tf.keras.regularizers.L1L2(l2=l2))

    if(use_bias):
      self.alpha_bias = tf.keras.layers.Embedding(alpha_dim, 1)
      self.beta_bias = tf.keras.layers.Embedding(beta_dim, 1)
    
    self.flat = tf.keras.layers.Flatten()
    self.mul = tf.keras.layers.Multiply()
    self.out = DenseBlock(output_dim, output_activation, dropout, l2)

  def call(self, inputs):
    """Model Call

    Args:
      inputs (dict): Python dictionary with two keys, one for alpha and one for beta

    Return:
      Model output using current weights

    """
    alpha_emb = self.flat(self.alpha_emb(inputs[self.alpha_key]))
    beta_emb = self.flat(self.beta_emb(inputs[self.beta_key]))
    X = self.mul([alpha_emb, beta_emb])

    if(self.use_bias):
      alpha_bias = self.flat(self.alpha_bias(inputs[self.alpha_key]))
      beta_bias = self.flat(self.beta_bias(inputs[self.beta_key]))
      bias = tf.add(alpha_bias, beta_bias)
      X = tf.add(X, bias)

    X = self.out(X)
    return X