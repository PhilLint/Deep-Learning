"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module.
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # initialize dicts
    self.params = {'weight': None, 'bias': None}
    self.grads = {'weight': None, 'bias': None}

    # as many bias terms as output features / units: dimension: (out, in)
    bias = np.zeros((out_features, 1))
    # random weights with mu = 0 sd = 0.0001 and as weight matrix is out_dim times in_dim
    # in_dim * out_dim samples have to be drawn (dimensions from assignment sheet)
    weights = np.random.normal(loc = 0, scale = 0.0001, size = (out_features, in_features))
    # zero initializated gradients
    gradients = np.zeros((out_features, in_features))
    # inituíalization
    self.params['bias'] = bias
    self.params['weight'] = weights
    self.grads['bias'] = bias
    self.grads['weight'] = gradients

    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.input = x
    # x has dim (1,in)
    # output = weights*x.T + bias / dimensions: outx1 = out x in * in x 1 + outx1 -> matches
    out = self.params['weight'] @ x.T + self.params['bias']
    ########################
    # END OF YOUR CODE    #
    #######################
    # transpose to be row vector as required
    return out.T

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # dimensions: out x in = out x 1 * 1 x in -> matches

    self.grads['weight'] = dout.T @ self.input
    self.grads['bias'] = np.reshape(np.sum(dout, axis=0), self.grads['bias'].shape)
    # dimensions: batch_size x in = batch_size  x out * out x in -> matches

    dx = dout @ self.params['weight']
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = np.maximum(0, x)
    # store inout of relu
    self.input = x
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # boolean condition > 0: if x > 0 then gradient 1, else 0
    # type conversion ensures 1; 0 values
    dx = dout * (self.input > 0).astype(int)
    ########################
    # END OF YOUR CODE    #
    #######################    

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # softmax with max trick for stability
    b = x.max(axis=1, keepdims=True)
    exp_x = np.exp(x-b)
    self.output = exp_x / exp_x.sum(axis=1, keepdims=True)


    ########################
    # END OF YOUR CODE    #
    #######################

    return self.output

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    # for each batch separately
    batch_size = self.output.shape[0] # how many observations in one batch
    dim_out = self.output.shape[1] # output dimension of features

    # create #batch_size diagonal matrices with the diagonal elements
    # of x[1,:,:], .., x[batch_size, :,:] in it by multiplying with identity
    diagonals = np.vsplit(self.output, batch_size) * np.eye(dim_out)
    # for each batch element: diagonal - x*x^T (batch wise -> einsum)
    dx_dtildex = diagonals - np.einsum('ij, ik -> ijk', self.output, self.output)
    # multiply dout batchwise to dx_tildex, which is also batchwise
    dx = np.einsum('ij, ijk -> ik', dout, dx_dtildex)
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    
    TODO:
    Implement forward pass of the module. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # to avoid log(0) issues: small term is added: 1e-6
    # according to given minibatch Loss: average over individual losses
    out = np.mean(np.sum((-1)* y * np.log(x + 1e-6), axis = 1))
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    batch_size = x.shape[0]
    dx = (1/batch_size) * np.divide(-y,x + 1e-6)
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

