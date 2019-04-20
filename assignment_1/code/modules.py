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
    self.params = {'weight': None, 'bias': None}
    self.grads = {'weight': None, 'bias': None}
    self.in_features = in_features
    self.out_features = out_features

    # as many bias terms as output features / units
    bias = np.zeros(out_features)
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
    # storage for in and outputs in forward and backward pass
    self.x = None
    self.out = None
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
    self.x = x
    # output = x*weights^T + bias / dimensions: 1xout = 1 x in * in x out + 1 x out -> matches
    out = x @ self.params['weight'].T + self.params['bias']
    self.out = out

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

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
    self.grads['weight'] = dout.T @ self.x
    self.grads['bias'] = np.sum(dout, axis=0)
    # dimensions: 1 x in = 1  x out * out x in -> matches
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
    # store output of relu
    self.x = out
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
    dx = dout @ (self.x > 0).astype(int)
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
    # softmax with max trick
    b = x.max(axis=1, keepdims=True)
    exp_x = np.exp(x-b)
    out = exp_x / exp_x.sum(axis=1, keepdims=True)
    self.x = out
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
    # for each batch separately
    batch_size = self.x.shape[0] # how many observations in one batch
    dim_out = self.x.shape[1] # output dimension of features

    # create #batch_size diagonal matrices with the diagonal elements
    # of x[1,:,:], .., x[batch_size, :,:] in it by multiplying with identity
    diagonals = np.vsplit(self.x, batch_size) * np.eye(dim_out)
    # for each batch element: diagonal - x*x^T (batch wise -> einsum)
    dx_dtildex = diagonals - np.einsum('ij, ik -> ijk', self.x, self.x)
    # multiply dout batchwise to dx_tildex, which is also batchwise
    dx = np.einsum('ij, ijk -> ik', dout, dx_dtildex)
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
    out = np.mean(np.sum(-y * np.log(x + 1e-6), axis = 1))
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
    dx = (1/batch_size) * (-y * (1/(x+ 1e-6))
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx
