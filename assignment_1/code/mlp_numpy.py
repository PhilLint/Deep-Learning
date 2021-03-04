"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *

class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes, neg_slope):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        # list of all layers
        self.layers = []

        # len(n_hidden) == 0:final layer -> only Linear Module with SoftMax
        current_input = n_inputs
        if len(n_hidden) > 0:
            for out_features in n_hidden:
                # LinearModule needs in_put and out_put dimensions
                self.layers.append(LinearModule(current_input, out_features))
                self.layers.append(LeakyReLUModule(neg_slope))
                current_input = out_features

        # after appending all hidden layers (or not if n_hidden = 0)
        # -> 1 LinearModule followed by softmax = Logistic regression
        self.layers.append(LinearModule(current_input, n_classes))
        self.layers.append(SoftMaxModule())

        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        # each element in self.layers is Module object and thus has function .forward
        # -> loop over layers and apply forward firstly on x then on output
        for module in self.layers:
            x = module.forward(x)
        out = x
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        # same as forward but other direction
        loss = dout
        for module in reversed(self.layers):
            loss = module.backward(loss)
        ########################
        # END OF YOUR CODE    #
        #######################

        return
