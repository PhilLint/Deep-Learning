"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '600, 300, 100'
LEARNING_RATE_DEFAULT = 0.02
MAX_STEPS_DEFAULT = 4001
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
weight_decay = 0.001
NEG_SLOPE_DEFAULT = 0.02
OPTIMIZER = "SGD"
NAME = "SGD_4000_600_decay_lr"
LR_FREQ = 400

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    class_predictions = torch.max(predictions, 1)[1]
    targets = torch.max(targets, 1)[1]

    return float(torch.sum(class_predictions == targets)) / float(predictions.shape[0])


def train():
    """
    Performs training and evaluation of MLP model.

    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # initialize required arrays for saving the results
    print(torch.cuda.is_available())
    # device = torch.device("cpu") # my gpu is not cuda conform
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_accuracies = []
    train_losses = []
    test_accuracies = []
    test_losses = []
    steps = []

    # load data from directory specified in the input
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)

    # load test images and labels
    test_images = cifar10['test'].images
    test_targets = cifar10['test'].labels

    # data dimensions
    # test_images.shape -> (10000, 3, 32, 32): n_images, channels, height, width
    # test_targets.shape <- (10000, 10): n_images, n_classes
    n_test = test_images.shape[0]
    # n_inputs is one vector for all channels of width and height
    # n_input = n_channel * width * height
    n_inputs = test_images.shape[1] * test_images.shape[2] * test_images.shape[3]
    # reshape to (n_samples, n_inputs)
    test_images = test_images.reshape((n_test, n_inputs))
    n_classes = 10

    # use torch tensors instead of np arrays, no grad needed as model is not trained on test images
    test_images = torch.tensor(test_images, requires_grad=False).to(device)
    test_targets = torch.tensor(test_targets, requires_grad=False).to(device)

    # initialize MLP model
    MLP_model = MLP(n_inputs=n_inputs, n_hidden=dnn_hidden_units, n_classes=n_classes, neg_slope=FLAGS.neg_slope)
    print(MLP_model)
    # loss function os loaded
    loss_module = nn.CrossEntropyLoss()

    learning_rate = FLAGS.learning_rate

    if OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD(MLP_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(MLP_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    batch_size = FLAGS.batch_size
    # extract max accuracy while training on test set
    max_acc = 0
    max_iter = 0

    # optimizer = torch.optimAdam(MLP_model.parameters(), lr=lr)
    for iteration in range(FLAGS.max_steps):

        train_images, train_targets = cifar10['train'].next_batch(batch_size)
        # input to MLP.forward is (batch_size, n_inputs)
        train_images = train_images.reshape((batch_size, n_inputs))

        # switch from numpy version to tensor and to device
        train_images = torch.tensor(train_images).type(torch.FloatTensor).to(device)
        train_targets = torch.tensor(train_targets).type(torch.LongTensor).to(device)

        if iteration % LR_FREQ == 0:
            learning_rate = learning_rate * 0.8
            optimizer = torch.optim.SGD(MLP_model.parameters(), lr=learning_rate,
                                        weight_decay=weight_decay)

        # gradients zero initialized
        optimizer.zero_grad()

        # predictions by forward pass
        train_predictions = MLP_model.forward(train_images)

        # loss acc to loss module, predictions and targets
        loss = loss_module(train_predictions, train_targets.argmax(dim=1))

        # Apply backward pass: MLP backward takes gradients of losses = dout
        # dout = backward of loss module
        loss.backward()
        # backward pass from loss (dout)
        optimizer.step()

        train_accuracies.append(accuracy(train_predictions, train_targets))
        train_losses.append(loss)
        steps.append(iteration)

        ## Save training statistics
        # save loss, acc, iteration for train evaluation afterwards
        if iteration % 100 == 0:
            print("iteration:" + str(iteration) + "train_acc:" + str(np.mean(train_accuracies)))

        # Consider FLAGS.EVAL_FREQ_DEFAULT for the evaluation of the current MLP
        # on the test data and training data
        if iteration % FLAGS.eval_freq == 0:
            ## Test Statistics
            test_predictions = MLP_model.forward(test_images)
            test_loss = loss_module.forward(test_predictions, test_targets.argmax(dim=1))
            test_acc = accuracy(test_predictions, test_targets)
            test_accuracies.append(test_acc)
            print("iteration:" + str(iteration) + "test_acc:" + str(test_accuracies[-1]))
            test_losses.append(test_loss)
            if (max_acc < test_acc):
                max_acc = test_acc
                max_iter = iteration

    print('Training is done')
    print('Save results in folder: .')
    # save loss and accuracies to plot from for report
    # folder for numpy results

    print('Training is done')
    print('Plot Results')

    plot_results(train_accuracies, test_accuracies, train_losses, test_losses)
    print("max accuracy: " + str(max_acc) + " at iteration: " + str(max_iter))


    ########################
    # END OF YOUR CODE    #
    #######################

def plot_results(train_accuracies, test_accuracies, train_losses, test_losses):
    plt.subplot(2, 1, 1)
    plt.title("Results")
    plt.plot(np.arange(len(train_accuracies)), train_accuracies, label="train acc")
    plt.plot(np.arange(len(test_accuracies) * FLAGS.eval_freq, step=FLAGS.eval_freq), test_accuracies, label="test acc")
    plt.ylabel('Accuracy (%)')
    plt.legend()
    # loss
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(train_losses)), train_losses, label=" train loss")
    plt.plot(np.arange(len(test_losses) * FLAGS.eval_freq, step=FLAGS.eval_freq), test_losses, label=" test loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('torch_' + NAME + '_results.png')
    plt.show()


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    parser.add_argument('--neg_slope', type=float, default=NEG_SLOPE_DEFAULT,
                        help='Negative slope parameter for LeakyReLU')
    FLAGS, unparsed = parser.parse_known_args()

    main()
