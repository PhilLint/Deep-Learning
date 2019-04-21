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
from torch.autograd import Variable

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
weight_decay = 0.001

lr = LEARNING_RATE_DEFAULT
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
    pred_index = np.argmax(predictions, axis=1)
    label_index = np.argmax(targets, axis=1)
    # sum the number of matches and divide by the total number of images per batch
    accuracy = np.sum(pred_index == label_index) / label_index.shape[0]
    ########################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def train():
    """
    Performs training and evaluation of MLP model.

    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

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
    train_accuracies = []
    train_losses = []
    test_accuracies = []
    test_losses = []
    steps = []
    # load data from directory specified in the input
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)

    # load test images
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
    # use torch tensors instead of np arrays
    test_images = torch.tensor(test_images)#, requires_grad = False)#.type(torch.FloatTensor)  # .to(device)
    test_targets = torch.tensor(test_targets)#,requires_grad = False)#.type(torch.LongTensor) # .to(device)

    # initialize MLP model
    MLP_model = MLP(n_inputs=n_inputs, n_hidden=dnn_hidden_units, n_classes=n_classes)
    # loss function os loaded
    loss_module = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(MLP_model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optimAdam(MLP_model.parameters(), lr=lr)
    for iteration in range(FLAGS.max_steps):
        print("iteration:" + str(iteration))
        train_images, train_targets = cifar10['train'].next_batch(FLAGS.batch_size)
        # input to MLP.forward is (batch_size, n_inputs)
        train_images = train_images.reshape((FLAGS.batch_size, n_inputs))

        # switch from numpy version to tensor and to device
        train_images = torch.tensor(train_images, requires_grad = True).type(torch.FloatTensor)#.to(device)
        train_targets = torch.tensor(train_targets, requires_grad = True).type(torch.FloatTensor)#.to(device)

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

        ## Save training statistics
        # save loss, acc, iteration for train evaluation afterwards
        print("iteration:" + str(iteration) + "train_acc:" + str(np.mean(train_accuracies)))
        train_accuracies.append(accuracy(train_predictions, train_targets))
        train_losses.append(loss)
        steps.append(iteration)

        # Consider FLAGS.EVAL_FREQ_DEFAULT for the evaluation of the current MLP
        # on the test data and training data
        if iteration % FLAGS.eval_freq == 0:
            ## Test Statistics
            test_predictions = MLP_model.forward(test_images)
            test_loss = loss_module.forward(test_predictions, test_targets)
            print("iteration:" + str(iteration) + "test_acc:" + str(test_accuracies))
            test_accuracies.append(accuracy(test_predictions, test_targets))
            test_losses.append(test_loss)

    print('Training is done')
    print('Save results in folder: ./code/numpy')
    # save loss and accuracies to plot from for report
    # folder for numpy results
    folder = "./torh/"
    test_acc_path = folder + "test accuracies"
    np.save(test_acc_path, test_accuracies)
    train_acc_path = folder + "train accuracies"
    np.save(train_acc_path, train_accuracies)
    test_loss_path = folder + "test losses"
    np.save(test_loss_path, test_losses)
    train_loss_path = folder + "train losses"
    np.save(train_loss_path, train_losses)
    np.save(folder + "steps", steps)

    #  plot_results()

    ########################
    # END OF YOUR CODE    #
    #######################


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
    FLAGS, unparsed = parser.parse_known_args()

    main()
