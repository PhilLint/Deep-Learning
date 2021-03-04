"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from convnet_pytorch import ConvNet
import cifar10_utils

import torch
from torch import nn
from torch import optim

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

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
  #######################
  corr_pred = torch.sum(torch.eq(torch.argmax(predictions, dim=1), targets)).item()
  acc = corr_pred / predictions.size()[0]  ########################
  # END OF YOUR CODE    #
  #######################

  return acc

def train():
  """
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ########################
  # PUT YOUR CODE HERE  #
  #######################

  if FLAGS.data_dir:
    DATA_DIR_DEFAULT = FLAGS.data_dir

  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

  batch_size = FLAGS.batch_size
  learning_rate = FLAGS.learning_rate

  cifar_data = cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)

  train_data = cifar_data['train']
  test_data = cifar_data['test']

  input_channels = train_data.images.shape[1]
  n_classes = train_data.labels.shape[1]

  criterion = nn.CrossEntropyLoss()
  model = ConvNet(input_channels, n_classes).to(device)

  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  # Train and Test losses
  losses = [[], []]
  # Train and Test accuracies
  accuracies = [[], []]

  # True iteration for plotting
  iterations = []

  for iteration in np.arange(FLAGS.max_steps):
    x, y = train_data.next_batch(batch_size)
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(np.argmax(y, axis=1)).type(torch.LongTensor).to(device)

    train_output = model.forward(x)
    loss = criterion(train_output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iteration % FLAGS.eval_freq == 0 or iteration == FLAGS.max_steps - 1:
      iterations.append(iteration)

      x_test, y_test = test_data.next_batch(10 * batch_size)
      x_test = torch.from_numpy(x_test).to(device)
      y_test = torch.from_numpy(np.argmax(y_test, axis=1)).type(torch.LongTensor).to(device)

      # Second forward pass for test set
      with torch.no_grad():
        test_output = model.forward(x_test)

      # Calculate losses
      train_loss = criterion.forward(train_output, y)
      losses[0].append(train_loss)

      test_loss = criterion.forward(test_output, y_test)
      losses[1].append(test_loss)

      # Calculate accuracies
      train_acc = accuracy(train_output, y)
      test_acc = accuracy(test_output, y_test)
      accuracies[0].append(train_acc)
      accuracies[1].append(test_acc)

      print("Iteration {}, Train loss: {}, Train accuracy: {}, Test accuracy: {}".format(iteration, train_loss,
                                                                                         train_acc, test_acc))

  fig = plt.figure(figsize=(25, 10), dpi=200)
  fig.suptitle('PyTorch ConvNet: Losses and Accuracies', fontsize=40)
  ax1 = fig.add_subplot(1, 2, 1)
  ax2 = fig.add_subplot(1, 2, 2)

  ax1.plot(iterations, losses[0], linewidth=4, color="g", label="Train loss")
  ax1.plot(iterations, losses[1], linewidth=4, color="c", label="Test loss")
  ax2.plot(iterations, accuracies[0], linewidth=4, color="g", label="Train accuracy")
  ax2.plot(iterations, accuracies[1], linewidth=4, color="c", label="Test accuracy")

  ax1.set_xlabel('$Iteration$', fontsize=28)
  ax1.set_ylabel('$Loss$', fontsize=28)
  ax2.set_xlabel('$Iteration$', fontsize=28)
  ax2.set_ylabel('$Accuracy$', fontsize=28)

  ax1.legend(fontsize=22)
  ax2.legend(fontsize=22)

  plt.savefig("../figures/pytorch_convnet.png")
  plt.show()

  with open('results_convnet.pkl', 'wb') as f:
    pickle.dump([losses, accuracies], f)  ########################
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
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()