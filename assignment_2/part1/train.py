################################################################################
# MIT License
# 
# Copyright (c) 2019
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM


# You may want to look into tensorboard for logging
# from torch.utils.tensorboard import SummaryWriter

################################################################################

def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the model that we are going to use
    if config.model_type == "RNN":
        model = VanillaRNN(
            seq_length = config.input_length,
            input_dim = config.input_dim,
            num_hidden = config.num_hidden,
            batch_size = config.batch_size,
            num_classes = config.num_classes,
            device = device)

    elif config.model_type == "LSTM":
        model = LSTM(
            seq_length = config.input_length,
            input_dim = config.input_dim,
            num_hidden = config.num_hidden,
            num_classes = config.num_classes,
            device = device,
            batch_size=config.batch_size
                    )
    # send model to device
    model.to(device)
    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    # track training statistics
    train_accuracies = []
    train_losses = []

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        # batch inputs  to device for cuda
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        # convert input batches to tensors on device
        ínput_sequences = torch.tensor(batch_inputs, dtype=torch.float, device=device)
        targets = torch.tensor(batch_targets, dtype=torch.long, device=device)

        #print(ínput_sequences)
        #print(targets)

        # Backward pass
        # reset gradients
        optimizer.zero_grad()

        # Forward pass
        # Debugging
        # predict classes for input batches
        # a = ínput_sequences[:, 0].unsqueeze(1)
        # print(ínput_sequences.size())
        # print(a.size())
        # break

        # predict input sequences
        predictions = model.forward(ínput_sequences)
        # accuracy
        accuracy = torch.div(torch.sum(targets == predictions.argmax(dim=1)).to(torch.float), config.batch_size)
        # print(accuracy)
        # backpropagate loss
        # compute loss per batch
        loss = criterion(predictions, targets)
        loss.backward()


        ############################################################################
        # QUESTION: what happens here and why?
        # --> # ANSWER: Gradients are reinforced at each layer. Thus, very large gradients can appear. This leads to
        #  learning problems. Cutting the gradients to a limit overcomes that issue.
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################
        # update weights according to optimizer
        optimizer.step()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)
        # save stats for each step
        train_accuracies.append(accuracy)
        train_losses.append(loss)

        if step % 10 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))

            # If the last 50 accuracies are already 1 (avg=1), stop the training, as convergence is reached and unnecessary
            # computations dont have to be done
            avg_accuracies = np.sum(train_accuracies[-50:]) / 50
            print(avg_accuracies)
            if avg_accuracies == 1:
                print("\nTraining finished for length: {} after {} steps".format(config.input_length, step))
                print("Avg Accuracy : {:.3f}".format(avg_accuracies))
                break

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')

    return max(train_accuracies), step


 ################################################################################
 ################################################################################

def run_experiments(min_len1, max_len1, step_size1):
    # list with sequence lengths with which experiments shall be done
    seq_lengths = list(range(min_len1, max_len1, step_size1))
    experiment_accuracies = []
    experiment_steps = []

    seeds = [1, 5, 10]
    for seq_length in seq_lengths:
        tmp_accs = []
        tmp_steps = []
        for seed in range(len(seeds)):
            torch.manual_seed(seeds[seed])

            # Train the model
            config.input_length = seq_length
            max_acc, num_steps = train(config)
            tmp_accs.append(max_acc)
            tmp_steps.append(max_acc)
            #print("maxacc {}".format(max_acc))
        #print("mean {}".format(torch.mean(torch.tensor(tmp_accs))))

        experiment_accuracies.append(torch.mean(torch.tensor(tmp_accs)))
        # experiment_steps.append(torch.mean(torch.tensor(num_steps)))
        np.save("tmp_accuracies_" + config.plot_name + "_" + config.model_type, experiment_accuracies)
        print("Training finished for palindromes of length {}".format(seq_length))

    return experiment_accuracies, experiment_steps, seq_lengths

def plot_results(seq_lengths, accuracies, name):

    fig = plt.figure()
    plt.plot(seq_lengths, accuracies, marker='o', label='default_RNN')
    plt.ylabel('Accuracy')
    plt.xlabel('Sequence lengths')
    plt.legend()
    plt.title(config.model_type + " Palindrome Accuracy per length")
    fig.savefig(name + '.png')


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=5, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=500, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--plot_name', type=str, default="_experiments_default", help="Name for plot")
    parser.add_argument('--max_len', type=int, default=45, help="max seq_len")
    parser.add_argument('--min_len', type=int, default=5, help="min seq_len")


    config = parser.parse_args()

    # Train the model
    # train(config)

    accuracies, num_steps, seq_lengths = run_experiments(min_len1 = config.min_len, max_len1 = config.max_len, step_size1=5)
    # lisa doesnt like
    #plot_results(seq_lengths, accuracies, config.model_type + config.plot_name)

    #Save the final model
    np.save("train_accuracies_" + config.plot_name + "_" + config.model_type, accuracies)