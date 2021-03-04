import numpy as np
import matplotlib.pyplot as plt

# load saved results
lstm = "train_accuracies_full_experiments_LSTM.npy"
lstm_results = np.load(lstm)
lstm_5= "train_accuracies_5_experiments_LSTM.npy"
lstm_results_5 = np.load(lstm_5)

full_lstm= list(lstm_results_5) + list(lstm_results)

rnn= "train_accuracies_full_experiments_RNN.npy"
rnn_results = np.load(rnn)

rnn_5= "train_accuracies_5_experiments_RNN.npy"
rnn_results_5 = np.load(rnn_5)

full_rnn = list(rnn_results_5) + list(rnn_results)


# plot helper function
def plot_results(seq_lengths, accuracies, name, model):

    fig = plt.figure()
    plt.plot(seq_lengths, accuracies, marker='o', label=model)
    plt.ylabel('Accuracy')
    plt.xlabel('Sequence lengths')
    plt.legend()
    plt.title(name + " Palindrome Accuracy per length")
    fig.savefig(name + "experiments" + '.png')

# make plots
plot_results([5,10,15,20,25,30,35, 40], full_rnn, "RNN_test", "RNN")
plot_results([5,10,15,20,25,30,35, 40], full_lstm, "LSTM_full_", "LSTM")


