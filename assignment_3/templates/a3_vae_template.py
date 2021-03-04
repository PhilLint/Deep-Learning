import argparse
import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from torchvision.utils import save_image
from scipy.stats import norm
from datasets.bmnist import bmnist


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        # BMnist input is 28x28 -> input_dim = 784
        in_out_dim = 28 * 28

        # initialize building blocks of VAE
        # activation
        self.ReLu = nn.ReLU()
        # linear layers
        self.in_Linear = nn.Linear(in_out_dim, hidden_dim)
        self.mu_Linear = nn.Linear(hidden_dim, z_dim)
        self.std_Linear = nn.Linear(hidden_dim, z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        # First pass of input through first linear layer and activation
        hidden = self.in_Linear(input)
        relu_in_Linear = self.ReLu(hidden)

        # Pass in_Linear through both mu and std linear layers
        mean = self.mu_Linear(relu_in_Linear)
        std = self.std_Linear(relu_in_Linear)

        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        in_out_dim = 28 * 28

        self.decode = nn.Sequential(nn.Linear(z_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, in_out_dim),
                                    nn.Sigmoid())


    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        # mean as latent linear input -> ReLu -> linear Mu layer -> Sigmoid activation
        mean = self.decode(input)

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        # flatten input to vector
        input = input.view(-1, 784)

        # ENCODE
        # pass input through encoder to obatin mu and std
        mean, std = self.encoder.forward(input)
        #print(mean)
        #print(std)

        # Reparametrization Trick
        # sample epsilon as standard normal of dimension of the returned mean, as z_dim incorporates batch_size
        eps = torch.randn(mean.shape)

        # as derived in my report
        z = mean + torch.sqrt(torch.exp(std)) * eps
        #print(z)
        #print("now decode")
        # DECODE
        # pass latent z through decoder to reconstruct original inputs
        out = self.decoder.forward(z)

        #print("out:{}".format(out))
        # LOSS
        # reconstruction loss (equation 11 in report)
        L_recon = torch.sum(- input * torch.log(out + 1e-6) - (1-input) * torch.log(1- out + 1e-6), dim =1)

        # regularization loss (equation 13 in report)
        #L_reg = -(1/2) * torch.sum(1 + std - torch.pow(mean, 2) - torch.log(std + 1e^-7))
        L_reg = -(1/2) * torch.sum(1 + std - torch.pow(mean, 2) - torch.pow(2, std), dim=1)

        #print("L_recon:{} L_reg:{}".format(L_recon, L_reg))
        # average negative elbo
        average_negative_elbo = torch.mean(L_recon + L_reg)

        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        # z sample according to standard normal prior
        #print(n_samples)
        #print(self.z_dim)

        z = torch.randn((n_samples, self.z_dim))
        #print(z)

        # generate images from the decoder based on the latent input z
        im_means = self.decoder(z)
        # im_means = torch.round(im_means)

        # bernoulli to average
        #print(im_means)
        sampled_ims = torch.bernoulli(im_means)
        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    total_avg_elbo = []
    # training mode or not
    if model.training:

        # iterate through data
        for iteration, input in enumerate(data):
            # initialize optimizer gradients to zero
            optimizer.zero_grad()
            #print("iteration:{}".format(iteration))
            #print(input)
            # forward pass
            average_negative_elbo = model(input)
            #print("average_negative_elbo : {}".format(average_negative_elbo ))
            # backward
            average_negative_elbo.backward()
            # update weights
            optimizer.step()
            #print("average_negative_elbo:{}".format(average_negative_elbo))
            # append current loss to loss_list (tensor, so .item required)
            total_avg_elbo.append(average_negative_elbo.item())

    else:
        # don't update weights if training is not activated
        with torch.no_grad():
            # iterate through data
            for iteration, input in enumerate(data):
                # forward pass
                average_negative_elbo = model.forward(input)
                # append current loss to loss_list
                total_avg_elbo.append(average_negative_elbo.item())

    average_epoch_elbo = np.mean(total_avg_elbo)

    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim)
    #print(model)
    optimizer = torch.optim.Adam(model.parameters())

    # store images in result folder
    result_folder = './results'

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        n_samples = 25 # allows for 5x5 grid of samples
        # plot before training
        if epoch == 0:
            plot_sample_means(model, n_samples, result_folder, epoch, name="before_training")
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------
        # samples plotted for each epoch only for latent space = 20
        if ARGS.zdim != 2:
            plot_sample_means(model, n_samples, result_folder, epoch, name="")

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------
    # only for latent dimension = 2 plot manifold
    if ARGS.zdim == 2:
        plot_manifold(model=model, result_folder=result_folder, epoch=epoch)

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')

def plot_sample_means(model, n_samples, result_folder, epoch, name=""):
    # get samples from model
    sampled_ims, im_means = model.sample(n_samples)
    # reshape samples, means
    sampled_ims = sampled_ims.view(n_samples, 1, 28, 28)
    im_means = im_means.view(n_samples, 1, 28, 28)

    # save sampled images
    save_image(sampled_ims, result_folder + '/samples/' + name + 'samples_iteration_{}.png'.format(epoch), nrow=5,
               normalize=True, padding=5)
    # save sampled means
    save_image(im_means, result_folder + '/means/' + name + 'means_iteration_{}.png'.format(epoch), nrow=5, normalize=True,
               padding=5)

def plot_manifold(model, result_folder, epoch, n_samples = 20):
        # turn model to evaluation mode
        model.eval()
        values = np.linspace(0.001, 0.999, n_samples)

        x_values = norm.ppf(values)
        y_values = norm.ppf(values)

        # fill tensor with latent values
        z = [torch.Tensor((x, y)) for x in x_values for y in y_values]
        z = torch.stack(z)
        # no gradients
        with torch.no_grad():
            images = model.decoder(z)

        save_image(images.view(images.shape[0], 1, 28, 28),
                   result_folder + '/manifold_plot_iteration{}.png'.format(epoch), normalize=True,
                    nrow=n_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()




