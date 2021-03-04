import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import torch
import torch.nn as nn

from torchvision.utils import save_image
from scipy import stats
import numpy as np
import os

import datetime

from datasets.bmnist import bmnist


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(28 * 28, hidden_dim)
        self.relu = nn.ReLU()
        #         self.relu = nn.LeakyReLU()
        self.sigma2 = nn.Linear(hidden_dim, z_dim)
        self.mu = nn.Linear(hidden_dim, z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.
        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        out = self.relu(self.linear(input))
        mean = self.sigma2(out)
        std = self.mu(out)

        # raise NotImplementedError()

        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 784),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        Perform forward pass of encoder.
        Returns mean with shape [batch_size, 784].
        """
        mean = self.model(input)
        # raise NotImplementedError()

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super(VAE, self).__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)
        self.binary_loss = nn.BCELoss(reduction='sum')

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        # encoding
        input = input.view(-1, 784)
        mu, log_var = self.encoder(input)

        # re-parametrize
        eps = torch.randn(mu.shape)
        z = mu + log_var.exp().sqrt() * eps

        # decoding
        pred = self.decoder(z)

        # Calculating recon and reg loss
        recon_loss = self.binary_loss(pred, input)
        #         print(recon_loss)
        reg_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())
        #         print(reg_loss)

        average_negative_elbo = (recon_loss + reg_loss) / input.size()[0]
        #         print(average_negative_elbo)

        # raise NotImplementedError()..
        return average_negative_elbo

    def sample(self, n_samples):
        np.random.seed(21)
        torch.manual_seed(21)
        self.decoder.eval()
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """

        # z = torch.from_numpy(np.random.normal(0, 1, size=(n_samples,self.z_dim))).float()
        # z = torch.autograd.Variable(z, requires_grad=False)
        z = torch.randn((n_samples, self.z_dim))
        means = self.decoder(z)

        sampled_ims = torch.bernoulli(means)
        im_means = torch.round(means)
        # raise NotImplementedError()

        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.
    Returns the average elbo for the complete epoch.
    """
    total_elbo = []

    if model.training:
        for i, input in enumerate(data):
            avg_elbo = model.forward(input)
            avg_elbo.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_elbo.append(avg_elbo.item())
    else:
        with torch.no_grad():
            for i, input in enumerate(data):
                avg_elbo = model.forward(input)
                total_elbo.append(avg_elbo.item())

    average_epoch_elbo = np.mean(total_elbo)

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
    current_date = datetime.datetime.now()

    # create path for storing models
    checkpoint_path = "checkpoints/VAE/%02d_%02d_%02d__%02d_%02d_%02d/" % (
    current_date.day, current_date.month, current_date.year, current_date.hour, current_date.minute,
    current_date.second)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # create path for storing sampled images during training
    results_path = "results/VAE/%02d_%02d_%02d__%02d_%02d_%02d/" % (
    current_date.day, current_date.month, current_date.year, current_date.hour, current_date.minute,
    current_date.second)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim, hidden_dim=512)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-6)

    n_samples = 25

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs + 1):
        if epoch % 1 == 0:
            sampled_ims, im_means = model.sample(n_samples)
            save_image(sampled_ims.view(n_samples, 1, 28, 28), results_path + 'sample_epoch_{}.png'.format(epoch),
                       nrow=5, normalize=True, padding=5)
            save_image(im_means.view(n_samples, 1, 28, 28), results_path + 'mean_epoch_{}.png'.format(epoch), nrow=5,
                       normalize=True, padding=5)
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------
    if ARGS.zdim == 2:
        N = 10
        percentiles = np.linspace(0, 1, N)
        percentiles[0] += (percentiles[1] - percentiles[0]) / 10
        percentiles[-1] -= (percentiles[-1] - percentiles[-2]) / 10
        inv_cdf = stats.norm.ppf(percentiles)
        plt.figure(figsize=(2 * N, 2 * N), dpi=160)
        for i in range(N):
            for j in range(N):
                pred = model.decoder(torch.Tensor([inv_cdf[i], inv_cdf[j]]))
                plt.subplot(N, N, i * (N) + j + 1)
                plt.imshow(pred.view(1, 28, 28).squeeze().data.numpy(), cmap='gray')
                plt.axis('off')
        plt.savefig(results_path + 'manifold.png')
    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')
    torch.save(model.state_dict(), checkpoint_path + "model.pickle")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--save_path', type=str, default='results/VAE/',
                        help='path for saving files')
    parser.add_argument('--model_path', type=str, default='checkpoints/',
                        help='path for saving models')

    ARGS = parser.parse_known_args()[0]
    main()