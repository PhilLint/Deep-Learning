import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity

        self.layers = nn.Sequential(nn.Linear(args.latent_dim, 128), nn.LeakyReLU(args.neg_slope),
                                    nn.Linear(128, 256),
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(args.neg_slope),
                                    nn.Linear(256, 512),
                                    nn.BatchNorm1d(512),
                                    nn.LeakyReLU(args.neg_slope),
                                    nn.Linear(512, 1024),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(args.neg_slope),
                                    nn.Linear(1024, 784),
                                    nn.Tanh()
                                    )

    def forward(self, z):
        # Generate images from z
        out = self.layers(z)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity

        self.layers = nn.Sequential(nn.Linear(784, 512),
                                    nn.LeakyReLU(args.neg_slope),
                                    nn.Linear(512, 256),
                                    nn.LeakyReLU(args.neg_slope),
                                    nn.Linear(256, 1),
                                    nn.Sigmoid()
                                    )

    def forward(self, img):
        # return discriminator score for img
        out = self.layers(img)
        return out


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    device = args.device
    discriminator = discriminator.to(device)
    generator = generator.to(device)
    n_batches = len(dataloader)

    for epoch in range(args.n_epochs):
        # initialize epoch losses for both G and D


        for i, (imgs, _) in enumerate(dataloader):

            # flatten input images to vector of dim batch_size x 784
            imgs = imgs.reshape(imgs.shape[0], -1).to(device) # add to device
            #print(imgs.shape)
            # Train Generator
            # ---------------

            # sample random noise from standard normal Gaussian
            z = torch.randn((imgs.shape[0], args.latent_dim)).to(device)
            gen_imgs = generator(z)

            D_G_Z = discriminator(gen_imgs)
            #print(D_G_Z.shape)

            gen_loss = - torch.mean(torch.log(D_G_Z))
            # print(torch.log(D_G_Z))
            # set gradients to zero
            optimizer_G.zero_grad()
            # backpropagate loss
            gen_loss.backward(retain_graph=True)
            # parameter update
            optimizer_G.step()

            # Train Discriminator
            # -------------------
            # set gradients to zero
            optimizer_D.zero_grad()

            # real images evaluated by Disciminator
            D_X = discriminator(imgs)
            #print(D_X.shape)

            # discriminator loss
            disc_loss =  (torch.mean(- torch.log(D_X) - torch.log(1 - D_G_Z)))

            # backpropagate loss
            disc_loss.backward(retain_graph=True)
            # parameter update
            optimizer_D.step()

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                # save_image(gen_imgs[:25],
                #            'images/{}.png'.format(batches_done),
                #            nrow=5, normalize=True)
                print("Epoch {}/{} Discriminator loss: {}] | Generator loss: {}]".format(epoch, args.n_epochs, disc_loss.item(), gen_loss.item()))
                if batches_done % args.save_interval == 0:
                    save_image(gen_imgs[:25].view(-1, 1, 28, 28), args.result_folder + 'images_{}.png'.format(batches_done), nrow=5, normalize=True)



def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, ),
                                                (0.5, ))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator()
    discriminator = Discriminator()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    # torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default= 1,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--neg_slope', type=float, default=0.2)
    parser.add_argument('--result_folder', type=str, default="./results/GAN/")
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")
    args = parser.parse_args()

    main()
