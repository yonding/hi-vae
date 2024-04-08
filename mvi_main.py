from mv_rand_generate import mv_rand_generate
from split_datasets import split_and_convert_to_tensor
from args import parse_args
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse


class MVIDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class Autoencoder(nn.Module):
    def __init__(self, D_in, H, H2, latent_dim):
        # Encoder
        super(Autoencoder, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.lin_bn1 = nn.BatchNorm1d(num_features=H)  # Batch normalization
        self.linear2 = nn.Linear(H, H2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=H2)  # Batch normalization
        self.linear3 = nn.Linear(H2, H2)
        self.lin_bn3 = nn.BatchNorm1d(num_features=H2)  # Batch normalization
        # Latent vectors mu and sigma
        self.fc1 = nn.Linear(H2, latent_dim)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)
        # Sampling vector
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, H2)
        self.fc_bn4 = nn.BatchNorm1d(H2)  # Batch normalization
        # Decoder
        self.linear4 = nn.Linear(H2, H2)
        self.lin_bn4 = nn.BatchNorm1d(num_features=H2)
        self.linear5 = nn.Linear(H2, H)
        self.lin_bn5 = nn.BatchNorm1d(num_features=H)
        self.linear6 = nn.Linear(H, D_in)
        self.lin_bn6 = nn.BatchNorm1d(num_features=D_in)
        self.relu = nn.ReLU()  # Activation Function

    def encode(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))
        fc1 = F.relu(self.bn1(self.fc1(lin3)))
        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)
        return r1, r2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))
        lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))
        return self.lin_bn6(self.linear6(lin5))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    # x_recon ist der im forward im Model erstellte recon_batch, x ist der originale x Batch, mu ist mu und logvar ist logvar
    def forward(self, x_recon, z, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, z)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss_MSE + loss_KLD


def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find("Linear") != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def main(args):
    train_losses = []

    x_df, z_df, y_df = mv_rand_generate(
        max_remove_count=args.max_remove_count,
        new_num_per_origin=args.new_num_per_origin,
    )

    x_train, x_val, x_test, y_train, y_val, y_test, z_train, z_val, z_test = (
        split_and_convert_to_tensor(
            x_df, z_df, y_df, args.val_size, args.test_size, args.random_state
        )
    )

    train_dataset = MVIDataset(x_train, z_train)
    val_dataset = MVIDataset(x_val, z_val)
    test_dataset = MVIDataset(x_test, z_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = Autoencoder(x_train.shape[1], args.H, args.H2, args.latent_dim).to(
        args.device
    )
    model.apply(weights_init_uniform_rule)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_mse = customLoss()  # MSE + KLD

    for epoch in range(1, args.epochs + 1):
        train(epoch, model, train_loader, optimizer, loss_mse, train_losses, args)


def train(epoch, model, train_loader, optimizer, loss_fn, train_losses, args):
    model.train()
    train_loss = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(args.device)
        targets = targets.to(args.device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_fn(recon_batch, targets, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    if epoch % 200 == 0:
        print(
            "====> Epoch: {} Average loss: {:.4f}".format(
                epoch, train_loss / len(train_loader.dataset)
            )
        )
        train_losses.append(train_loss / len(train_loader.dataset))


if __name__ == "__main__":
    args = parse_args()
    main(args)
