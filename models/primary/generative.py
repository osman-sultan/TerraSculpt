# Imports
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchinfo import summary
import torch.utils.data as data


# Variational Autoencoder
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(  # like the Composition layer you built
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),
        )
        # self.fc1 = nn.Linear(64 * 32 * 32, 350)
        self.relu = nn.ReLU()
        self.fc2m = nn.Linear(64 * 32 * 32, zdim)  # mu layer
        self.fc2s = nn.Linear(64 * 32 * 32, zdim)  # sd layer

        # decoder
        self.fc3 = nn.Linear(zdim, 64 * 32 * 32)
        # self.fc4 = nn.Linear(350, 64 * 32 * 32)
        # self.sigmoid = nn.Sigmoid()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=2, output_padding=1),
            nn.Tanh(),
        )

    def encode(self, x):
        x = self.encoder(x)
        h1 = x.view(-1, 64 * 32 * 32)
        return self.fc2m(h1), self.fc2s(h1)

    # reparameterize
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        x = h3.view(-1, 64, 32, 32)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# loss function for VAE are unique and use Kullback-Leibler
# divergence measure to force distribution to match unit Gaussian
def loss_function(recon_x, x, mu, logvar, batch_size):
    mse = F.mse_loss(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld /= batch_size * 150 * 150 * 3
    return mse + kld


def train_test_split():
    main = "./landscape-generation-and-classification"  # update this path as per your local directory structure

    # Define data transformations
    transform = transforms.Compose(
        [
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 70-15-15 split
    dataset = torchvision.datasets.ImageFolder(main + "/dataset", transform=transform)

    train_size = int(0.7 * len(dataset))
    test_size = int(0.15 * len(dataset)) + 1
    val_size = int(0.15 * len(dataset))

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    return train_dataset, val_dataset, test_dataset


def data_augmentation(train_dataset):
    # Weighted Oversampling using Image Augmentation
    transform_train_subset = transforms.Compose(
        [
            transforms.Resize((150, 150)),
            transforms.RandomCrop((125, 125)),
            transforms.RandomRotation(degrees=45),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.05),
            transforms.RandomResizedCrop(size=150, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # calculate the number of examples to sample from each class
    class_indices = np.array([1831, 1701, 3614, 3732, 3780, 3664, 2001])
    max_class_size = 3780
    class_weights = [(3780 / class_indices[c]) for c in range(7)]
    num_samples = [int(class_weights[c] * class_indices[c]) for c in range(7)]

    sample_weights = np.zeros(len(train_dataset))
    sample_weights = [class_weights[label] for _, label in train_dataset]

    for idx, (tensor, label) in enumerate(train_dataset):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight

    # create a WeightedRandomSampler to oversample the training set
    sampler = data.WeightedRandomSampler(
        weights=sample_weights, num_samples=sum(num_samples), replacement=True
    )

    # create new training set with oversampled examples
    oversampled_train_dataset = data.Subset(train_dataset, indices=list(sampler))

    # Sampling the subset
    oversampled_train_dataset.transform = transform_train_subset

    return oversampled_train_dataset


def train(model, num_epochs=1, batch_size=64, learning_rate=0.0001):
    model.train()  # train mode so that we do reparameterization

    train_loader = torch.utils.data.DataLoader(
        oversampled_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    optimizer = optim.Adam(model.parameters(), learning_rate)

    outputs = []
    for epoch in range(num_epochs):
        for data in train_loader:  # load batch
            img, _ = data

            #############################################
            # To Enable GPU Usag
            img = img.cuda()
            #############################################
            recon, mu, logvar = model(img)

            loss = loss_function(recon, img, mu, logvar, batch_size)  # calculate loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("Epoch:{}, Loss:{:.4f}".format(epoch + 1, float(loss)))
        outputs.append(
            (epoch, img, recon),
        )

    return model, outputs


def visualize_model(outputs):
    for k in range(0, 10, 5):
        plt.figure(figsize=(9, 2))
        imgs = outputs[k][1].cpu().detach().numpy()
        recon = outputs[k][2].cpu().detach().numpy()
        for i, item in enumerate(imgs):
            if i >= 9:
                break
            plt.subplot(2, 9, i + 1)
            item1 = item.transpose(1, 2, 0)
            plt.imshow(item1)

        for i, item in enumerate(recon):
            if i >= 9:
                break
            plt.subplot(2, 9, 9 + i + 1)
            item2 = item.transpose(1, 2, 0)
            plt.imshow(item2)


if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = train_test_split()
    oversampled_train_dataset = data_augmentation(train_dataset)

    # dimensions of latent space
    zdim = 64

    modelVAE = VAE()

    # for debugging/fine-tuning. visualizes model architecuture. comment out if not needed
    summary(
        model=modelVAE,
        input_size=(
            64,
            3,
            150,
            150,
        ),  # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"],
    )

    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        modelVAE.cuda()
        print("CUDA is available! Training on GPU ...")
    else:
        print("CUDA is not available. Training on CPU ...")

    batch_size = 64
    modelVAE, outputs = train(
        modelVAE, oversampled_train_dataset, num_epochs=20, batch_size=batch_size
    )

    # test vae
    visualize_model(outputs)

    torch.save(modelVAE, "VAE")
