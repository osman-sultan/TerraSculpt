# Imports
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms
from torchinfo import summary
import torch.utils.data as data


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(  # like the Composition layer you built
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=2, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


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


def train(model, train_dataset, num_epochs=5, batch_size=64, learning_rate=1e-3):
    torch.manual_seed(42)
    criterion = nn.MSELoss()  # mean square error loss
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    outputs = []
    for epoch in range(num_epochs):
        for data in train_loader:
            img, _ = data

            #############################################
            # To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
                img = img.cuda()
            #############################################

            recon = model(img)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("Epoch:{}, Loss:{:.4f}".format(epoch + 1, float(loss)))
        outputs.append(
            (epoch, img, recon),
        )
    return outputs


def visualize_model(outputsAE):
    for k in range(0, 10, 5):
        plt.figure(figsize=(9, 2))
        imgs = outputsAE[k][1].cpu().detach().numpy()
        recon = outputsAE[k][2].cpu().detach().numpy()
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

    model = Autoencoder()
    summary(
        model=model,
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
        model.cuda()
        print("CUDA is available! Training on GPU ...")
    else:
        print("CUDA is not available. Training on CPU ...")

    max_epochs = 10
    outputsAE = train(model, train_dataset, num_epochs=max_epochs)
    visualize_model(outputsAE)
