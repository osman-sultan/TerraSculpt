# Imports
from __future__ import print_function, division

import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torchinfo import summary
import torch.utils.data as data


class CNNbaseline(nn.Module):
    def __init__(self, name="CNN"):
        super(CNNbaseline, self).__init__()
        self.name = name
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 37 * 37, 128)
        self.fc2 = nn.Linear(128, 7)

    def forward(self, img):
        x = self.pool(F.relu(self.conv1(img)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 37 * 37)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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


def get_model_path(name, batch_size, learning_rate, epoch):
    """Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(
        name, batch_size, learning_rate, epoch
    )
    return path


def get_accuracy(model, data_loader):
    data = data_loader
    use_cuda = True
    correct = 0
    total = 0
    for imgs, labels in data:
        #############################################
        # To Enable GPU Usage
        if use_cuda and torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        #############################################

        torch.unsqueeze(labels, 1)
        output = model(imgs)

        # select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]

    return correct / total


def trainCNN(
    model, train_dataset, val_dataset, batch_size=32, learn_rate=0.01, num_epochs=1
):
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    criterion = nn.CrossEntropyLoss()

    # optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    iters, losses, train_acc, val_acc = [], [], [], []

    use_cuda = True
    # training
    start_time = time.time()
    n = 0  # the number of iterations
    for epoch in range(num_epochs):
        time1 = time.time()
        for imgs, labels in iter(train_loader):
            #############################################
            # To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()
            #############################################

            out = model(imgs)  # forward pass
            loss = criterion(out, labels)  # compute the total loss
            loss.backward()  # backward pass (compute parameter updates)
            # print(".backward() works")
            optimizer.step()  # make the updates for each parameter
            # print(".step() works")
            optimizer.zero_grad()  # a clean up step for PyTorch
            # print(".zero_grad() works")

        # save the current training information
        iters.append(n)
        losses.append(float(loss) / batch_size)  # compute *average* loss
        train_acc.append(get_accuracy(model, train_loader))
        epoch_time = time.time() - time1
        val_acc.append(get_accuracy(model, val_loader))  # compute validation accuracy
        print(
            f"Epoch {epoch} train accuracy: {train_acc[epoch]} | val accuracy: {val_acc[epoch]} | Time elapsed: {epoch_time}"
        )

        model_path = get_model_path(model.name, batch_size, learn_rate, epoch)
        torch.save(model.state_dict(), model_path)

        n += 1

    # plotting
    plt.title("Training Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Training Curve")
    plt.plot(iters, train_acc, label="Train")
    plt.plot(iters, val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc="best")
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))


# Test Accuracy
def get_test_accuracy(model, test_loader):
    correct = 0
    total = 0
    for imgs, labels in test_loader:
        #############################################
        # To Enable GPU Usage
        if use_cuda and torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        #############################################

        output = model(imgs)

        # select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]

    return correct / total


# Visualize EfficientNet/CNN
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(model, test_dataset, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=10, shuffle=True, num_workers=4, pin_memory=True
    )
    y_pred = []
    y_true = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred += preds.tolist()
            y_true += labels.tolist()

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis("off")
                ax.set_title(f"predicted: {class_names[preds[j]]}")
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

        # confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.set()
        sns.heatmap(cm, annot=True, fmt="g", cmap="Blues", cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()


def test_single_image(model, path):
    # Load the image and convert it to a tensor
    image = Image.open(path)
    transform = transforms.Compose(
        [
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_tensor = transform(image)

    # Move the input tensor to the same device as the model's weights
    image_tensor = image_tensor.to(device)

    # Pass the image through the model and get the predicted class
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
        predicted = torch.argmax(output.data, 1)

    # Print the predicted class
    print(predicted.item())
    image.show()


def main():
    train_dataset, val_dataset, test_dataset = train_test_split()

    oversampled_train_dataset = data_augmentation(train_dataset)

    model = CNNbaseline(name="OversampledCNN")

    # summarizes architecture (num params, output size, etc) -> comment out if not needed, used for debugging/fine tuning mostly
    print(
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
    )

    # train baseline CNN
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        model.cuda()
        print("CUDA is available! Training on GPU ...")
    else:
        print("CUDA is not available. Training on CPU ...")

    trainCNN(
        model,
        oversampled_train_dataset,
        val_dataset,
        batch_size=64,
        learn_rate=0.0001,
        num_epochs=10,
    )

    # get baseline CNN accuracy
    model.eval()

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
    )

    test_accuracy = get_test_accuracy(model, test_loader)
    print(f"Test accuracy: {test_accuracy * 100}%")

    class_names = [
        "buildings",
        "desert",
        "forest",
        "glacier",
        "mountain",
        "sea",
        "street",
    ]

    train_loader = torch.utils.data.DataLoader(
        oversampled_train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Get a batch of training data
    inputs, classes = next(iter(train_loader))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])

    # test model with test data and show corresponding confusion matrix/heat map
    visualize_model(model, test_dataset, class_names, 3)

    # optionally test an image not within the dataset
    test_single_image(model, "./images/abbey-road-london.webp")


if __name__ == "__main__":
    # Set the manual seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Setup device agnostic code
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = True
    cudnn.benchmark = True

    plt.ion()  # interactive mode

    main()
