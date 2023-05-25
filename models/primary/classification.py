# Imports
from __future__ import print_function, division

import numpy as np
import time
import matplotlib.pyplot as plt
import time
import copy
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torchvision import transforms
from torchinfo import summary
import torch.backends.cudnn as cudnn
import torch.utils.data as data


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


def trainTL(model, criterion, optimizer, scheduler, num_epochs=25, batch_size=64):
    since = time.time()
    train_loader = torch.utils.data.DataLoader(
        oversampled_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_size = sum([3792, 3834, 3851, 3681, 3781, 3745, 3774])
    val_size = sum([419, 360, 744, 840, 830, 744, 418])

    iters, train_loss, val_loss, train_acc, val_acc = [], [], [], [], []

    n = 0  # the number of iterations
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
                dataloader = train_loader
                dataset_sizes = train_size
            else:
                model.eval()  # Set model to evaluate mode
                dataloader = val_loader
                dataset_sizes = val_size

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes
            epoch_acc = running_corrects.double() / dataset_sizes

            print(f"{phase} Loss: {epoch_loss} Acc: {epoch_acc}")

            if phase == "train":
                train_acc.append(epoch_acc.cpu())
                train_loss.append(epoch_loss)
                iters.append(n)

            else:
                val_acc.append(epoch_acc.cpu())
                val_loss.append(epoch_loss)

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        n += 1
        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # plotting
    plt.title("Loss Curve")
    plt.plot(iters, train_loss, label="Train")
    plt.plot(iters, val_loss, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(loc="best")
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

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


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


# Visualize EfficientNet
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


def visualize_model(model, test_dataset, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=10, shuffle=True, num_workers=4, pin_memory=True
    )
    y_pred = []
    y_true = []

    class_names = [
        "buildings",
        "desert",
        "forest",
        "glacier",
        "mountain",
        "sea",
        "street",
    ]

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

    # Move the model to the GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    modelEN.to(device)

    # Move the input tensor to the same device as the model's weights
    image_tensor = image_tensor.to(device)

    # Pass the image through the model and get the predicted class
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
        predicted = torch.argmax(output.data, 1)

    # Print the predicted class
    print(predicted.item())
    image.show()


if __name__ == "__main__":
    # Set the manual seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Setup device agnostic code
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = True
    cudnn.benchmark = True

    plt.ion()  # interactive mode

    train_dataset, val_dataset, test_dataset = train_test_split()
    oversampled_train_dataset = data_augmentation(train_dataset)

    # Setup the model with pretrained weights and send it to the target device (torchvision v0.13+)
    weights = (
        torchvision.models.EfficientNet_V2_L_Weights.DEFAULT
    )  # .DEFAULT = best available weights

    modelEN = torchvision.models.efficientnet_v2_l(weights=weights)

    # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
    for param in modelEN.features.parameters():
        param.requires_grad = False

    modelEN.classifier = nn.Sequential(
        nn.Linear(1280, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 7),
    )

    # Do a summary *after* freezing the features and changing the output classifier layer (uncomment for actual output)
    # Used for fine tuning/debugging. Comment out if not needed
    summary(
        modelEN,
        input_size=(
            64,
            3,
            150,
            150,
        ),  # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
        verbose=0,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"],
    )

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(modelEN.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    modelEN.cuda()

    model_ft = trainTL(modelEN, criterion, optimizer, exp_lr_scheduler, num_epochs=10)
    torch.save(model_ft, "efficientnet")

    # test accuracy of EfficientNet
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
    )

    modelEN.eval()
    test_accuracy = get_test_accuracy(modelEN, test_loader)
    print(f"Test accuracy: {test_accuracy * 100}%")

    # test model on test data
    visualize_model(modelEN, test_dataset)

    test_single_image(modelEN, "./images/abbey-road-london.webp")
