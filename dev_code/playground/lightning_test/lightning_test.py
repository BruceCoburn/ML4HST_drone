"""
Create a python script which trains a convolutional neural network on the MNIST dataset using PyTorch Lightning.
The script should be able to be run from the command line as follows:

python pytorch_cnn_mnist.py --epochs 10 --batch_size 64 --learning_rate 0.01

The script should output the training loss and accuracy to the command line every 100 iterations.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pytorch_lightning as pl

data_folder = "./data_lightning"


class CNN(pl.LightningModule):
    """
    A lot of this code was generated from referencing the PyTorch Lightning documentation:
    https://lightning.ai/docs/pytorch/stable/

    TODO: Add more comments
    """

    def __init__(self):
        super(CNN, self).__init__()
        # 1 input image channel, 10 output channels, 3x3 square convolution
        # kernel

        # Dummy input to calculate the output shape of each layer
        self.dummy_input = torch.ones(1, 1, 28, 28)

        self.architecture = nn.Sequential()

        ###############################
        # Convolution Layer 1
        ###############################
        self._nice_print("Convolution Layer 1")

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=0
        )
        self.architecture.add_module("conv1", self.conv1)
        self._print_layer_output_shape("conv1", self.architecture)

        self.relu1 = nn.ReLU()
        self.architecture.add_module("relu1", self.relu1)
        self._print_layer_output_shape("relu1", self.architecture)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.architecture.add_module("maxpool1", self.maxpool1)
        self._print_layer_output_shape("maxpool1", self.architecture)

        ###############################
        # Convolution Layer 2
        ###############################
        self._nice_print("Convolution Layer 2")

        self.conv2 = nn.Conv2d(
            in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=0
        )
        self.architecture.add_module("conv2", self.conv2)
        self._print_layer_output_shape("conv2", self.architecture)

        self.relu2 = nn.ReLU()
        self.architecture.add_module("relu2", self.relu2)
        self._print_layer_output_shape("relu2", self.architecture)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.architecture.add_module("maxpool2", self.maxpool2)
        self._print_layer_output_shape("maxpool2", self.architecture)

        ###############################
        # Flatten Layer
        ###############################
        self._nice_print("Flatten Layer")

        self.flatten = nn.Flatten()
        self.architecture.add_module("flatten", self.flatten)
        self._print_layer_output_shape("flatten", self.architecture)

        ###############################
        # Fully Connected Layer 1
        ###############################
        self._nice_print("Fully Connected Layer 1")

        self.FULLY_CONNECTED_INPUTS = self._get_layer_output_shape(self.architecture)[1]
        self.fc1 = nn.Linear(self.FULLY_CONNECTED_INPUTS, 100)
        self.architecture.add_module("fc1", self.fc1)
        self._print_layer_output_shape("fc1", self.architecture)

        self.relu3 = nn.ReLU()
        self.architecture.add_module("relu3", self.relu3)
        self._print_layer_output_shape("relu3", self.architecture)

        ###############################
        # Output Layer
        ###############################
        self._nice_print("Output Layer")

        self.fc2 = nn.Linear(100, 10)  # 10 classes
        self.architecture.add_module("fc2", self.fc2)
        self._print_layer_output_shape("fc2", self.architecture)

    def _print_layer_output_shape(self, name, model_in):
        print(f"Output shape after {name}: {model_in(self.dummy_input).shape}")

    def _get_layer_output_shape(self, model_in):
        return model_in(self.dummy_input).shape

    def _nice_print(self, string_in):
        border_length = len(string_in) + 4
        top_border = "*" * border_length
        bottom_border = "-" * border_length

        print(top_border)
        print(f"* {string_in} *")
        print(bottom_border)

    def forward(self, x):
        x = self.architecture(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=0.01)
        return opt

    def prepare_data(self):
        # download data
        datasets.MNIST(
            data_folder, train=True, download=True, transform=transforms.ToTensor()
        )
        datasets.MNIST(
            data_folder, train=False, download=True, transform=transforms.ToTensor()
        )

    def train_dataloader(self):
        mnist_train = datasets.MNIST(
            data_folder, train=True, download=False, transform=transforms.ToTensor()
        )
        loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True)
        return loader

    def val_dataloader(self):
        mnist_test = datasets.MNIST(
            data_folder, train=False, download=False, transform=transforms.ToTensor()
        )
        loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=False)
        return loader

    def test_dataloader(self):
        mnist_test = datasets.MNIST(
            data_folder, train=False, download=False, transform=transforms.ToTensor()
        )
        loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=False)
        return loader

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.log("test_loss", loss)
        return loss


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        metavar="N",
        help="learning rate (default: 0.01)",
    )
    args = parser.parse_args()

    model = CNN()
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="gpu", devices=1)
    trainer.fit(model)
    trainer.test(model)

    # Save model
    torch.save(model.state_dict(), "lightning_cnn_mnist.pt")

    # Load model
    model = CNN()
    model.load_state_dict(torch.load("lightning_cnn_mnist.pt"))
    model.eval()

    # Test model
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            data_folder, train=False, download=False, transform=transforms.ToTensor()
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    """
    # Test model on single image
    img = Image.open('test.png').convert('L')
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    output = model(img)
    pred = output.argmax(dim=1, keepdim=True)
    print(f'Prediction: {pred.item()}')
    """

    """
    # Test model on multiple images
    img = Image.open('test.png').convert('L')
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    output = model(img)
    pred = output.argmax(dim=1, keepdim=True)
    print(f'Prediction: {pred.item()}')

    img = Image.open('test2.png').convert('L')
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    output = model(img)
    pred = output.argmax(dim=1, keepdim=True)
    print(f'Prediction: {pred.item()}')
    """
