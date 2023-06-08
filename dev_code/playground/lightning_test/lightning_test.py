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

# Import cnn model for torchvision
from torchvision.models import resnet18

import pytorch_lightning as pl

from PIL import Image

from torchsummary import summary

class CNN(pl.LightningModule):
    """
    A lot of this code was generated from the PyTorch Lightning documentation:
    https://lightning.ai/docs/pytorch/stable/

    TODO: Add more comments
    """
    def __init__(self):
        super(CNN, self).__init__()
        # 1 input image channel, 10 output channels, 3x3 square convolution
        # kernel

        # Convolution Layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Convolution Layer 2
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Flatten Layer
        self.flatten = nn.Flatten()

        # Fully Connected Layer 1
        self.FULLY_CONNECTED_INPUTS = self.flatten(torch.ones(1, 20, 5, 5)).shape[1]
        self.fc1 = nn.Linear(self.FULLY_CONNECTED_INPUTS, 100)
        self.relu3 = nn.ReLU()

        # Output Layer
        self.fc2 = nn.Linear(100, 10)  # 10 classes

        self.dummy_input = torch.ones(64, 1, 28, 28)

        self._print_layer_output_shape()

    def nice_shape_print(self, layer_name, module_names, tensors):

        border_length = len(layer_name) + 4
        top_border = '*' * border_length
        bottom_border = '-' * border_length

        print(top_border)
        print(f'* {layer_name} *')
        print(bottom_border)
        for idx, name in enumerate(module_names):
            print(name, 'output shape:', tensors[idx].shape)

    def calculate_layer(self, first_input, first_layer, *args):
        outputs = []
        output = first_layer(first_input)
        outputs.append(output)

        for arg in args:
            output = arg(output)
            outputs.append(output)

        return outputs

    def _print_layer_output_shape(self):

        # Dummy input
        self.nice_shape_print('Dummy Input', ['Input shape'], [self.dummy_input])

        # Convolution Layer 1
        modules = ['conv1', 'relu1', 'maxpool1']
        outputs = self.calculate_layer(self.dummy_input, self.conv1, self.relu1, self.maxpool1)
        self.nice_shape_print('Convolution Layer 1', modules, outputs)

        # Convolution Layer 2
        modules = ['conv2', 'relu2', 'maxpool2']
        outputs = self.calculate_layer(outputs[-1], self.conv2, self.relu2, self.maxpool2)
        self.nice_shape_print('Convolution Layer 2', modules, outputs)

        # Flatten Layer
        modules = ['flatten']
        outputs = [self.flatten(outputs[-1])]

        # Fully Connected Layer 1
        modules = ['fc1', 'relu3']
        outputs = self.calculate_layer(outputs[-1], self.fc1, self.relu3)
        self.nice_shape_print('Fully Connected Layer 1', modules, outputs)

        # Output Layer
        modules = ['fc2']
        outputs = [self.fc2(outputs[-1])]
        self.nice_shape_print('Output Layer', modules, outputs)

    def forward(self, x):
        # Convolution Layer 1
        x = self.maxpool1(self.relu1(self.conv1(x)))

        # Convolution Layer 2
        x = self.maxpool2(self.relu2(self.conv2(x)))

        # Flatten Layer
        x = self.flatten(x)

        # Fully Connected Layer 1
        x = self.relu3(self.fc1(x))

        # Output Layer
        x = self.fc2(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=0.01)
        return opt

    def prepare_data(self):
        # download data
        datasets.MNIST('./data_lightning', train=True, download=True, transform=transforms.ToTensor())
        datasets.MNIST('./data_lightning', train=False, download=True, transform=transforms.ToTensor())

    def train_dataloader(self):
        mnist_train = datasets.MNIST('./data_lightning', train=True, download=False, transform=transforms.ToTensor())
        loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True)
        return loader

    def val_dataloader(self):
        mnist_test = datasets.MNIST('./data_lightning', train=False, download=False, transform=transforms.ToTensor())
        loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=False)
        return loader

    def test_dataloader(self):
        mnist_test = datasets.MNIST('./data_lightning', train=False, download=False, transform=transforms.ToTensor())
        loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=False)
        return loader

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.log('test_loss', loss)
        return loss

if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=0.01, metavar='N',
                        help='learning rate (default: 0.01)')
    args = parser.parse_args()

    model = CNN()
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator='gpu', devices=1)
    trainer.fit(model)
    trainer.test(model)

    # Save model
    torch.save(model.state_dict(), 'lightning_cnn_mnist.pt')

    # Load model
    model = CNN()
    model.load_state_dict(torch.load('lightning_cnn_mnist.pt'))
    model.eval()

    # Test model
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data_lightning', train=False, download=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

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
