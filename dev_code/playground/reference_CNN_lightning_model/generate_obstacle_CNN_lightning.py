"""
Create a python script which trains a convolutional neural network on our Drone Obstacle Dataset using PyTorch Lightning
The script should be able to be run from the command line as follows:

python generate_obstacle_CNN_lightning.py --epochs 10 --batch_size 64 --learning_rate 0.01

The script should output the training loss and accuracy to the command line every 100 iterations
"""

# Import Python-native modules
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datetime import datetime

# Define the data folder
data_folder = './DRONE_OBSTACLES'

class CNN(pl.LightningModule):
    """
    A lot of this code was generated from referencing the PyTorch Lightning documentation:
    https://lightning.ai/docs/pytorch/stable/

    TODO: Add more comments
    """

    def __init__(self, num_channels, image_width, image_height):
        super(CNN, self).__init__()
        # 3 input image channel, 10 output channels, 3x3 square convolution
        # kernel

        # Dummy input to calculate the output shape of each layer
        self.dummy_input = torch.ones(1, num_channels, image_width, image_height)

        self.architecture = nn.Sequential()

        ###############################
        # Convolution Layer 1
        ###############################
        self._nice_print('Convolution Layer 1')

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=10,
            kernel_size=3,
            stride=1,
            padding=0)
        self.architecture.add_module('conv1', self.conv1)
        output_shape = self._get_layer_output_shape('conv1', self.architecture)

        self.relu1 = nn.ReLU()
        self.architecture.add_module('relu1', self.relu1)
        output_shape = self._get_layer_output_shape('relu1', self.architecture)

        self.b1 = nn.BatchNorm2d(output_shape[1])
        self.architecture.add_module('b1', self.b1)
        output_shape = self._get_layer_output_shape('b1', self.architecture)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.architecture.add_module('maxpool1', self.maxpool1)
        output_shape = self._get_layer_output_shape('maxpool1', self.architecture)

        ###############################
        # Convolution Layer 2
        ###############################
        self._nice_print('Convolution Layer 2')

        self.conv2 = nn.Conv2d(
            in_channels=10,
            out_channels=20,
            kernel_size=3,
            stride=1,
            padding=0)
        self.architecture.add_module('conv2', self.conv2)
        output_shape = self._get_layer_output_shape('conv2', self.architecture)

        self.relu2 = nn.ReLU()
        self.architecture.add_module('relu2', self.relu2)
        output_shape = self._get_layer_output_shape('relu2', self.architecture)

        self.b2 = nn.BatchNorm2d(output_shape[1])
        self.architecture.add_module('b2', self.b2)
        output_shape = self._get_layer_output_shape('b2', self.architecture)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.architecture.add_module('maxpool2', self.maxpool2)
        output_shape = self._get_layer_output_shape('maxpool2', self.architecture)

        ###############################
        # Flatten Layer
        ###############################
        self._nice_print('Flatten Layer')

        self.flatten = nn.Flatten()
        self.architecture.add_module('flatten', self.flatten)
        output_shape = self._get_layer_output_shape('flatten', self.architecture)

        ###############################
        # Output Layer
        ###############################
        self._nice_print('Output Layer')

        self.FULLY_CONNECTED_INPUTS = self._get_layer_output_shape(name='fc_calc', model_in=self.architecture, print_shape=False)[
            1]
        self.fc1 = nn.Linear(self.FULLY_CONNECTED_INPUTS, 1)
        self.architecture.add_module('fc1', self.fc1)
        output_shape = self._get_layer_output_shape('fc1', self.architecture)

        self.sigmoid = nn.Sigmoid()
        self.architecture.add_module('sigmoid', self.sigmoid)
        output_shape = self._get_layer_output_shape('sigmoid', self.architecture)

    def _get_layer_output_shape(self, name, model_in, print_shape=True):
        if print_shape:
            print(f'Output shape after {name}: {model_in(self.dummy_input).shape}')
        return model_in(self.dummy_input).shape

    def _nice_print(self, string_in):
        border_length = len(string_in) + 4
        top_border = '*' * border_length
        bottom_border = '-' * border_length

        print(top_border)
        print(f'* {string_in} *')
        print(bottom_border)

    def forward(self, x):
        x = self.architecture(x)
        return x

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=0.001)
        return opt

    def _calculate_loss_and_accuracy(self, typeName, logits, y):
        loss = nn.BCELoss()(logits, y)
        preds = torch.round(logits)
        acc = (preds == y).sum().item() / len(preds)
        self.log(f'{typeName}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{typeName}_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        output_dict = {
            f'{typeName}_loss': loss,
            f'{typeName}_accuracy': acc
        }

        return output_dict

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1)
        y = y.float()
        logits = self.forward(x)
        training_dict = self._calculate_loss_and_accuracy('train', logits, y)
        return training_dict['train_loss']

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1)
        y = y.float()
        logits = self.forward(x)
        validation_dict = self._calculate_loss_and_accuracy('val', logits, y)
        return validation_dict['val_loss']

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1)
        y = y.float()
        logits = self.forward(x)
        testing_dict = self._calculate_loss_and_accuracy('test', logits, y)
        return testing_dict['test_loss']


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, image_width, image_height, batch_size=64):
        super(ImageDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((image_width, image_height)),
            transforms.ToTensor(),
        ])

    def prepare_data(self):
        # This method is used for download or preparing data
        # It is called only once for the entire run
        # torchvision.datasets.ImageFolder() does not require this step
        return

    def setup(self, stage=None):
        # This method is used for splitting the dataset into train, validation, and test datasets
        # It is called every time the trainer is initialized or the data module is re-initialized
        self.train_dataset = datasets.ImageFolder(
            root=self.data_dir + '/Train',
            transform=self.transform
        )
        self.val_dataset = datasets.ImageFolder(
            root=self.data_dir + '/Val',
            transform=self.transform
        )
        self.test_dataset = datasets.ImageFolder(
            root=self.data_dir + '/Test',
            transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )

def resize_image_dimensions(image_width, image_height, size_reduction_factor):

    new_width = image_width / size_reduction_factor
    new_height = image_height / size_reduction_factor

    new_width = int(new_width)
    new_height = int(new_height)

    return new_width, new_height


if __name__ == '__main__':

    start_time = datetime.now()
    print(f'Training run started at: {start_time}')

    # Training settings
    parser = argparse.ArgumentParser(description='Drone Obstacle Course CNN Training')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=0.01, metavar='N',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--data_dir', type=str, default='data', metavar='N',
                        help='data directory (default: data)')
    args = parser.parse_args()

    # Set image dimensions
    num_channels = 3
    image_width, image_height = resize_image_dimensions(image_width=960, image_height=720, size_reduction_factor=4)

    # Create an instance of our data module
    dm = ImageDataModule(data_dir=args.data_dir,
                         image_width=image_width,
                         image_height=image_height,
                         batch_size=args.batch_size)

    # Create an instance of our model
    model = CNN(num_channels=num_channels,
                image_width=image_width,
                image_height=image_height)
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator='gpu', devices=1)
    print(f'Training model for {args.epochs} epochs')
    trainer.fit(model, datamodule=dm)
    print(f'Testing model...')
    trainer.test(model, datamodule=dm)

    # Save model
    torch_model_filename = 'drone_obstacle_cnn.pt'
    print(f'Saving model as {torch_model_filename}')
    torch.save(model.state_dict(), torch_model_filename)

    end_time = datetime.now()
    print(f'Training run ended at: {end_time}')
    print(f'Training run duration: {end_time - start_time}')

    """
    start_time = datetime.now()
    print(f'Testing run started at: {start_time}')
    
    # Load model
    model = CNN(num_channels=num_channels,
                image_width=image_width,
                image_height=image_height)
    print(f'Load model: {torch_model_filename}')
    model.load_state_dict(torch.load(torch_model_filename))
    model.eval()
    
    # Test model
    test_result = trainer.test(model, datamodule=dm)

    end_time = datetime.now()
    print(f'Testing run ended at: {end_time}')
    print(f'Testing run duration: {end_time - start_time}')
    """
