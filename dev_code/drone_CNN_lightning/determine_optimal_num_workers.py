"""
This script determines the optimal num_workers value for the DataLoader class in PyTorch.

Run this script prior to running the 3_train.py script to determine the optimal num_workers value for your system. Then,
set the NUM_WORKERS variable in config.py to the optimal value.
"""

# Import Python-native modules
import time
import shutil
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


def measure_data_loading_time(mnist_dir, num_workers):
    """
    Measure the data loading time for the MNIST dataset using the DataLoader class in PyTorch.
    """
    dataset = MNIST(root=mnist_dir, train=True, transform=ToTensor(), download=True)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=num_workers)

    start_time = time.time()
    for _ in dataloader:
        pass
    end_time = time.time()

    execution_time = end_time - start_time

    nice_print(f"num_workers: {num_workers} ||  execution time: {execution_time}")

    return execution_time


def nice_print(string_in):
    """
    Print a string in a nice format
    """
    border_length = len(string_in) + 4
    top_border = "*" * border_length
    bottom_border = "-" * border_length

    print(top_border)
    print(f"* {string_in} *")
    print(bottom_border)


if __name__ == "__main__":
    # Define the directory where the MNIST dataset will be downloaded
    mnist_dir = "data/"

    # Define the range of num_workers values to test
    num_workers_values = [1, 2, 4, 8, 12, 16]

    # Measure the data loading time for each num_workers value
    loading_times = []
    for num_workers in num_workers_values:
        loading_time = measure_data_loading_time(mnist_dir, num_workers)
        loading_times.append(loading_time)

    # Find the optimal num_workers value with the minimum loading time
    optimal_num_workers = num_workers_values[loading_times.index(min(loading_times))]

    nice_print("OPTIMAL NUM_WORKERS:" + str(optimal_num_workers))

    # Delete the MNIST dataset directory
    shutil.rmtree(mnist_dir)
