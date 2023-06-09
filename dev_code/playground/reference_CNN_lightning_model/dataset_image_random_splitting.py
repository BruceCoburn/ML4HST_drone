"""
Python script to automatically split images into training, validation and test sets. This occurs randomly and depends
on the split ratio. The script should be able to be run from the command line as follows:



"""

import os
import shutil
import random
import argparse
import time

file_extension = '.jpg'

def split_images(destination_folder, class_a_name, class_a_folder, class_b_name, class_b_folder, split_ratio=(0.7, 0.2, 0.1)):
    # Create destination folders if they don't exist
    for folder in ['Train', 'Val', 'Test']:
        os.makedirs(os.path.join(destination_folder, folder), exist_ok=True)

    # Collect the image files from ClassA and ClassB folders
    class_a_images = [file for file in os.listdir(class_a_folder) if file.endswith(file_extension)]
    class_b_images = [file for file in os.listdir(class_b_folder) if file.endswith(file_extension)]

    # Randomly shuffle the image files
    random.shuffle(class_a_images)
    random.shuffle(class_b_images)

    # Calculate the number of images for each split
    total_images = len(class_a_images) + len(class_b_images)
    train_ratio, val_ratio, test_ratio = split_ratio
    train_count = int(total_images * train_ratio)
    print(f'Train Count: {train_count}')
    val_count = int(total_images * val_ratio)
    print(f'Val Count: {val_count}')
    test_count = total_images - train_count - val_count
    print(f'Test Count: {test_count}')

    number_class_a_images = len(class_a_images)
    number_class_b_images = len(class_b_images)
    class_a_train_count = int(number_class_a_images * train_ratio)
    class_a_val_count = int(number_class_a_images * val_ratio)
    class_a_test_count = number_class_a_images - class_a_train_count - class_a_val_count
    class_b_train_count = int(number_class_b_images * train_ratio)
    class_b_val_count = int(number_class_b_images * val_ratio)
    class_b_test_count = number_class_b_images - class_b_train_count - class_b_val_count

    # Copy images to the destination folders based on the split count
    copy_images(class_a_images[:class_a_train_count], os.path.join(destination_folder, 'Train', class_a_name), class_a_folder)
    print(f'class a train length: {len(class_a_images[:class_a_train_count])}')
    copy_images(class_a_images[class_a_train_count:class_a_train_count + class_a_val_count], os.path.join(destination_folder, 'Val', class_a_name), class_a_folder)
    print(f'class a val length: {len(class_a_images[class_a_train_count:class_a_train_count + class_a_val_count])}')
    copy_images(class_a_images[class_a_train_count + class_a_val_count:class_a_train_count + class_a_val_count + class_a_test_count], os.path.join(destination_folder, 'Test', class_a_name), class_a_folder)
    print(f'class a test length: {len(class_a_images[class_a_train_count + class_a_val_count:class_a_train_count + class_a_val_count + class_a_test_count])}')

    copy_images(class_b_images[:class_b_train_count], os.path.join(destination_folder, 'Train', class_b_name), class_b_folder)
    print(f'class b train length: {len(class_b_images[:class_b_train_count])}')
    copy_images(class_b_images[class_b_train_count:class_b_train_count + class_b_val_count], os.path.join(destination_folder, 'Val', class_b_name), class_b_folder)
    print(f'class b val length: {len(class_b_images[class_b_train_count:class_b_train_count + class_b_val_count])}')
    copy_images(class_b_images[class_b_train_count + class_b_val_count:class_b_train_count + class_b_val_count + class_b_test_count], os.path.join(destination_folder, 'Test', class_b_name), class_b_folder)
    print(f'class b test length: {len(class_b_images[class_b_train_count + class_b_val_count:class_b_train_count + class_b_val_count + class_b_test_count])}')

    print('=============================')
    print('----> Splitting completed')
    print('=============================')

def copy_images(image_list, destination_folder, source_folder):
    # Copy images from the source folder to the destination folder
    for image in image_list:
        source_path = os.path.join(source_folder, image)
        destination_path = os.path.join(destination_folder, image)
        shutil.copyfile(source_path, destination_path)

def delete_images_in_folder(folder_path, image_extension=file_extension):
    print(f'----------> Removing images from {folder_path}...')
    # Delete all images in the folder
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath) and filename.endswith(image_extension):
            os.remove(filepath)

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f'---- Created folder: {folder_path}')
    else:
        print(f'---- Folder already exists: {folder_path}')
        delete_images_in_folder(folder_path)

if __name__ == '__main__':

    # Receive the source folder, destination folder, class A folder and class B folder from the command line
    parser = argparse.ArgumentParser(description='Split images into training, validation and test sets')
    parser.add_argument('--source_folder', type=str, help='Source folder')
    parser.add_argument('--destination_folder', type=str, help='Destination folder')
    parser.add_argument('--class_a_name', type=str, help = 'Name of the first class')
    parser.add_argument('--class_b_name', type=str, help = 'Name of the second class')
    parser.add_argument('--split_ratio', type=float, nargs='+', default=(0.7, 0.2, 0.1), help='Split ratio')

    args = parser.parse_args()

    # Example usage
    source_folder = args.source_folder
    destination_folder = args.destination_folder
    class_a_name = args.class_a_name
    class_b_name = args.class_b_name
    class_a_folder = os.path.join(source_folder, class_a_name)
    class_b_folder = os.path.join(source_folder, class_b_name)

    print(f'class_a_folder: {class_a_folder}')
    print(f'class_b_folder: {class_b_folder}')

    create_folder_if_not_exists(destination_folder)
    for folder in ['Train', 'Val', 'Test']:
        create_folder_if_not_exists(os.path.join(destination_folder, folder))
        create_folder_if_not_exists(os.path.join(destination_folder, folder, class_a_name))
        create_folder_if_not_exists(os.path.join(destination_folder, folder, class_b_name))

    split_images(destination_folder, class_a_name, class_a_folder, class_b_name, class_b_folder, split_ratio=args.split_ratio)
