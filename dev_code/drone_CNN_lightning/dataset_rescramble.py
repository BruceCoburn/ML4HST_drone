"""
Python script to 'rescramble' our dataset to ensure that the training, validation and test sets are different each time
this script is run. This is done by randomly selecting images from the original dataset and placing them into the
training, validation and test sets. NOTE: Right now, this script is only compatible with BINARY CLASSIFICATION.

Note that this script relies on parameters found in config.py
"""

# Import Python-native modules
import os

# Import custom modules
from supplemental_functions import create_folder_if_not_exists, split_images
import config

if __name__ == "__main__":

    # Create variables specific to the user input
    source_folder = config.SOURCE_FOLDER
    destination_folder = config.DESTINATION_FOLDER
    class_a_name = config.CLASS_A_NAME
    class_b_name = config.CLASS_B_NAME
    class_a_folder = os.path.join(source_folder, class_a_name)
    class_b_folder = os.path.join(source_folder, class_b_name)

    print("***************************************************************")
    print(f"class_a_folder: {class_a_folder}")
    print(f"class_b_folder: {class_b_folder}")
    print("***************************************************************")

    # Create the destination folder if it does not exist (and all the
    # associated subfolders)
    create_folder_if_not_exists(destination_folder)
    for folder in ["Train", "Val", "Test"]:
        create_folder_if_not_exists(os.path.join(destination_folder, folder))
        create_folder_if_not_exists(
            os.path.join(destination_folder, folder, class_a_name)
        )
        create_folder_if_not_exists(
            os.path.join(destination_folder, folder, class_b_name)
        )

    # Randomly split the images into training, validation and test sets
    split_images(
        destination_folder,
        class_a_name,
        class_a_folder,
        class_b_name,
        class_b_folder,
        split_ratio=config.SPLIT_RATIO,
        file_extension=config.DATASET_FILE_EXT,
    )
