"""
This file contains functions that are used in the a __main__ file, but are not
directly related to the main functionality of the file.
"""

def resize_image_dimensions(image_width, image_height, size_reduction_factor):
    new_width = image_width / size_reduction_factor
    new_height = image_height / size_reduction_factor

    new_width = int(new_width)
    new_height = int(new_height)

    return new_width, new_height
