################################################################################
# This script is to automatically iterate through a directory and use
# the pep8 standard on Python files
# +++ It should be noted that a command line argument containing the
#     path to the directory that you want to be "pep-ified" should be provided
# +++ Ensure that when running this script you have ADMIN privileges
#       "sudo" for Linux, "Run as Administrator" for Windows...
#
# This script can be run from the command line as follows:
# python auto_autopep8.py <directory_path>
################################################################################

import os
import subprocess
import sys

# Check if the directory path is provided as a command-line argument
if len(sys.argv) < 2:
    print("Please provide the directory path as a command-line argument.")
    sys.exit(1)

# Get the directory path from the command-line argument
directory_path = sys.argv[1]

directory_path = os.path.abspath(directory_path)

if sys.platform.startswith("win"):
    directory_path = os.path.normpath(directory_path)
    print(f"Windows directory path: {directory_path}")

# Validate if the provided path is a directory
if not os.path.isdir(directory_path):
    print("Invalid directory path.")
    sys.exit(1)

# Recursively iterate through the directory and its subdirectories
for root, dirs, files in os.walk(directory_path):
    for file in files:
        # Check if the file is a Python file
        if file.endswith(".py"):
            file_path = os.path.join(root, file)
            print(f"Formatting file: {file_path}")

            # Run autopep8 on the file
            subprocess.run(["autopep8", "--in-place", "--aggressive", file_path])
