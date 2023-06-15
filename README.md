# ML4HST_drone
Development modules for Drone-related activities during ML4HST University of Wyoming 2023

The structure for this repo will be as follows:

* dev_code/
    + directory containing the "best" working code for various DJI Tello related demos
    + The most up-to-date CNN training script is under drone_CNN_lightning/train.py
* dev_code/quick_tests/
    + quick python scripts to test very minor functionalities

* old_code/
    + directory containing python scripts at various stages of production. These scripts should NOT be used but can be referenced for certain behaviors

* tello/
    + exported virtual environment to align on the Python Packages used

Drone Obstacle Dataset (./DRONE_OBSTACLES) can be found and downloaded from the following Google Drive link: https://drive.google.com/drive/folders/12EsJg-sO3LIuRX_C-Lc-aXwpTvWMLvBK?usp=sharing
+ This automatically gives sub-directories of Train/ Test/ and Val/ for easier usage of datasets.ImageFolder() object
+ Put this directory in the same directory as ~/ML4HST/dev_code/drone_CNN_lightning/train.py (for now)

Raw Drone Obstacle Dataset (./obstacle_dataset) can be found and downloaded from the following Google Drive link: https://drive.google.com/drive/folders/1BLVCnFsPHkJczSPstAs8UnKTrwB9Cd4x?usp=sharing
+ This is a more 'raw' version of the dataset (only two sub-directories 'BLOCKED/' and 'UNBLOCKED/' corresponding to our two classes)
+ If you download the dataset through this method, you first need to utilize ~/ML4HST/dev_code/drone_CNN_lightning/dataset_rescramble.py to partition our dataset into corresponding
+   Train/ Test/ and Val/ sub-directories. Once this has been done, then we can use ~/ML4HST/dev_code/drone_CNN_lightning/train.py
