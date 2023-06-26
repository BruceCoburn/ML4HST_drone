# Import relevant modules
import datetime
from djitellopy import Tello
import cv2
import time
from threading import Thread
import os
import torch
from torchvision import transforms

# Import custom modules
import config
from CNN_lightning import CNN_lightning
from supplemental_functions import resize_image_dimensions


class VideoStreamTello(object):
    def __init__(
        self,
        unit_dp=30,
        window_name="Drone Camera",
        run_inference=True,
        save_images=True,
        inference_model_filepath=config.TORCH_MODEL_FILENAME,
    ):
        if inference_model_filepath is None:
            raise ValueError(
                "inference_model cannot be None, please include a filepath to the inference model"
            )

        # Load our inference model
        self.inference_model = CNN_lightning(
            num_dummy_images=config.NUM_DUMMY_IMAGES,
            num_channels=config.NUM_CHANNELS,
            image_width=config.IMAGE_WIDTH,
            image_height=config.IMAGE_HEIGHT,
        )
        print(f"Load model: {inference_model_filepath}")
        self.inference_model.load_state_dict(torch.load(inference_model_filepath))
        self.inference_model.eval()

        # Establish Tello() object
        self.tello = Tello()

        # Connect to the tello
        self.tello.connect()

        # Query and print out the battery percentage
        self.query_battery()

        # Create a command dictionary where we expect a single keyword
        self.state_dictionary = {
            "kill": self.kill_sequence,
            "l": self.initiate_land,
            "t": self.initiate_takeoff,
            "diag": self.diag,
        }

        # Create a command dictionary where we expect a single keyword AND a
        # parameter
        self.movement_dictionary = {
            "w": self.tello.move_forward,
            "s": self.tello.move_back,
            "a": self.tello.move_left,
            "d": self.tello.move_right,
            "e": self.tello.rotate_clockwise,
            "q": self.tello.rotate_counter_clockwise,
            "r": self.tello.move_up,
            "f": self.tello.move_down,
        }

        # Turn on the video stream from the tello
        self.tello.streamon()

        # Get the current video feed frame and convert into an image (for
        # display purposes)
        self.camera_frame = self.tello.get_frame_read()

        self.img = self.camera_frame.frame
        self.most_recent_image = self.img

        self.num_images_written = 0
        self.time_to_save_imgs_start = 0
        self.time_to_save_imgs_end = 0

        # Establish object attributes
        self.unit_dp = unit_dp  # Length of spatial displacement
        self.window_name = window_name  # Name of the video stream popup window
        self.image_refresh_rate = (
            config.IMAGE_REFRESH_RATE
        )  # How often to refresh the video stream
        # Boolean flag to determine whether the tello is on the ground
        self.landed = True
        # Boolean flag to determine whether the tello should be streaming or
        # not
        self.stream = True
        self.popup = True
        self.main_loop = True

        self.save = save_images
        self.run_inference = run_inference

        # Setting some attributes which will be necessary for saving frames
        # from the camera feed
        self.base_directory = "raw_data"
        self.image_extension = ".jpg"

        # These attributes are also necessary for saving frames from the camera
        # feed, but will be altered from other methods
        self.existing_runs = None
        self.run_number = None
        self.directory_name = None
        self.timestamp = None
        self.filename = None
        self.image_path = None

        # Image save based methods
        self.instantiate_base_directory()
        self.check_for_run_directories()

        # Threading is necessary to concurrently display the live video feed, get keystrokes from user, and save
        # images from the camera feed
        self.video_stream_t = Thread(target=self.update_frame, args=())
        self.video_stream_t.start()

        self.image_save_t = Thread(target=self.image_save, args=())
        self.image_save_t.start()

        self.inference_t = Thread(target=self.run_through_inference, args=())
        self.inference_t.start()

    def run_through_inference(self):
        """
        Method to collect the most recently saved image from the camera feed, and feed it to the inference model
        """
        while self.run_inference:
            # Resize the image to the dimensions expected by the inference model
            image_width, image_height = resize_image_dimensions(
                image_width=config.IMAGE_WIDTH,
                image_height=config.IMAGE_HEIGHT,
                size_reduction_factor=config.SIZE_REDUCTION_FACTOR,
            )

            # Create the resize transform
            resize_transform = transforms.Resize((image_width, image_height))

            # Apply the transform to the image prior to feeding it to the inference model
            resized_image = resize_transform(self.most_recent_image)

            # Feed the image to the inference model
            blocked_or_unblocked = self.inference_model(resized_image)
            blocked_or_unblocked = round(blocked_or_unblocked.item(), 4)

            # self.nice_print(f'p(blocked_or_unblocked): {blocked_or_unblocked}')
            self._inline_print(f"p(blocked_or_unblocked): {blocked_or_unblocked}")

            # Wait for a bit before trying again
            time.sleep(self.image_refresh_rate)

    @staticmethod
    def resize_image_dimensions(
        image_width, image_height, size_reduction_factor, verbose=False
    ):
        """
        This function takes in original image dimensions and returns the new
        image dimensions after applying a size reduction factor.
        """
        new_width = image_width / size_reduction_factor
        new_height = image_height / size_reduction_factor

        new_width = int(new_width)
        new_height = int(new_height)

        if verbose:
            print(f"========================================================")
            print(f"\tsize_reduction_factor: {size_reduction_factor}")
            print(f"Resizing image_width from {image_width} to {new_width}")
            print(f"Resizing image_height from {image_height} to {new_height}")
            print(f"========================================================")

        return new_width, new_height

    def instantiate_base_directory(self):
        """
        Method to establish where you want your 'run' directories to be stored (for camera feed images)
        """

        if not os.path.exists(self.base_directory):
            self.nice_print(f'Creating "{self.base_directory}" directory')
            os.makedirs(self.base_directory)

    def check_for_run_directories(self):
        """
        Method to create a directory corresponding to the current run within a 'base_directory'
        """

        # Check for existing run directories
        self.existing_runs = [
            dir_name
            for dir_name in os.listdir(self.base_directory)
            if dir_name.startswith("run")
        ]

        # Determine the run number
        self.run_number = len(self.existing_runs) + 1
        self.nice_print(f"Number of existing run directories: {self.run_number}")

        # Establish the new directory name
        self.directory_name = f"run{self.run_number:03}"

        # Check if the directory already exists
        while self.directory_name in self.existing_runs:
            self.nice_print(
                f'"{self.directory_name}" already exists. Incrementing run number...'
            )
            self.run_number += 1
            self.directory_name = f"run{self.run_number:03}"

        # Create the new directory
        self.nice_print(f"Creating {self.directory_name}...")
        os.makedirs(
            os.path.join(self.base_directory, self.directory_name), exist_ok=True
        )

    def image_save(self):
        """
        Method to save images from the Tello Camera feed
        """
        self.time_to_save_imgs_start = time.time()

        while self.save:
            try:
                # Create timestamp which will be used for the saved image
                # filename (creates timestamp down to the millisecond)
                self.timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")

                # Generate the filename using the timestamp
                self.filename = self.timestamp + self.image_extension

                # Set the path for the new image
                self.image_path = os.path.join(
                    self.base_directory, self.directory_name, self.filename
                )

                # Save the image in the new directory
                cv2.imwrite(self.image_path, self.img)
                self.num_images_written += 1

                ###############################################################
                # Adjust this sleep parameter to change the number of images saved per second
                # Ex: time.sleep(0.1) will (roughly) save 10 images per second
                ###############################################################
                time.sleep(self.image_refresh_rate)
                ###############################################################

            except KeyboardInterrupt:
                break

        self.time_to_save_imgs_end = time.time() - self.time_to_save_imgs_start

    @staticmethod
    def nice_print(string):
        """
        Method for a nice print of a '*' lined border!
        """
        border_length = len(string) + 4
        border = "*" * border_length

        print(border)
        print(f"* {string} *")
        print(border)

    def query_battery(self):
        """
        Method to query and print the current battery percentage of the tello
        """
        print(f"Battery Life: {self.tello.query_battery()}%")

    def update_frame(self):
        """
        Method to update the live video feed from the tello (thread-based)
        """
        while self.stream:
            try:
                # Get the current image frame from the video feed and display
                # in a popup window
                self.camera_frame = self.tello.get_frame_read()
                self.img = self.camera_frame.frame
                self.most_recent_image = self.img
                cv2.imshow(self.window_name, self.img)
                # 'waitKey' is necessary to properly display a cv2 popup window
                cv2.waitKey(1)
            except KeyboardInterrupt:
                break

    def initiate_land(self):
        """
        Method to land the tello and set the 'landed' attribute to True
        """
        self.tello.land()
        self.landed = True

    def initiate_takeoff(self):
        """
        Method to have the tello takeoff and set the 'landed' attribute to False
        """
        self.tello.takeoff()
        self.landed = False

    def poll_keystrokes(self):
        """
        Method to capture user input (for tello-based movements)
        """
        command = input("Enter command (and argument(s)): ")

        # Split the input into separate strings
        command_list = command.split()

        # Check whether the input should be decoded using 'state_dictionary' (single keyword)
        # or using 'movement_dictionary' (single keyword and parameter)
        if len(command_list) == 1:
            try:
                requested_command = self.state_dictionary[command_list[0]]
                print(f"Calling {requested_command.__name__}")
                requested_command()
            except KeyError:
                print(f"Attempted the following command: {command_list}")
        elif len(command_list) == 2:
            try:
                requested_command = self.movement_dictionary[command_list[0]]
                print(f"Calling {requested_command.__name__} {command_list[1]}")
                requested_command(int(command_list[1]))
            except KeyError:
                print(f"Attempted the following command: {command_list}")
        else:
            print(f"Unrecognized inputs: {command_list}")

        # Get remaining battery percentage after each command has been sent
        self.query_battery()

    def diag(self):
        """
        Method to print out the current state of various Boolean values
        """
        print(f"stream: {self.stream}")
        print(f"landed: {self.landed}")
        print(f"main_loop: {self.main_loop}")
        print(f"save: {self.save}")
        print(f"run_inference: {self.run_inference}")

    def kill_sequence(self):
        """
        Method to completely stop all Tello operations other than the connection
        """

        print(f"killing main loop...")
        if self.main_loop:
            self.main_loop = False

        print(f"killing stream...")
        if self.stream:
            self.tello.streamoff()
            self.stream = False

        print(f"killing landing...")
        if not self.landed:
            self.tello.land()
            self.landed = True

        print(f"killing popups...")
        if self.popup:
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            self.popup = False

        print(f"killing save state...")
        if self.save:
            self.save = False

        print(f"killing collect data state...")
        if self.run_inference:
            self.run_inference = False

    @staticmethod
    def _inline_print(string, verbose=True):
        """
        Method to print a string inline
        """
        if verbose:
            # Clear the line
            print_string = "\b" * len(string)
            print(print_string, end="", flush=True)
            print(string, end="", flush=True)
