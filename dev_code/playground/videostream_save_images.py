# Import relevant modules
import datetime
from djitellopy import Tello
import cv2
import time
from threading import Thread
import os


class VideoStreamTello(object):
    def __init__(self, unit_dp=30, window_name="Drone Camera", collect_data=True):
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
        self.num_images_written = 0
        self.time_to_save_imgs_start = 0
        self.time_to_save_imgs_end = 0
        # self.collect_data = collect_data

        # Establish object attributes
        self.unit_dp = unit_dp  # Length of spatial displacement
        self.window_name = window_name  # Name of the video stream popup window
        # Boolean flag to determine whether the tello is on the ground
        self.landed = True
        # Boolean flag to determine whether the tello should be streaming or
        # not
        self.stream = True
        self.popup = True
        self.main_loop = True
        self.save = True

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
                time.sleep(0.1)
                ###############################################################

            except KeyboardInterrupt:
                break

        self.time_to_save_imgs_end = time.time() - self.time_to_save_imgs_start

    def nice_print(self, string):
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


# Main script execution
if __name__ == "__main__":
    # Start timing how long this script takes to run
    start_time = time.time()

    # Create VideoStreamTello() object
    tello_video_stream = VideoStreamTello()

    # Enter our main execution loop (can only be exited via a user input
    # 'kill' or KeyboardInterrupt)
    while tello_video_stream.main_loop:
        try:
            tello_video_stream.poll_keystrokes()
        except KeyboardInterrupt:
            print(f"!!!Interrupted!!!")

            # Stop our main loop
            tello_video_stream.main_loop = False

            # Initiate the kill sequence
            tello_video_stream.kill_sequence()

            # Join our running threads
            tello_video_stream.video_stream_t.join()
            tello_video_stream.image_save_t.join()

    # Calculate how long our script takes to run
    end_time = time.time() - start_time

    # Print our ending information
    print(
        f"Wrote {tello_video_stream.num_images_written} images in {tello_video_stream.time_to_save_imgs_end} seconds"
    )
    print(f"done with main loop in {end_time} seconds...")
