import datetime

from djitellopy import Tello
import cv2
import time
from threading import Thread
import os


class VideoStreamTello(object):
    def __init__(self, unit_dp=30, window_name="Drone Camera"):
        # Establish Tello() object
        self.tello = Tello()

        # Connect to the tello
        self.tello.connect()

        # Query and print out the battery percentage
        self.query_battery()

        # Turn on the video stream from the tello
        self.tello.streamon()

        # Get the current video feed frame and convert into an image (for
        # display purposes)
        self.camera_frame = self.tello.get_frame_read()
        self.img = self.camera_frame.frame

        # Establish object attributes
        self.unit_dp = unit_dp          # Length of spatial displacement
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
        self.base_directory = 'raw_data'
        self.image_extension = '.jpg'

        # These attributes are also necessary for saving frames from the camera
        # feed, but are altered from other methods
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
        self.existing_runs = [dir_name for dir_name in os.listdir(
            self.base_directory) if dir_name.startswith('run')]

        # Determine the run number
        self.run_number = len(self.existing_runs) + 1

        # Create the new directory
        self.directory_name = f'run{self.run_number:03}'

        # Create the new directory
        os.makedirs(
            os.path.join(
                self.base_directory,
                self.directory_name),
            exist_ok=True)

    def image_save(self):
        """
        Method to save images from the Tello Camera feed
        """
        while self.save:
            try:
                # Create timestamp which will be used for the saved image
                # filename
                self.timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

                # Generate the filename using the timestamp
                self.filename = self.timestamp + self.image_extension

                # Set the path for the new image
                self.image_path = os.path.join(
                    self.base_directory, self.directory_name, self.filename)

                # Save the image in the new directory
                cv2.imwrite(self.image_path, self.img)
            except KeyboardInterrupt:
                break

    def nice_print(self, string):
        """
        Method for a nice print of a '*' lined border!
        """
        border_length = len(string) + 4
        border = '*' * border_length

        print(border)
        print(f'* {string} *')
        print(border)

    def query_battery(self):
        """
        Method to query and print the current battery percentage of the tello
        """
        print(f'Battery Life: {self.tello.query_battery()}%')

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

        # Once we are no longer interested in streaming, land the tello and exit out of all windows
        # self.killSequence()

    def poll_keystrokes(self):
        """
        Method to capture user input (for tello-based movements)
        """
        command = input("Enter input: ")

        if command == 'kill':
            print(f'self kill')
            self.killSequence()
        elif command == 'w':
            self.tello.move_forward(self.unit_dp)
        elif command == 's':
            self.tello.move_back(self.unit_dp)
        elif command == 'a':
            self.tello.move_left(self.unit_dp)
        elif command == 'd':
            self.tello.move_right(self.unit_dp)
        elif command == 'e':
            self.tello.rotate_clockwise(self.unit_dp)
        elif command == 'q':
            self.tello.rotate_counter_clockwise(self.unit_dp)
        elif command == 'r':
            self.tello.move_up(self.unit_dp)
        elif command == 'f':
            self.tello.move_down(self.unit_dp)
        elif command == 'l':
            self.tello.land()
            self.landed = True
        elif ((command == 't') and (self.landed == True)):
            self.tello.takeoff()
            self.landed = False
        elif command == 'diag':
            print(f'diag')
            self.diag()
        elif command == 'save':
            cv2.imwrite("sample_image.jpg", self.img)
            print(f'Image Saved!')
        else:
            print(f'command: {command}')

    def diag(self):
        print(f'stream: {self.stream}')
        print(f'landed: {self.landed}')
        print(f'main_loop: {self.main_loop}')
        print(f'save: {self.save}')

    def killSequence(self):
        print(f'killing...')

        if self.main_loop:
            self.main_loop = False

        if self.stream:
            self.tello.streamoff()
            self.stream = False

        if not self.landed:
            self.tello.land()
            self.landed = True

        if self.popup:
            cv2.destroyWindow(self.window_name)
            cv2.destroyAllWindows()
            self.popup = False

        if self.save:
            self.save = False


if __name__ == "__main__":

    tello_video_stream = VideoStreamTello()

    while tello_video_stream.main_loop:
        try:
            tello_video_stream.poll_keystrokes()
        except KeyboardInterrupt:
            print(f'!!!Interrupted!!!')
            tello_video_stream.main_loop = False
            tello_video_stream.killSequence()
            tello_video_stream.video_stream_t.join()

    print(f'done with main loop...')
