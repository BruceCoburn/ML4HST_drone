from djitellopy import Tello
import cv2
import time
from threading import Thread


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
        self.unit_dp = unit_dp  # Length of spatial displacement
        self.window_name = window_name  # Name of the video stream popup window
        self.landed = (
            True  # Boolean flag to determine whether the tello is on the ground
        )
        self.stream = True  # Boolean flag to determine whether the tello should be streaming or not
        self.popup = True
        self.main_loop = True

        ##########################
        # Velocity attributes
        ##########################

        # Velocity constants
        self.velocity_val = 20
        self.no_velocity = 0

        # Velocity params - TODO: Check that these values align with expected
        self.left_velocity = -self.velocity_val
        self.right_velocity = self.velocity_val
        self.forward_velocity = self.velocity_val
        self.backward_velocity = -self.velocity_val
        self.up_velocity = -self.velocity_val
        self.down_velocity = self.velocity_val
        self.left_turn_velocity = -self.velocity_val
        self.right_turn_velocity = self.velocity_val

        # Threading is necessary to concurrently display the live video feed
        # and get keystrokes from user
        self.video_stream_t = Thread(target=self.update_frame, args=())
        self.video_stream_t.start()

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

        # Once we are no longer interested in streaming, land the tello and exit out of all windows
        # self.killSequence()

    def poll_keystrokes(self):
        """
        Method to capture user input (for tello-based movements)
        """
        command = input("Enter input: ")

        if command == "kill":
            print(f"self kill")
            self.killSequence()
        elif command == "w":
            # Move forward
            self.tello.send_rc_control(
                left_right_velocity=self.no_velocity,
                forward_backward_velocity=self.forward_velocity,
                up_down_velocity=self.no_velocity,
                yaw_velocity=self.no_velocity,
            )
        elif command == "s":
            # Move backward
            self.tello.send_rc_control(
                left_right_velocity=self.no_velocity,
                forward_backward_velocity=self.backward_velocity,
                up_down_velocity=self.no_velocity,
                yaw_velocity=self.no_velocity,
            )
        elif command == "a":
            # Move left
            self.tello.send_rc_control(
                left_right_velocity=self.left_velocity,
                forward_backward_velocity=self.no_velocity,
                up_down_velocity=self.no_velocity,
                yaw_velocity=self.no_velocity,
            )
        elif command == "d":
            # Move right
            self.tello.send_rc_control(
                left_right_velocity=self.right_velocity,
                forward_backward_velocity=self.no_velocity,
                up_down_velocity=self.no_velocity,
                yaw_velocity=self.no_velocity,
            )
        elif command == "e":
            # Turn right
            self.tello.send_rc_control(
                left_right_velocity=self.no_velocity,
                forward_backward_velocity=self.no_velocity,
                up_down_velocity=self.no_velocity,
                yaw_velocity=self.right_turn_velocity,
            )
        elif command == "q":
            # Turn left
            self.tello.send_rc_control(
                left_right_velocity=self.no_velocity,
                forward_backward_velocity=self.no_velocity,
                up_down_velocity=self.no_velocity,
                yaw_velocity=self.left_turn_velocity,
            )
        elif command == "stop":
            # Stop all movement
            self.tello.send_rc_control(
                left_right_velocity=self.no_velocity,
                forward_backward_velocity=self.no_velocity,
                up_down_velocity=self.no_velocity,
                yaw_velocity=self.no_velocity,
            )
        elif command == "r":
            # Move up
            self.tello.move_up(self.unit_dp)
        elif command == "f":
            # Move down
            self.tello.move_down(self.unit_dp)
        elif command == "l":
            # Land
            self.tello.land()
            self.landed = True
        elif (command == "t") and (self.landed == True):
            # Takeoff
            self.tello.takeoff()
            self.landed = False
        elif command == "diag":
            print(f"diag")
            self.diag()
        else:
            print(f"command: {command}")

    def diag(self):
        print(f"stream: {self.stream}")
        print(f"landed: {self.landed}")
        print(f"main_loop: {self.main_loop}")

    def killSequence(self):
        print(f"killing...")

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


if __name__ == "__main__":
    # Start timing how long this script takes to run
    start_time = time.time()

    # Create a tello object to connect to the tello and initialize the video stream and get user input
    tello_video_stream = VideoStreamTello()

    # Start the video stream and user input threads
    while tello_video_stream.main_loop:
        try:
            tello_video_stream.poll_keystrokes()
        except KeyboardInterrupt:
            print(f"!!!Interrupted!!!")
            tello_video_stream.main_loop = False
            tello_video_stream.killSequence()
            tello_video_stream.video_stream_t.join()

    # End timing how long this script takes to run
    end_time = time.time()

    # Print out how long this script took to run
    print(f"Total runtime: {end_time - start_time} seconds")
