from djitellopy import Tello
import cv2
import time
from threading import Thread
from multiprocessing import Process, Manager


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

        # Get the current video feed frame and convert into an image (for display purposes)
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

        # Threading is necessary to concurrently display the live video feed and get keystrokes from user
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
                # Get the current image frame from the video feed and display in a popup window
                self.camera_frame = self.tello.get_frame_read()
                self.img = self.camera_frame.frame
                cv2.imshow(self.window_name, self.img)
                cv2.waitKey(
                    1
                )  # 'waitKey' is necessary to properly display a cv2 popup window
            except KeyboardInterrupt:
                break

        # Once we are no longer interested in streaming, land the tello and exit out of all windows
        self.kill_sequence()

    def poll_keystrokes(self, command_queue, main_loop_flag):
        """
        Method to capture user input (for tello-based movements)
        """
        while main_loop_flag.value:
            command = input("Enter input: ")

            if command == "kill":
                command_queue.put("kill")
            else:
                command_queue.put(command)

    def process_commands(self, command_queue, main_loop_flag):
        """
        Method to process the commands received from the queue
        """
        while main_loop_flag.value:
            if not command_queue.empty():
                command = command_queue.get()

                if command == "kill":
                    self.kill_sequence()
                elif command == "w":
                    self.tello.move_forward(self.unit_dp)
                elif command == "s":
                    self.tello.move_back(self.unit_dp)
                elif command == "a":
                    self.tello.move_left(self.unit_dp)
                elif command == "d":
                    self.tello.move_right(self.unit_dp)
                elif command == "e":
                    self.tello.rotate_clockwise(self.unit_dp)
                elif command == "q":
                    self.tello.rotate_counter_clockwise(self.unit_dp)
                elif command == "r":
                    self.tello.move_up(self.unit_dp)
                elif command == "f":
                    self.tello.move_down(self.unit_dp)
                elif command == "l":
                    self.tello.land()
                    self.landed = True
                elif command == "t" and self.landed:
                    self.tello.takeoff()
                    self.landed = False
                elif command == "diag":
                    self.diag()
                else:
                    print(f"command: {command}")

    def diag(self):
        print(f"stream: {self.stream}")
        print(f"landed: {self.landed}")
        print(f"main_loop: {self.main_loop}")

    def kill_sequence(self):
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
    manager = Manager()
    command_queue = manager.Queue()
    main_loop_flag = manager.Value("b", True)

    tello_video_stream = VideoStreamTello()

    command_process = Process(
        target=tello_video_stream.poll_keystrokes, args=(command_queue, main_loop_flag)
    )
    command_process.start()

    tello_video_stream.process_commands(command_queue, main_loop_flag)

    command_process.join()

    print(f"done with main loop...")
