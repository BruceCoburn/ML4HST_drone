import cv2
from djitellopy import Tello
from multiprocessing import Process, Manager


class VideoStreamTello(object):
    def __init__(self, unit_dp=30, window_name="Drone Camera"):
        # Establish Tello() object
        self.tello = Tello()

        # Connect to the tello
        self.tello.connect()

        # Query and print out the battery percentage
        self.query_battery()

        # Get the current video feed frame and convert into an image (for display purposes)
        self.img = None

        # Establish object attributes
        self.unit_dp = unit_dp          # Length of spatial displacement
        self.window_name = window_name  # Name of the video stream popup window
        self.landed = True              # Boolean flag to determine whether the tello is on the ground
        self.stream = True              # Boolean flag to determine whether the tello should be streaming or not
        self.popup = True
        self.main_loop = True

    def query_battery(self):
        """
        Method to query and print the current battery percentage of the tello
        """
        print(f'Battery Life: {self.tello.query_battery()}%')

    def update_frame(self, img_queue, stream_flag):
        """
        Method to update the live video feed from the tello (process-based)
        """
        camera = cv2.VideoCapture('udp://@0.0.0.0:11111')

        while stream_flag.value:
            ret, frame = camera.read()

            if ret:
                img_queue.put(frame)

        camera.release()

    def process_commands(self, command_queue, main_loop_flag):
        """
        Method to process the commands received from the queue
        """
        while main_loop_flag.value:
            if not command_queue.empty():
                command = command_queue.get()

                if command == 'kill':
                    self.kill_sequence()
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
                elif command == 't' and self.landed:
                    self.tello.takeoff()
                    self.landed = False
                elif command == 'diag':
                    self.diag()
                else:
                    print(f'command: {command}')

    def diag(self):
        print(f'stream: {self.stream}')
        print(f'landed: {self.landed}')
        print(f'main_loop: {self.main_loop}')

    def kill_sequence(self):
        print(f'killing...')

        if self.main_loop:
            self.main_loop = False

        if self.stream:
            self.stream = False

        if not self.landed:
            self.tello.land()
            self.landed = True

        if self.popup:
            cv2.destroyWindow(self.window_name)
            self.popup = False


if __name__ == "__main__":
    manager = Manager()
    command_queue = manager.Queue()
    main_loop_flag = manager.Value('b', True)
    img_queue = manager.Queue()

    tello_video_stream = VideoStreamTello()

    video_process = Process(target=tello_video_stream.update_frame, args=(img_queue, main_loop_flag))
    video_process.start()

    command_process = Process(target=tello_video_stream.process_commands, args=(command_queue, main_loop_flag))
    command_process.start()

    while tello_video_stream.main_loop:
        if not img_queue.empty():
            frame = img_queue.get()
            cv2.imshow(tello_video_stream.window_name, frame)
            cv2.waitKey(1)

        try:
            command = input("Enter input: ")
            command_queue.put(command)
        except KeyboardInterrupt:
            print(f'!!!Interrupted!!!')
            main_loop_flag.value = False
            command_process.join()
            video_process.join()
            break

    print(f'done with main loop...')
