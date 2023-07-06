from djitellopy import Tello
import cv2
import time
from multiprocessing import Process, Manager


def video_stream(queue, stream_flag, tello_queue):
    # Establish Tello() object
    # tello = Tello('', 8895)

    # Connect to the tello
    # tello.connect()

    tello = tello_queue.get()

    # Turn on the video stream from the tello
    tello.streamon()

    while stream_flag.value:
        frame = tello.get_frame_read().frame
        queue.put(frame)

    # Clean up resources
    tello.streamoff()
    tello.land()
    tello.end()


def control_tello(command_queue, tello_queue):
    # Establish Tello() object
    # tello = Tello('', 8889)

    # Connect to the tello
    # tello.connect()

    while True:
        if not command_queue.empty():
            command = command_queue.get()
            tello = tello_queue.get()

            if command == "takeoff":
                tello.takeoff()
            elif command == "land":
                tello.land()
            elif command == "forward":
                tello.move_forward(50)  # Replace '50' with desired distance in cm
            elif command == "backward":
                tello.move_back(50)  # Replace '50' with desired distance in cm
            # Add more commands as needed


def main():
    manager = Manager()
    video_queue = manager.Queue()
    command_queue = manager.Queue()
    stream_flag = manager.Value("b", True)

    tello = Tello()
    tello.connect()

    tello_queue = manager.Queue()
    tello_queue.put(tello)

    video_process = Process(
        target=video_stream, args=(video_queue, stream_flag, tello_queue)
    )
    video_process.start()

    control_process = Process(target=control_tello, args=(command_queue, tello_queue))
    control_process.start()

    while True:
        if not video_queue.empty():
            frame = video_queue.get()
            cv2.imshow("Tello Stream", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # Press 'q' to quit the program
            break
        elif key == ord("t"):  # Press 't' to takeoff
            command_queue.put("takeoff")
        elif key == ord("l"):  # Press 'l' to land
            command_queue.put("land")
        elif key == ord("f"):  # Press 'f' to move forward
            command_queue.put("forward")
        elif key == ord("b"):  # Press 'b' to move backward
            command_queue.put("backward")
        # Add more key mappings as needed

    # Stop the video stream process
    stream_flag.value = False
    video_process.join()

    # Stop the control process
    control_process.terminate()
    control_process.join()

    # Clean up
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
