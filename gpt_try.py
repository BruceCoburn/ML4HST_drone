import cv2
import multiprocessing
from djitellopy import Tello

def video_stream(drone, command_queue):
    # Connect to the Tello drone
    drone.connect()

    # Start the video stream
    drone.streamon()

    while True:
        frame = drone.get_frame_read().frame

        # Display the frame
        cv2.imshow('Video Stream', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if not command_queue.empty():
            command = command_queue.get()
            if command == 'q':
                break
            drone.send_command(command)

    # Stop the video stream and disconnect from the drone
    drone.streamoff()
    drone.disconnect()
    cv2.destroyAllWindows()

def get_user_input(command_queue):
    # User input handling code
    while True:
        user_input = input("Enter a command (takeoff/land/up/down/left/right/flip): ")

        command_queue.put(user_input)

        if user_input == 'q':
            break

if __name__ == '__main__':
    # Create a Tello object
    drone = Tello()

    # Create a multiprocessing queue for interprocess communication
    command_queue = multiprocessing.Queue()

    # Create separate processes for video streaming and user input
    video_process = multiprocessing.Process(target=video_stream, args=(drone, command_queue))
    input_process = multiprocessing.Process(target=get_user_input, args=(command_queue,))

    # Start both processes
    video_process.start()
    input_process.start()

    # Wait for both processes to finish
    video_process.join()
    input_process.join()