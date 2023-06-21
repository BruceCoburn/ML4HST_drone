"""
This Python script aims to load and utilize an inference model to classify images from a Tello drone's camera.
"""

# Import python-native modules
import datetime
from djitellopy import Tello
import cv2
import time
from threading import Thread
import os

# Import custom modules
from VideoStreamTello import VideoStreamTello


# Main script execution
if __name__ == "__main__":
    # Start timing how long this script takes to run
    start_time = time.time()

    # Create VideoStreamTello() object
    tello_video_stream = VideoStreamTello(save_images=False)

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
            tello_video_stream.inference_t.join()

    # Calculate how long our script takes to run
    end_time = time.time() - start_time

    # Print our ending information
    print(
        f"Wrote {tello_video_stream.num_images_written} images in {tello_video_stream.time_to_save_imgs_end} seconds"
    )
    print(f"done with main loop in {end_time} seconds...")
