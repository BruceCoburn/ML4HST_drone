"""
This Python script aims to connect to a Tello Drone and save images from its camera to a local directory.

The user still maintains control of the drone, but it is recommended to hold and rotate the drone while the
camera feed is being saved. This will allow for more distinct control while obtaining the dataset.
"""

# Import python-native modules
import time

# Import custom modules
from VideoStreamTello import VideoStreamTello
import config
from supplemental_functions import nice_print

# Main script execution
if __name__ == "__main__":
    # Start timing how long this script takes to run
    start_time = time.time()

    # Create VideoStreamTello() object and automatically start the video stream and user input polling
    ########################################################################
    tello_video_stream = VideoStreamTello(save_images=True)
    ########################################################################

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

    if config.SAVE_IMAGES:
        # Print our 'saving images' information
        nice_print(
            f"Wrote {tello_video_stream.num_images_written} images in {tello_video_stream.time_to_save_imgs_start} seconds"
        )

    # Calculate how long our script takes to run
    end_time = time.time() - start_time

    # Print our ending information
    nice_print(f"Done with main loop in {end_time} seconds...")
