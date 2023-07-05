"""
This Python script aims to load and utilize an inference model to classify images from a Tello drone's camera.
"""

# Import python-native modules
import time
import queue

# Import custom modules
from VideoStreamTello import VideoStreamTello
import config
from supplemental_functions import nice_print
from CommandPopup import CommandPopup


# Main script execution
if __name__ == "__main__":
    # Start timing how long this script takes to run
    start_time = time.time()

    # Create VideoStreamTello() object and automatically start the video stream and user input polling
    ########################################################################
    # Ensure that config.SAVE_IMAGES is set your preference (True/False) PRIOR to running this script
    # Ensure that config.AUTO_CONTROL is set your preference (True/False) PRIOR to running this script
    tello_video_stream = VideoStreamTello(
        save_images=config.SAVE_IMAGES,
        auto_control=config.AUTO_CONTROL,
        run_inference=config.RUN_INFERENCE,
    )
    ########################################################################

    command_queue = queue.Queue()
    command_popup = CommandPopup(command_queue)

    # Enter our main execution loop (can only be exited via a user input
    # 'kill' or KeyboardInterrupt)
    while tello_video_stream.main_loop:
        command_popup.window.update()
        try:
            # tello_video_stream.poll_keystrokes()
            command = command_queue.get_nowait()
            print(f"Handing off command: {command}")
            tello_video_stream.get_button_command(command)
        except queue.Empty:
            pass
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

    # Destroy command popup window
    command_popup.window.destroy()

    if config.SAVE_IMAGES:
        # Print our 'saving images' information
        nice_print(
            f"Wrote {tello_video_stream.num_images_written} images in {tello_video_stream.time_to_save_imgs_start} seconds"
        )

    # Calculate how long our script takes to run
    end_time = time.time() - start_time

    # Print our ending information
    nice_print(f"Ran Tello code for {end_time} seconds...")
