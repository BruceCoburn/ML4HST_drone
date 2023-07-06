from CommandPopup import CommandPopup
from VideoStreamTello import VideoStreamTello
import queue
import time
from threading import Thread

if __name__ == "__main__":
    tello_video_stream = VideoStreamTello(
        save_images=False,
        auto_control=False,
        run_inference=False,
    )
    command_queue = queue.Queue()
    command_popup = CommandPopup(command_queue)

    while tello_video_stream.main_loop:
        time.sleep(1)
        command_popup.window.update()
        print(f"In main loop...")
        try:
            command = command_queue.get_nowait()
            print(f"Handing off command: {command}")
            tello_video_stream.get_button_command(command)
        except queue.Empty:
            pass
        except KeyboardInterrupt:
            print(f"!!!Interrupted!!!")
            tello_video_stream.main_loop = False
            tello_video_stream.kill_sequence()
            tello_video_stream.video_stream_t.join()
            tello_video_stream.image_save_t.join()
            tello_video_stream.inference_t.join()
            command_popup.window.destroy()
            break
    # command_popup.window.destroy()
    print("Done")
