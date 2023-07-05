from CommandPopup import CommandPopup
import queue

if __name__ == "__main__":

    command_queue = queue.Queue()
    command_popup = CommandPopup(command_queue)

    command_popup.start()

    while True:
        try:
            command = command_queue.get_nowait()
            print(command)
        except queue.Empty:
            pass
        except KeyboardInterrupt:
            break
    command_popup.window.destroy()
    print("Done")
