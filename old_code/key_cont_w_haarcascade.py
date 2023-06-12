from multiprocessing import Process, Value
from djitellopy import Tello
import cv2
import numpy as np
import keyboard
from time import sleep


def video_stream(stop_flag, drone):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    while True:
        frame = drone.get_frame_read().frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow("Tello Stream", frame)

        # Exit the loop and end the stream if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    drone.streamoff()
    drone.end()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    stop_flag = Value('i', 0)

    drone = Tello()
    drone.connect()
    drone.streamon()
    print(drone.get_battery())

    video_process = Process(target=video_stream, args=(stop_flag, drone))

    video_process.start()
    sleep(10)
    while(1):
        if keyboard.is_pressed('t'):    # Takeoff
            drone.takeoff()
        elif keyboard.is_pressed('w'):  # Move forward
            drone.send_rc_control(0, 70, 0, 0)
        elif keyboard.is_pressed('s'):  # Move backward
            drone.send_rc_control(0, -70, 0, 0)
        elif keyboard.is_pressed('a'):  # Move left
            drone.send_rc_control(-70, 0, 0, 0)
        elif keyboard.is_pressed('d'):  # Move right
            drone.send_rc_control(70, 0, 0, 0)
        elif keyboard.is_pressed('q'):  # Rotate counter-clockwise
            drone.send_rc_control(0, 0, 0, 70)
        elif keyboard.is_pressed('e'):  # Rotate clockwise
            drone.send_rc_control(0, 0, 0, -70)
        elif keyboard.is_pressed('r'):  # Move up
            drone.send_rc_control(0, 0, 70, 0)
        elif keyboard.is_pressed('f'):  # Move down
            drone.send_rc_control(0, 0, -70, 0)
        elif keyboard.is_pressed('l'):  # Land
            drone.land()
        else:
            drone.send_rc_control(0, 0, 0, 0)

    drone.streamoff()
    drone.end()
    cv2.destroyAllWindows()
    video_process.join()
