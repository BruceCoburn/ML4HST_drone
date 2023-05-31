from threading import Thread
from djitellopy import Tello
import cv2, math, time
import os

tello = Tello()
tello.connect()
tello.streamon()

print(f'battery life: {tello.query_battery()}')

try:
    while True:
        img = tello.get_frame_read().frame
        cv2.imshow('frame', img)
        cv2.waitKey(1) # Wait 1 ms between frames
except KeyboardInterrupt:
    exit(1)
finally:
    print("fin")
