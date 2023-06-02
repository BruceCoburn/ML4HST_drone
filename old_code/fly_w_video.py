from djitellopy import Tello
import cv2
from threading import Thread

tello = Tello()
tello.connect()

tello.streamon()
frame_read = tello.get_frame_read()

print(f'Battery Life: {tello.query_battery()}%')

tello.takeoff()

unit_dp = 30
window_name = "Drone Camera"

def landSequence(tello_obj, window_to_destroy):
    tello_obj.land()
    tello_obj.streamoff()
    cv2.destroyWindow(window_to_destroy)
    cv2.destroyAllWindows()

try: 
    while True:
        
        # Get the current frame from the drone
        img = frame_read.frame
        # Display the drone camera frame from a popup window
        cv2.imshow(window_name, img)
        
        # Get the current key input
        key = cv2.waitKey(1) & 0xff
        
        # If ESC is pressed, then initiate the landing sequence and close all windows
        if key == 27: # ESC
            break
        
        # ord(...) returns the unicode equivalent of a given string
        elif key == ord('w'):
            tello.move_forward(unit_dp)
        elif key == ord('s'):
            tello.move_back(unit_dp)
        elif key == ord('a'):
            tello.move_left(unit_dp)
        elif key == ord('d'):
            tello.move_right(unit_dp)
        elif key == ord('e'):
            tello.rotate_clockwise(unit_dp)
        elif key == ord('q'):
            tello.rotate_counter_clockwise(unit_dp)
        elif key == ord('r'):
            tello.move_up(unit_dp)
        elif key == ord('f'):
            tello.move_down(unit_dp)
        elif key == ord('l'):
            tello.land()
            break
except KeyboardInterrupt:
    
    landSequence(tello_obj=tello, window_to_destroy=window_name)
    
finally:
    
    landSequence(tello_obj=tello, window_to_destroy=window_name)
