from djitellopy import Tello
import cv2, time
from threading import Thread

"""
tello = Tello()
tello.connect()

tello.streamon()
frame_read = tello.get_frame_read()

print(f'Battery Life: {tello.query_battery()}%')
"""
Stream = True

class VideoStreamTello(object):
    def __init__(self, unit_dp=30, window_name="Drone Camera"):
        self.tello = Tello()
        self.tello.connect()
        self.query_battery()
        self.tello.streamon()
        self.camera_frame = self.tello.get_frame_read()
        self.img = self.camera_frame.frame
        self.unit_dp = unit_dp
        self.window_name = window_name
        
        self.landed = True
        
        self.video_stream_t = Thread(target=self.update_frame, args=())
        self.keystroke_t = Thread(target=self.poll_keystrokes, args=())
        
        #self.video_stream_t.daemon = True
        #self.keystroke_t.daemon = True
        
        self.video_stream_t.start()
        # self.keystroke_t.start()
        
        """
        self.img = self.camera_frame.frame
        cv2.imshow(self.window_name, self.img)
        """
        
    def query_battery(self):
        print(f'Battery Life: {self.tello.query_battery()}%')
        
    def update_frame(self):
        # Capture the next frame
        while Stream:
            try:
                self.camera_frame = self.tello.get_frame_read()
                self.img = self.camera_frame.frame
                cv2.imshow(self.window_name, self.img)
                cv2.waitKey(1)
            except KeyboardInterrupt:
                break
            
        self.killSequence()
            
    def show_frame(self):
        cv2.imshow(self.window_name, self.img)
            
    def poll_keystrokes(self):
        # Capture user input
        # while True:
            # Get the current key input
            # key = cv2.waitKey(1) & 0xff
            key = input("Enter input: ")
            
            # If ESC is pressed, then initiate the landing sequence and close all windows
            if key == 27: # ESC
                print(f'ESC')
                # break
            
                # ord(...) returns the unicode equivalent of a given string
                """
                elif key == ord('w'):
                    print(f'w')
                elif key == ord('s'):
                    print(f's')
                elif key == ord('a'):
                    print(f'a')
                elif key == ord('d'):
                    print(f'd')
                else:
                    print(f'key: {key}')
                """
            
            elif key == 'w':
                self.tello.move_forward(self.unit_dp)
            elif key == 's':
                self.tello.move_back(self.unit_dp)
            elif key == 'a':
                self.tello.move_left(self.unit_dp)
            elif key == 'd':
                self.tello.move_right(self.unit_dp)
            elif key == 'e':
                self.tello.rotate_clockwise(self.unit_dp)
            elif key == 'q':
                self.tello.rotate_counter_clockwise(self.unit_dp)
            elif key == 'r':
                self.tello.move_up(self.unit_dp)
            elif key == 'f':
                self.tello.move_down(self.unit_dp)
            elif key == 'l':
                self.tello.land()
                self.landed = True
            elif ((key == 't') and (self.landed == True)):
                self.tello.takeoff()
                self.landed = False

    def killSequence(self):
        main_loop = False
        self.tello.streamoff()
        cv2.destroyWindow(self.window_name)
        cv2.destroyAllWindows()

    """
    def landSequence(self, tello_obj, window_to_destroy):
        tello_obj.land()
        tello_obj.streamoff()
        cv2.destroyWindow(window_to_destroy)
        cv2.destroyAllWindows()
    """
    

if __name__ == "__main__":
    tello_video_stream = VideoStreamTello()
    main_loop = True
    
    while main_loop:
        try:
            # tello_video_stream.show_frame()
            tello_video_stream.poll_keystrokes()
        except KeyboardInterrupt:
            print(f'!!!Interrupted!!!')
            Stream = False
            main_loop = False
            # tello_video_stream.killSequence()
            tello_video_stream.video_stream_t.join()
                
