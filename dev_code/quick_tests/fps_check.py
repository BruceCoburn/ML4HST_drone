from djitellopy import Tello
import time
import cv2

# Initialize Tello
tello = Tello()
tello.connect()

# Enable video streaming
tello.streamon()

# Start the timer
start_time = time.time()

# Counter for frames received
frame_count = 0

# Main loop to receive frames and calculate FPS
while True:
    # Read a frame from the video stream
    frame = tello.get_frame_read().frame

    # Increase frame count
    frame_count += 1

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Calculate FPS every second
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        print(f"FPS: {fps:.2f}")

        # Reset frame count and timer
        frame_count = 0
        start_time = time.time()

    # Check for 'q' keypress to exit the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
tello.streamoff()
tello.end()
