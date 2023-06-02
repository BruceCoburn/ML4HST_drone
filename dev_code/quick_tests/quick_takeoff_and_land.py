from djitellopy import Tello

tello = Tello()

tello.connect()
tello.takeoff()

print(f'sdk_version: {tello.query_sdk_version()}')

"""
tello.move_left(100)
tello.rotate_counter_clockwise(90)
tello.move_forward(100)    
"""

tello.land()
