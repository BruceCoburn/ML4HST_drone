from djitellopy import Tello

tello = Tello()
tello.connect()

print(f'Battery Life: {tello.query_battery()}%')

# Clean up
tello.end()
