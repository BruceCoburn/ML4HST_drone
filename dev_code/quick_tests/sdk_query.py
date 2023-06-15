from djitellopy import Tello

tello = Tello()

tello.connect()
print(f"sdk_version: {tello.query_sdk_version()}")
