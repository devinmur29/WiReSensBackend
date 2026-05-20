from flaskApp.index import start_server_web, notify_device_connected, update_sensors, start_server, replay_sensors
from wiresens_backend.TouchSensorWireless import MultiProtocolReceiver # This will work once the package is installed

# Create a dictionary of the callback functions that the backend library needs
callbacks = {
    "notify_device_connected": notify_device_connected,
    "update_sensors": update_sensors,
    "start_server": start_server,
    "replay_sensors": replay_sensors
}

# Instantiate the receiver and pass in the callbacks
myReceiver = MultiProtocolReceiver(callbacks=callbacks)

# Start the web server, passing it the receiver instance
start_server_web(myReceiver)