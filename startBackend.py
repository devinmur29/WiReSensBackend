from flaskApp.index import start_server_web
from TouchSensorWireless import MultiProtocolReceiver
myReceiver = MultiProtocolReceiver()
start_server_web(myReceiver)