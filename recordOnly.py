from TouchSensorWireless import MultiProtocolReceiver
from utils import discoverPorts
# discoverPorts()
myReceiver = MultiProtocolReceiver("./configs/twoGlovesSerial.json")
myReceiver.record()