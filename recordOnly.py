import argparse
from TouchSensorWireless import MultiProtocolReceiver

# Argument parsing
parser = argparse.ArgumentParser(description="Run MultiProtocolReceiver with specified folder and config.")
parser.add_argument('--foldername', required=True, help='Name of the folder to record into.')
parser.add_argument('--small', action='store_true', help='Use small config if set.')
parser.add_argument('--right', action='store_true', help='Use single right config if set.')
parser.add_argument('--left', action='store_true', help='Use single left config if set.')
args = parser.parse_args()

# Select config path
if args.small:
    if args.right:
        config_path = "./configs/oneGloveSerialReceiverRightSmall.json"
    elif args.left:
        config_path = "./configs/oneGloveSerialReceiverLeftSmall.json"
    else:
        config_path = "./configs/twoGlovesSmall.json"
else:
    if args.right:
        config_path = "./configs/oneGloveSerialReceiverRightLarge.json"
    elif args.left:
        config_path = "./configs/oneGloveSerialReceiverLeftLarge.json"
    else:
        config_path = "./configs/twoGlovesLarge.json"

# Initialize and run
myReceiver = MultiProtocolReceiver(args.foldername, config_path)
myReceiver.record()
