import argparse
from TouchSensorWireless import MultiProtocolReceiver

# Argument parsing
parser = argparse.ArgumentParser(description="Run MultiProtocolReceiver with specified folder and config.")
parser.add_argument('--foldername', required=True, help='Name of the folder to record into.')
parser.add_argument('--small', action='store_true', help='Use small config if set.')
args = parser.parse_args()

# Select config path
config_path = "./configs/twoGlovesSmall.json" if args.small else "./configs/twoGlovesLarge.json"

# Initialize and run
myReceiver = MultiProtocolReceiver(args.foldername, config_path)
myReceiver.record()
