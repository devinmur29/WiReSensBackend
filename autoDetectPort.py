import argparse
from utils import discoverPorts

# Parse the --small flag
parser = argparse.ArgumentParser(description="Run discoverPorts with config file based on glove size.")
parser.add_argument('--small', action='store_true', help='Use the small glove config if set.')
parser.add_argument('--right', action='store_true', help='Use single right config if set.')
parser.add_argument('--left', action='store_true', help='Use single left config if set.')
args = parser.parse_args()

# Choose the config path based on the flag
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

# Run discoverPorts with error handling
try:
    discoverPorts(json_path=config_path)
    print("✅ Success: Config file used and ports discovered.")
except Exception as e:
    print(f"❌ Error: {e}")
