import argparse
from utils import discoverPorts

# Parse the --small flag
parser = argparse.ArgumentParser(description="Run discoverPorts with config file based on glove size.")
parser.add_argument('--small', action='store_true', help='Use the small glove config if set.')
args = parser.parse_args()

# Choose the config path based on the flag
config_path = "./configs/twoGlovesSmall.json" if args.small else "./configs/twoGlovesLarge.json"

# Run discoverPorts with error handling
try:
    discoverPorts(json_path=config_path)
    print("✅ Success: Config file used and ports discovered.")
except Exception as e:
    print(f"❌ Error: {e}")
