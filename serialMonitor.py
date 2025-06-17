import serial
import json
import time
import os # Import the os module for path operations

# Define the name of your JSON configuration file
json_file_name = "oneGloveserial.json"

# Construct the full path to the JSON file
# This assumes the JSON file is in the same directory as the script.
# If it's elsewhere, provide the absolute path.
script_dir = os.path.dirname(__file__)
json_file_path = os.path.join(script_dir, json_file_name)

# 1. Read the JSON configuration from the file
try:
    with open(json_file_path, 'r') as f:
        json_config_string = f.read()
    print(f"Successfully read configuration from {json_file_path}")
except FileNotFoundError:
    print(f"Error: The file '{json_file_path}' was not found.")
    print("Please ensure 'oneGloveserial.json' is in the same directory as the script, or provide the full path.")
    exit()
except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    exit()

# 2. Parse the JSON string into a Python dictionary
try:
    config_data = json.loads(json_config_string)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON from '{json_file_name}': {e}")
    exit()

# 3. Extract serial port and baud rate from the parsed JSON
try:
    serial_port = config_data["sensors"][0]["serialPort"]
    baud_rate = config_data["serialOptions"]["baudrate"]
except KeyError as e:
    print(f"Could not find required serial port or baud rate in JSON from '{json_file_name}': {e}")
    exit()

print(f"Attempting to open serial port: {serial_port} with baud rate: {baud_rate}")

# 4. Initialize and configure the serial port
try:
    ser = serial.Serial(serial_port, baud_rate, timeout=1)
    print("Serial port opened successfully.")
except serial.SerialException as e:
    print(f"Error opening serial port {serial_port}: {e}")
    print("Please ensure the port is correct and not in use by another application.")
    exit()

# 5. Prepare the data to be sent
# The entire JSON string will be sent.
# You must encode the string to bytes before sending it over serial.
# Adding a newline character '\n' can be useful for the receiving device
# to know when a complete message has been sent.
data_to_send = json_config_string + '\n'
encoded_data = data_to_send.encode('utf-8')

# 6. Send the data over the serial port
try:
    ser.write(encoded_data)
    print(f"Sent {len(encoded_data)} bytes to serial port.")
    print("Data sent:")
    print(encoded_data.decode('utf-8').strip())

    # Optional: Read response from serial port (if your device sends one)
    # You might need to adjust the delay and buffer size based on your device.
    time.sleep(0.1) # Give the device some time to respond
    if ser.in_waiting > 0:
        response = ser.read_all().decode('utf-8').strip()
        print(f"\nReceived response from serial port:\n{response}")
    else:
        print("\nNo response received from serial port.")

except serial.SerialException as e:
    print(f"Error writing to serial port: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    # 7. Close the serial port
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Serial port closed.")

