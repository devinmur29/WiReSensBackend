import serial

# Change this to your actual port, e.g., 'COM5' on Windows or '/dev/ttyUSB0' on Linux/macOS
PORT = 'COM6'
BAUD_RATE = 250000  # Must match ESP32's Serial.begin()

try:
    with serial.Serial(PORT, BAUD_RATE, timeout=1) as ser:
        print(f"Listening on {PORT} at {BAUD_RATE} baud...")
        while True:
            data = ser.read(ser.in_waiting or 1)
            if data:
                print("Raw bytes:", data)
                try:
                    print("Decoded:", data.decode('utf-8').strip())
                except UnicodeDecodeError:
                    print("Could not decode as UTF-8")

except serial.SerialException as e:
    print(f"Serial error: {e}")
except KeyboardInterrupt:
    print("Exiting...")
