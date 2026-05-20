import h5py
import numpy as np
import json5
import json
import serial
import datetime
from datetime import datetime
import subprocess
import struct
import time
import serial.tools.list_ports


def tactile_reading(path):
    f = h5py.File(path, 'r')
    fc = f['frame_count'][0]
    ts = np.array(f['ts'][:fc])
    pressure = np.array(f['pressure'][:fc]).astype(np.float32)

    return pressure, fc, ts

def find_closest_index(array, value):
    index = (np.abs(array - value)).argmin()
    return index, array[index]

def getUnixTimestamp():
    return np.datetime64(datetime.now()).astype(np.int64) / 1e6  # unix TS in secs and microsecs

def start_nextjs():
    try:
        subprocess.Popen(['npm', 'run', 'next-dev'], cwd='./ui/nextjs-flask', shell=True)
    except Exception as e:
        print(f"Failed to start Next.js: {e}")

def programSensor(sensor_id, config="./WiSensConfigClean.json"):
    # Read the JSON file
    with open(config, 'r') as file:
        data = json5.load(file)
    
    # Find the sensor with the given ID
    sensor = next((s for s in data['sensors'] if s['id'] == sensor_id), None)
    if not sensor:
        raise ValueError(f"Sensor with ID {sensor_id} not found.")

    # Determine the protocol
    protocol = sensor.get('protocol')
    protocol_key = f"{protocol}Options"
    if protocol_key not in data:
        raise ValueError(f"Protocol '{protocol}' not supported.")

    # Get the protocol options
    protocol_options = data.get(protocol_key, {})

    #get readout options
    readout_options = data.get("readoutOptions",{})

    # Merge sensor data with the protocol options
    merged_data = {**sensor, **protocol_options, **readout_options}

    # Convert the merged data to a JSON string with proper quoting
    json_string = json.dumps(merged_data)
    print(json_string)
    # Send the JSON string over the serial port
    ser = serial.Serial(baudrate=data['serialOptions']['baudrate'], timeout=1)
    if "serialPort" in sensor:
        ser.port=sensor["serialPort"]
    else:
        ser.port=data['serialOptions']['port']
    ser.dtr = False
    ser.rts = False
    ser.open()
    ser.write(json_string.encode('utf-8'))
    ser.close()


def unpackBytesPacket(byteString):
        format_string = '=b' + 'H' * (1+120) + 'I'  # 'b' for int8_t, 'H' for uint16_t
        tupData = struct.unpack(format_string,byteString)
        sendId = tupData[0]
        startIdx = tupData[1]
        sensorReadings = tupData[2:-1]
        packetNumber = tupData[-1]
        return sendId, startIdx,sensorReadings, packetNumber

def unpackBytesPacket(byteString):
    format_string = '=b' + 'H' * (1 + 120) + 'I'  # 'b' for int8_t, 'H' for uint16_t, 'I' for uint32_t
    tupData = struct.unpack(format_string, byteString)
    sendId = tupData[0]
    startIdx = tupData[1]
    sensorReadings = tupData[2:-1]
    packetNumber = tupData[-1]
    return sendId, startIdx, sensorReadings, packetNumber


def get_send_id(line):
    expected_len = 1 + (1 + 120) * 2 + 4  # 1 byte for sendId, 121 uint16_t (2 bytes each), 4 bytes for uint32_t
    if len(line) == expected_len:
        sendId, startIdx, readings, packetID = unpackBytesPacket(line)
        return sendId
    return None


def read_line(port, timeout=2.0):
    port.timeout = 0.1
    start_time = time.time()
    buffer = b""
    
    while time.time() - start_time < timeout:
        try:
            data = port.read(256)
            if data:
                buffer += data
                # Split by "wr" marker, which is ASCII for b"wr"
                parts = buffer.split(b"wr")
                for part in parts:
                    sendId = get_send_id(part)
                    if sendId is not None:
                        return sendId
                # Retain only the last part in buffer for next iteration
                buffer = parts[-1] if parts else b""
        except Exception as e:
            print(f"Error reading from port: {e}")
    return None


def discoverPorts(json_path="twoGlovesSerial.json"):
    with open(json_path, "r") as f:
        config = json.load(f)

    sensor_config = config["sensors"]
    discovered_ids = {}

    ports = serial.tools.list_ports.comports()
    for portInfo in ports:
        port_name = portInfo.device
        try:
            ser = serial.Serial(port=port_name, baudrate=250000, timeout=0.5)
            time.sleep(1.5)  # Give the device time to boot/send
            sendId = read_line(ser)
            ser.close()

            if sendId is not None:
                print(f"Found device with sendId={sendId} on port {port_name}")
                discovered_ids[sendId] = port_name
        except Exception as e:
            print(f"Could not open port {port_name}: {e}")

    # Update config
    for sensor in sensor_config:
        sid = sensor.get("id")
        if sid in discovered_ids:
            sensor["serialPort"] = discovered_ids[sid]
            print(f"Updated sensor ID {sid} to port {discovered_ids[sid]}")

    # Write back the config
    with open(json_path, "w") as f:
        json.dump(config, f, indent=2)
        print(f"Updated {json_path}")