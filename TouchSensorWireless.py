import numpy as np
import json5
import json
from GenericReceiver import GenericReceiverClass
import socket
import select
import threading
from typing import List
from Sensor import Sensor
import aioconsole
import serial
import webbrowser
import re
import time
import asyncio
from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic
import serial_asyncio
from flaskApp.index import update_sensors, replay_sensors, start_server, notify_device_connected
import utils
import os
import platform
import qrcode
import utils  # assuming this contains your getUnixTimestamp function
import cv2
from pathlib import Path


class SerialProtocol(asyncio.Protocol):
    def __init__(self, receiver, sensorId):
        self.receiver = receiver
        self.sensorId = sensorId
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport
        print(f"Connection made for sensor {self.sensorId}")

    def data_received(self, data):
        asyncio.ensure_future(self.receiver.buffers[self.sensorId].put(data))

    def connection_lost(self, exc):
        print(f"Connection lost for sensor {self.sensorId}")



class WifiReceiver(GenericReceiverClass):
    def __init__(self,numNodes,sensors:List[Sensor], tcp_ip="10.0.0.67", tcp_port=7000, record=True, stopFlag=None):
        super().__init__(numNodes,sensors,record)
        self.TCP_IP = tcp_ip
        self.tcp_port = tcp_port
        self.connection_is_open = False
        self.connections = {}
        self.setup_TCP()
        self.stopFlag = stopFlag
    
    def setup_TCP(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.TCP_IP, self.tcp_port))
        sock.listen(len(self.sensors))  # Listen for N connections
        print("Waiting for connections")
        while len(self.connections) < len(self.sensors):
            connection, client_address = sock.accept()
            print("Connection found")
            sensorId = self.getSensorIdFromBuffer(connection)
            print(f"Connection found from {sensorId}")
            notify_device_connected(sensorId,True)
            self.connections[sensorId]=connection
        sock.settimeout(30)
        print("All connections found")

    def getSensorIdFromBuffer(self, connection):
        while True:
            ready_to_read, ready_to_write, in_error = select.select([connection], [], [], 30)
            if len(ready_to_read)>0:
                numBytes = 1+(int(self.numNodes)+1)*2+4
                inBuffer =   connection.recv(numBytes, socket.MSG_PEEK)
                if len(inBuffer) >= numBytes:
                    sendId, startIdx, sensorReadings, packet = self.unpackBytesPacket(inBuffer)
                return sendId

    def reconnect(self, sensorId):
        print(f"Reconnecting to sensor {sensorId}")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.TCP_IP, self.tcp_port))
        sock.listen(len(self.sensors))
        print("waiting for connections")
        found_Conn = False
        while not found_Conn:
            connection, client_address = sock.accept()
            connectionSensorId = self.getSensorIdFromBuffer(connection)
            if connectionSensorId == sensorId:
                print(f"Connection from Sensor {sensorId} found")
                self.connections[sensorId]=connection
                found_Conn = True
            else:
                print(f"Connection refused from {client_address}")
                connection.close()

    async def receiveData(self, sensorId):
        print("Receiving Data")
        while not self.stopFlag.is_set():
            connection = self.connections[sensorId]
            ready_to_read, ready_to_write, in_error = await asyncio.get_event_loop().run_in_executor(
                None, select.select, [connection], [], [], 30)
            if len(ready_to_read)>0:
                numBytes = 1+(int(self.numNodes)+1)*2+4
                inBuffer =   await asyncio.get_event_loop().run_in_executor(None, connection.recv, numBytes, socket.MSG_PEEK)
                if len(inBuffer) >= numBytes:
                    data = await asyncio.get_event_loop().run_in_executor(None, connection.recv, numBytes)
                    sendId, startIdx, sensorReadings, packet = self.unpackBytesPacket(data)
                    sensor = self.sensors[sendId]
                    if(sensor.intermittent):
                        sensor.processRowIntermittent(startIdx,sensorReadings,packet,record=self.record)
                    else:
                        if (startIdx==20000):
                            sensor.processRowReadNode(sensorReadings,packet,record=self.record)
                        else:
                            sensor.processRow(startIdx,sensorReadings,packet,record=self.record)
            else:
                print(f"Sensor {sensorId} is disconnected: Reconnecting...")
                await asyncio.get_event_loop().run_in_executor(None, connection.shutdown, 2)
                await asyncio.get_event_loop().run_in_executor(None, connection.close)
                self.reconnect(sensorId)

    def startReceiverThreads(self):
        tasks = []
        for sensorId in self.connections:
            task = self.receiveData(sensorId)
            tasks.append(task)
        return tasks


class BLEReceiver(GenericReceiverClass):
    def __init__(self, numNodes, sensors: List[Sensor],record=True, stopFlag=None):
        super().__init__(numNodes, sensors, record)
        self.deviceNames = [(sensor.deviceName, sensor.id) for sensor in sensors]
        self.clients={}
        self.stopFlag = stopFlag

    async def connect_to_device(self, lock, deviceTuple):
        def on_disconnect(client):
            print(f"Device {deviceTuple[0]} disconnected")
            # asyncio.create_task(self.connect_to_device(lock, deviceName))
        async with lock:
            device = await BleakScanner.find_device_by_name(deviceTuple[0],timeout=30)
            if device:
                print(f"Found device: {deviceTuple[0]}")
                client = BleakClient(device)
                client.set_disconnected_callback(on_disconnect)
                self.clients[deviceTuple[0]] = client
                await client.connect()

        def notification_handler(characteristic: BleakGATTCharacteristic, data: bytearray):
                sendId, startIdx, sensorReadings, packet = self.unpackBytesPacket(data)
                sensor = self.sensors[sendId]
                if(sensor.intermittent):
                    sensor.processRowIntermittent(startIdx,sensorReadings,packet,record=self.record)
                else:
                    if (startIdx==20000):
                        sensor.processRowReadNode(sensorReadings,packet,record=self.record)
                    else:
                        sensor.processRow(startIdx,sensorReadings,packet,record=self.record)

        try:
            await client.start_notify("1766324e-8b30-4d23-bff2-e5209c3d986f", notification_handler)
            notify_device_connected(deviceTuple[1], True)
            print(f"Connected to {deviceTuple[0]}")
        except Exception as e:
            notify_device_connected(deviceTuple[1], False)
            print(f"Failed to connect to {deviceTuple[0]}: {e}")
    
    import asyncio

    async def stopReceiver(self, max_retries=30, delay=2):
        for deviceName, client in self.clients.items():
            retries = 0
            while retries < max_retries:
                try:
                    if client.is_connected:
                        await client.stop_notify("1766324e-8b30-4d23-bff2-e5209c3d986f")
                        # await client.disconnect()
                        print(f"{deviceName}: Notifications stopped and client disconnected.")
                    else:
                        print(f"{deviceName}: Client already disconnected.")
                    break  # Exit retry loop if successful
                except Exception as e:
                    retries += 1
                    print(f"Error stopping client {deviceName} (attempt {retries}/{max_retries}): {e}")
                    await asyncio.sleep(delay)
            else:
                print(f"{deviceName}: Failed to disconnect after {max_retries} retries.")
        print("All clients cleaned up.")


    def startReceiverThreads(self):
        lock = asyncio.Lock()
        tasks = [self.connect_to_device(lock, name) for name in self.deviceNames]
        return tasks

# New class to manage a single serial connection
class SerialConnectionHandler:
    def __init__(self, sensor, receiver):
        self.sensor = sensor
        self.receiver = receiver
        self.partialData = b''
        self.buffer = asyncio.Queue()
        self.stopFlag = receiver.stopFlag
        self.delimiter = b'wr'

    async def read_lines(self):
        print(f"Reading lines for sensor {self.sensor}")
        while not self.stopFlag.is_set():
            try:
                data = await self.buffer.get()
                self.partialData += data

                while True:
                    index = self.partialData.find(self.delimiter)
                    if index == -1:
                        break

                    packet = self.partialData[:index]
                    await self.receiver.process_line(packet)
                    self.partialData = self.partialData[index + len(self.delimiter):]

            except Exception as e:
                print(f"[❌ read_lines error for sensor {self.sensor}] {e}")
                break

class SerialReceiver(GenericReceiverClass):
    def __init__(self, numNodes, sensors: List[Sensor], baudrate, stopFlag=None, record=True):
        super().__init__(numNodes, sensors, record)
        self.baudrate = baudrate
        self.stopFlag = stopFlag if stopFlag else asyncio.Event()
        self.connections = {}  # sensorId -> (transport, protocol)
        self.buffers = {}      # sensorId -> asyncio.Queue()
        self.delimiter = b'wr'
        self.transports = []   # List to keep track of transports for cleanup

    async def listen_for_stop(self):
        print("Listening for stop")
        while not self.stopFlag.is_set():
            input_str = await aioconsole.ainput("Press Enter to stop...\n")
            if input_str == "":
                self.stopFlag.set()
                await self.stopReceiver()
        print("Stop flag set, done listening for stop")
    async def stopReceiver(self):
        print("🔻 Stopping SerialReceiver...")
        for transport in self.transports:
            try:
                transport.close()
                transport.abort()  # ensure shutdown
            except Exception as e:
                print(f"⚠️ Error closing transport: {e}")

        for sensorId, buffer in self.buffers.items():
            try:
                buffer.put_nowait(b"")  # inject dummy data to unblock .get()
            except Exception as e:
                print(f"⚠️ Error unblocking buffer for sensor {sensorId}: {e}")

        print("✅ Transports closed and buffers unblocked.")



    async def connect_sensor(self, sensor, port):
        loop = asyncio.get_running_loop()
        # transport, protocol = await serial_asyncio.connection_for_serial(loop, lambda: SerialProtocol(self), ser)
        # print(f"making serial connection{self.sensors[sensor].id}")
        transport, protocol = await serial_asyncio.create_serial_connection(
            loop,
            lambda: SerialProtocol(self, sensor),
            port,
            baudrate=self.baudrate
        )
        self.buffers[sensor] = asyncio.Queue()
        notify_device_connected(self.sensors[sensor].id, True)
        self.transports.append(transport)
        while not self.stopFlag.is_set():
            await asyncio.sleep(0.1)  # Avoid blocking loop
        print(f"Disconnected to sensor {sensor} on port {port}")

    async def receiveData(self, sensorId):
        print(f"Receiving data from sensor {sensorId}")
        buffer = self.buffers[sensorId]
        partialData = b''
        while not self.stopFlag.is_set():
            try:
                data = await buffer.get()
                partialData += data

                while True:
                    index = partialData.find(self.delimiter)
                    if index == -1:
                        break
                    packet = partialData[:index]
                    await self.process_line(packet)
                    partialData = partialData[index + len(self.delimiter):]

            except Exception as e:
                print(f"[❌ receiveData error for sensor {sensorId}] {e}")
                break
        print(f"Stopping data reception for sensor {sensorId}")
        

    def startReceiverThreads(self):
        # Sync method to connect all sensors and return receive tasks
        # loop = asyncio.get_event_loop()
        tasks = []
        for sensor in self.sensors:
            tasks.append(self.connect_sensor(sensor, self.sensors[sensor].port))
            tasks.append(self.receiveData(sensor))
        tasks.append(self.listen_for_stop())  # Add stop listener task
        return tasks


def readConfigFile(file):
    with open(file, 'r') as file:
        data = json5.load(file)
    return data

def recordingsDirectory(recordings):
    if not os.path.exists(recordings):
        os.makedirs(recordings)

class MultiProtocolReceiver():
    def __init__(self, folder="recordings", configFilePath="./WiSensConfigClean.json"):
        self.recording_dir = os.path.join(os.getcwd(), folder)
        recordingsDirectory(self.recording_dir)
        print(f"✅ Created directory: {self.recording_dir}")
        self.config = readConfigFile(configFilePath)
        self.sensors = self.config['sensors']
        self.bleSensors = []
        self.wifiSensors = []
        self.serialSensors = []
        self.allSensors = []
        self.stopFlag = asyncio.Event()
        for sensorConfig in self.sensors:
            sensorKeys = list(sensorConfig.keys())
            intermittent = False
            p = 15

            if 'intermittent' in sensorKeys:
                intermittent = sensorConfig['intermittent']['enabled']
                p = sensorConfig['intermittent']['p']

            deviceName = "Esp1"
            userNumNodes = 120

            match sensorConfig['protocol']:
                case 'wifi':
                    userNumNodes = self.config['wifiOptions']['numNodes']
                case 'ble':
                    deviceName = sensorConfig['deviceName']
                    userNumNodes = self.config['bleOptions']['numNodes']
                case 'serial':
                    userNumNodes = self.config['serialOptions']['numNodes']

            print("Hello 2s")
            numGroundWires = sensorConfig['endCoord'][1] - sensorConfig['startCoord'][1] + 1
            numReadWires = sensorConfig['endCoord'][0] - sensorConfig['startCoord'][0] + 1
            numNodes = min(userNumNodes,min(120, numGroundWires*numReadWires))
            id = sensorConfig['id']
            newSensor = Sensor(numGroundWires,numReadWires,numNodes,sensorConfig['id'],deviceName=deviceName,intermittent=intermittent, p=p, port=sensorConfig["serialPort"])
            
            match sensorConfig['protocol']:
                case 'wifi':
                    self.wifiSensors.append(newSensor)
                case 'ble':
                    self.bleSensors.append(newSensor)
                case 'serial':
                    self.serialSensors.append(newSensor)
            self.allSensors.append(newSensor)

        self.receivers = []
        self.receiveTasks = []
        self.activeThreads = []

    def updateConfig(self, config):
        self.config = config
        self.sensors = self.config['sensors']
        self.bleSensors = []
        self.wifiSensors = []
        self.serialSensors = []
        self.allSensors = []
        self.stopFlag = asyncio.Event()
        for sensorConfig in self.sensors:
            sensorKeys = list(sensorConfig.keys())
            intermittent = False
            p = 15

            if 'intermittent' in sensorKeys:
                intermittent = sensorConfig['intermittent']['enabled']
                p = sensorConfig['intermittent']['p']

            deviceName = "Esp1"
            userNumNodes = 120

            match sensorConfig['protocol']:
                case 'wifi':
                    userNumNodes = int(self.config['wifiOptions']['numNodes'])
                case 'ble':
                    deviceName = sensorConfig['deviceName']
                    userNumNodes = int(self.config['bleOptions']['numNodes'])
                case 'serial':
                    userNumNodes = int(self.config['serialOptions']['numNodes'])

            numGroundWires = sensorConfig['endCoord'][1] - sensorConfig['startCoord'][1] + 1
            numReadWires = sensorConfig['endCoord'][0] - sensorConfig['startCoord'][0] + 1
            numNodes = min(userNumNodes,min(120, numGroundWires*numReadWires))
            newSensor = Sensor(numGroundWires,numReadWires,numNodes,sensorConfig['id'],deviceName=deviceName,intermittent=intermittent, p=p, port=sensorConfig["serialPort"])
            
            match sensorConfig['protocol']:
                case 'wifi':
                    self.wifiSensors.append(newSensor)
                case 'ble':
                    self.bleSensors.append(newSensor)
                case 'serial':
                    self.serialSensors.append(newSensor)
            self.allSensors.append(newSensor)

        self.receivers = []
        self.receiveTasks = []
        self.activeThreads = []

    def programSensor(self, sensor_id, config):
        data = config
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

        merged_data['resistance'] = max(0, min(127, (127 - int(merged_data['resistance']))))

        # Convert the merged data to a JSON string with proper quoting
        json_string = json.dumps(merged_data)
        print(json_string)

        if platform.system() =="Darwin":
            ser2 = serial.Serial(baudrate=data['serialOptions']['baudrate'], timeout=1)
            if "serialPort" in sensor:
                ser2.port=sensor["serialPort"]
                
            else:
                ser2.port=data['serialOptions']['port']

            print(ser2.port)
            print(ser2.baudrate)
            ser2.open()
        else:
            # Send the JSON string over the serial port
            ser = serial.Serial(baudrate=data['serialOptions']['baudrate'], timeout=1)
            if "serialPort" in sensor:
                ser.port=sensor["serialPort"]
            else:
                ser.port=data['serialOptions']['port']
            ser.open()
            ser.dtr = False  # Explicitly clear settings
            ser.rts = False
            ser.flush()
            ser.close()
            time.sleep(1)
            ser2 = serial.Serial(baudrate=data['serialOptions']['baudrate'], timeout=1)
            if "serialPort" in sensor:
                ser2.port=sensor["serialPort"]
                
            else:
                ser2.port=data['serialOptions']['port']

            print(ser2.port)
            print(ser2.baudrate)
            ser2.dtr = False  
            ser2.rts = False
            ser2.open()
            ser2.flush()

        ser2.write((json_string).encode('utf-8'))
        time.sleep(3)
        ser2.close()
        print("finished program")
        
    def calibrateSensor(self, sensor_id):
        data = self.config
        # Find the sensor with the given ID
        sensor = next((s for s in data['sensors'] if s['id'] == sensor_id), None)
        # Send the JSON string over the serial port
        ser = serial.Serial(baudrate=data['serialOptions']['baudrate'], timeout=1)
        if "serialPort" in sensor:
            ser.port=sensor["serialPort"]
        else:
            ser.port=data['serialOptions']['port']
        ser.dtr = False
        ser.rts = False
        try:
            ser.open()
            ser.write(b"calibrate\n")  # Send command (ensure correct encoding)

            # Wait for response
            while True:
                line = ser.readline().decode('utf-8').strip()  # Read line from serial
                match = re.search(r"Resistance\s*:\s*([\d.]+)", line)  # Match pattern
                if match:
                    resistance_value = float(match.group(1))  # Extract and convert value
                    return 127-resistance_value  # Return calibration value
        except Exception as e:
            print(f"Error: {e}")
        finally:
            ser.close()
    
    async def startReceiversAsync(self):
        await asyncio.gather(*self.receiveTasks, return_exceptions=True)
        print("[🟢] All tasks completed.")
        # await self.listen_for_stop()

    async def listen_for_stop(self):
        print("Listening for stop")
        while not self.stopFlag.is_set():
            input_str = await aioconsole.ainput("Press Enter to stop...\n")
            if input_str == "":
                print("Stop flag set")
                self.stopFlag.set()
        for receiver in self.receivers:
            if isinstance(receiver, BLEReceiver):
                await receiver.stopReceiver()


    async def listen_for_stop_web(self):
        print("Listening for stop")
        while not self.stopFlag.is_set():
            await asyncio.sleep(0.5)
        for receiver in self.receivers:
            if isinstance(receiver, BLEReceiver) or isinstance(receiver, SerialReceiver):
                await receiver.stopReceiver()


        self.receiveTasks = []  # Clear receiveTasks to allow fresh tasks on restart
        self.receivers = []  # Clear receivers list
        self.stopFlag.clear()  # Reset flag for future use
        print("All receivers stopped and cleaned up.")



    def initializeReceivers(self, record, web=False):
        if len(self.bleSensors)!=0:
            bleReceiver = BLEReceiver(int(self.config['bleOptions']['numNodes']),self.bleSensors, record)
            self.receivers.append(bleReceiver)
            self.receiveTasks += bleReceiver.startReceiverThreads()
        if len(self.wifiSensors)!=0:
            wifiReceiver = WifiReceiver(int(self.config['wifiOptions']['numNodes']),self.wifiSensors,self.config['wifiOptions']['tcp_ip'],int(self.config['wifiOptions']['port']), stopFlag=self.stopFlag, record=record)
            self.receivers.append(wifiReceiver)
            self.receiveTasks += wifiReceiver.startReceiverThreads()
        if len(self.serialSensors)!=0:
            serialReceiver = SerialReceiver(
                int(self.config['serialOptions']['numNodes']),
                self.serialSensors,
                self.config['serialOptions']['baudrate'],
                stopFlag=self.stopFlag,
                record=record
            )
            self.receivers.append(serialReceiver)
            self.receiveTasks += serialReceiver.startReceiverThreads()
        if web:
            self.receiveTasks.append(self.listen_for_stop_web())
        # else:
        #     self.receiveTasks.append(self.listen_for_stop())

    def startReceiverThread(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.startReceiversAsync())
        finally:
            print("Stopping receiver thread.")
            loop.close()


    def timeSyncQR(self):
        ts = utils.getUnixTimestamp()
        display_ts = time.time()

        payload = {
            "gen": ts,
            "displayed": round(display_ts, 3)
        }

        qr_string = f"{payload}"

        # Generate QR code as PIL image
        qr_img = qrcode.make(qr_string)

        # Convert PIL image to OpenCV format
        qr_img = qr_img.convert("RGB")
        qr_np = np.array(qr_img)
        qr_cv = cv2.cvtColor(qr_np, cv2.COLOR_RGB2BGR)

        # Display using OpenCV (faster than external viewer)
        window_name = "Time Sync QR"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 800)
        cv2.imshow(window_name, qr_cv)

        # Wait for any key (or keep visible for N seconds)
        print("Press any key to close QR display...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def record(self):
        self.initializeReceivers(True)
        captureThread = threading.Thread(target=self.startReceiverThread)
        captureThread.start()
         # Keep main thread alive so interpreter doesn’t shut down
        self.timeSyncQR()
        while captureThread.is_alive():
            time.sleep(0.1) 
            if self.stopFlag.is_set():
                print("Stop flag detected, stopping capture thread.")
                break
        self.stopFlag.set()  # Signal the stop flag to all receivers
        print("Waiting for capture thread to finish...")
        self.receiveTasks=[]
        captureThread.join()  # Wait for full thread cleanup
        print("Capture thread stopped cleanly.")

    def visualizeAndRecord(self):
        self.initializeReceivers(True)
        threads=[]
        captureThread = threading.Thread(target=self.startReceiverThread)
        captureThread.start()
        threads.append(captureThread)
        vizThread = threading.Thread(target=update_sensors, args=(self.allSensors,))
        vizThread.start()
        threads.append(vizThread)
        utils.start_nextjs()
        url = "http://localhost:3000"
        webbrowser.open_new_tab(url)
        start_server()
        for thread in threads:
            thread.join()


    def visualize(self):
        self.initializeReceivers(False)
        threads=[]
        captureThread = threading.Thread(target=self.startReceiverThread)
        captureThread.start()
        threads.append(captureThread)
        vizThread = threading.Thread(target=update_sensors, args=(self.allSensors,))
        vizThread.start()
        threads.append(vizThread)
        # utils.start_nextjs()
        # url = "http://localhost:3000"
        # webbrowser.open_new_tab(url)
        start_server()
        for thread in threads:
            thread.join()

    def visualize_web(self):
        self.initializeReceivers(True, True)
        captureThread = threading.Thread(target=self.startReceiverThread)
        captureThread.start()
        # self.activeThreads.append(captureThread)
        vizThread = threading.Thread(target=update_sensors, args=(self.allSensors,))
        vizThread.start()
        # self.activeThreads.append(vizThread)

    def replayData(self,fileDict, startTs=None,endTs=None, speed=1):
        pressureDict = {}
        totalFrames = None
        frameRate = None
        for sensorId in fileDict:
            pressure, fc, ts = utils.tactile_reading(fileDict[sensorId])
            startIdx = 0
            beginTs = ts[0]
            if startTs is not None:
                startIdx,beginTs = utils.find_closest_index(ts,startTs)
            endIdx, lastTs = len(ts), ts[-1]
            if endTs is not None:
                endIdx, lastTs = utils.find_closest_index(ts, endTs)
            if totalFrames is None or endIdx-startIdx<totalFrames:
                totalFrames = endIdx-startIdx
                frameRate = (totalFrames/(lastTs-beginTs)) * speed
            pressureDict[sensorId] = pressure[startIdx:endIdx,:,:]
        vizThread = threading.Thread(target=replay_sensors, args=(pressureDict,frameRate,totalFrames,))
        vizThread.start()
        utils.start_nextjs()
        url = "http://localhost:3000"
        # webbrowser.open_new_tab(url)
        start_server()

    def replayDataWeb(self,fileDict, startTs=None,endTs=None, speed=1):
        self.stopFlag.clear()
        pressureDict = {}
        totalFrames = None
        frameRate = None
        recordings_dir = Path("recordings")
        for sensorId in fileDict:
            print(sensorId,fileDict)
            pressure, fc, ts = utils.tactile_reading(recordings_dir / fileDict[sensorId])
            startIdx = 0
            beginTs = ts[0]
            if startTs is not None:
                startIdx,beginTs = utils.find_closest_index(ts,startTs)
            endIdx, lastTs = len(ts), ts[-1]
            if endTs is not None:
                endIdx, lastTs = utils.find_closest_index(ts, endTs)
            if totalFrames is None or endIdx-startIdx<totalFrames:
                totalFrames = endIdx-startIdx
                frameRate = (totalFrames/(lastTs-beginTs)) * speed
            pressureDict[sensorId] = pressure[startIdx:endIdx,:,:]
        vizThread = threading.Thread(target=replay_sensors, args=(pressureDict,frameRate,totalFrames,))
        vizThread.start()
        self.activeThreads.append(vizThread)

    # Sends all sensors (with real time pressure updates) as input to the custom method
    def runCustomMethod(self, method, record=False, viz=False):
        self.initializeReceivers(record)
        threads=[]
        captureThread = threading.Thread(target=self.startReceiverThread)
        captureThread.start()
        threads.append(captureThread)
        customThread = threading.Thread(target=method, args=(self.allSensors,))
        customThread.start()
        threads.append(customThread)
        if viz:
            vizThread = threading.Thread(target=update_sensors, args=(self.allSensors,))
            vizThread.start()
            threads.append(vizThread)
            utils.start_nextjs()
            url = "http://localhost:3000"
            webbrowser.open_new_tab(url)
            start_server()
        for thread in threads:
            thread.join()
        

    

if __name__ == "__main__":
    utils.programSensor(1)
    myReceiver = MultiProtocolReceiver()
    myReceiver.visualize()