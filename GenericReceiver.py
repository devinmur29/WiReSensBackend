import time
import struct
from Sensor import Sensor
from typing import List

class GenericReceiverClass():
    def __init__(self, numNodes, sensors: List[Sensor], record):
        self.frameRate = None
        self.startTime = time.time()
        self.numNodes = numNodes
        self.sensors = {sensor.id: sensor for sensor in sensors}
        self.record = record
        self.stopFlag = None # This will be set by the implementing class

    def startReceiver(self):
        raise NotImplementedError("Receivers must implement a startReceiver method")

    async def stopReceiver(self):
        raise NotImplementedError("Receivers must implement a stop receiver method")

    def unpackBytesPacket(self, byteString):
        format_string = '<B' + 'H' * (1 + self.numNodes) + 'I'
        tupData = struct.unpack(format_string, byteString)
        sendId = tupData[0]
        startIdx = tupData[1]
        sensorReadings = tupData[2:-1]
        packetNumber = tupData[-1]
        return sendId, startIdx, sensorReadings, packetNumber

    async def process_line(self, line):
        # This function remains a class method of the receiver, 
        # as it still uses shared state like `self.sensors`.
        if len(line) == 1 + (1 + self.numNodes) * 2 + 4:
            sendId, startIdx, readings, packetID = self.unpackBytesPacket(line)
            sensor = self.sensors[sendId]
            await sensor.processRowAsync(startIdx, readings, packetID)


    
    