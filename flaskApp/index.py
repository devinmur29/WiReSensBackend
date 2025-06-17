import eventlet
eventlet.monkey_patch()
from flask import Flask
import json
from flask_socketio import SocketIO
from flask_cors import CORS
import time
import numpy as np
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) 
# socketio = SocketIO(app, cors_allowed_origins="*")

socketio = SocketIO(app, async_mode="eventlet", cors_allowed_origins="*")  # Explicitly set async mode

def start_server_web(receiver):
    global myReceiver
    myReceiver = receiver
    socketio.run(app, host="0.0.0.0", port=5328, debug=True, use_reloader=False)


myReceiver = None  # This will be assigned later

def replay_sensors( pressureDict, frameRate, numFrames):
    with app.app_context():
        for i in range(numFrames):
            if myReceiver.stopFlag.is_set():
                return
            sensors={}
            for j in pressureDict:
                pressure=pressureDict[j][i]
                sensors[j]= (pressure).tolist()
            jsonSensors = json.dumps(sensors)
            socketio.emit('sensor_data', jsonSensors)
            time.sleep(1/frameRate)

def update_sensors(allSensors):
    with app.app_context():
        sensors={}
        if myReceiver:
            while not myReceiver.stopFlag.is_set():
                for i in range(len(allSensors)):
                    pressure=allSensors[i].pressure.reshape(allSensors[i].selWires,allSensors[i].readWires)
                    sensors[allSensors[i].id]= (pressure).tolist()
                jsonSensors = json.dumps(sensors)
                socketio.emit('sensor_data', jsonSensors)
                time.sleep(1/30)  # 50 FPS
        else:
            while True:
                for i in range(len(allSensors)):
                    pressure=allSensors[i].pressure.reshape(allSensors[i].selWires,allSensors[i].readWires)
                    sensors[allSensors[i].id]= (pressure).tolist()
                    
                jsonSensors = json.dumps(sensors)
                socketio.emit('sensor_data', jsonSensors)
                time.sleep(1/100)  # 50 FPS
        
def notify_device_connected(id, connected):
    with app.app_context():
        print("emitting detected")
        socketio.emit('connection_status', {'id':id, 'connected':connected})


@socketio.on('startViz')
def handle_start_viz(data):
    print("Received startViz message")
    if myReceiver:
        myReceiver.updateConfig(data)
        socketio.start_background_task(myReceiver.visualize_web)

@socketio.on('stopViz')
def handle_stop_viz():
    if myReceiver:
        myReceiver.stopFlag.set()
        # for thread in myReceiver.activeThreads:
        #     thread.join()  # Wait for threads to exit safely
        # myReceiver.activeThreads = []  # Clear the list after stopping
        # print("All visualization threads stopped")

@socketio.on('program')
def handle_program(data, id):
    print("Received program message")
    if myReceiver:
        myReceiver.programSensor(id, data)
        print("finished function")

@socketio.on('calibrate')
def handle_calibrate(id):
    if myReceiver:
        resValue = myReceiver.calibrateSensor(id)
        socketio.emit('calibration_done', {'id': id, 'value': resValue})

@socketio.on('replay')
def handle_replay(settings):
    fileDict=settings['sensorFiles']
    print(fileDict)
    if settings['startTimestamp'] != '':
        startTs = float(settings['startTimestamp'])
    else:
        startTs = None
    if settings['endTimestamp'] != '':
        endTs = float(settings['endTimestamp'])
    else:
        endTs = None
    playback = float(settings['playbackRate'])
    if myReceiver:
        socketio.start_background_task(myReceiver.replayDataWeb,fileDict,startTs,endTs,playback)

def start_server():
    socketio.run(app, host="0.0.0.0", port=5328, debug=True, use_reloader=False)

def start_server_web(receiver):
    global myReceiver
    myReceiver = receiver  # Assign receiver instance
    socketio.run(app, host="0.0.0.0", port=5328, debug=True, use_reloader=False)


@app.route('/api/python')
def index():
    return "WebSocket server is running..."