# **WiReSens Recording Setup**

This guide walks you through the full process of recording tactile sensor data using the **WiReSens Toolkit**—from initial setup and device connection to data capture and visualization.

---

## **Software Setup**

1. **Install Python (version 3.10 or later)**
   Download the latest version of Python from:
   👉 [https://www.python.org/downloads/](https://www.python.org/downloads/)

   > ⚠️ Make sure to check the box that says "Add Python to PATH" during installation.

2. **Install required Python packages**
   In your terminal or command prompt, navigate to the project root and run:

   ```
   pip install -r requirements.txt
   ```

3. **Create a folder for recordings**
   Inside the project’s root directory, create a folder to store your recordings:

   ```
   mkdir recordings
   ```

---

## **Hardware Setup**

1. **Plug in your microcontrollers**

   Depending on how your system is configured to transmit data, follow the appropriate instructions:

   * **If the transmitter sends data via USB:**
     → Connect the **transmitter** directly to your computer via USB.

   * **If the transmitter uses ESP-NOW:**
     → Connect the **receiver** to your computer via USB.
     → The transmitter can be powered by USB or a LiPo battery.

   * **If the transmitter uses WiFi or Bluetooth:**
     → The transmitter can operate untethered (powered via battery).

   * **If the transmitter has just been flashed or reconfigured:**
     → You **must** connect it to your computer over USB at least once to complete configuration.

2. **Identify your device’s serial port**

   WiReSens uses JSON configuration files to reference sensor devices by their serial port. To auto-detect and assign the correct port, run:

   ```
   python autoDetectPort.py
   ```

   This script scans available serial ports, identifies known microcontrollers, and updates the relevant JSON config files.

   > 📝 If the script fails to detect your device, follow the manual method below to find your serial port.

   * **Linux:**

     ```
     dmesg | grep tty
     ```

     Look for entries like `/dev/ttyUSB0` or `/dev/ttyACM0`.

   * **macOS:**

     ```
     ls /dev/tty.*
     ```

     You’ll see ports like `/dev/tty.usbserial-110` or `/dev/tty.usbmodem14101`.

   * **Windows:**

     1. Open **Device Manager**
     2. Expand **Ports (COM & LPT)**
     3. Look for something like `COM5`

3. **Update your configuration file**

   Open the relevant JSON config file from the `configs/` folder.
   Locate the `"serialPort"` field under the `"sensors"` section and update it with your device’s port. If the autoDetectPort method works it will have updated this config file automatically. 

   ```json
   "serialPort": "/dev/ttyUSB0"
   ```

   > 💡 On Windows, this would be something like `"serialPort": "COM5"`

---

## **To Record Using the Web Visualizer**

1. **Start the backend server**

   In the root directory, run:

   ```
   python startBackend.py
   ```

2. **Open the web interface**
   In your browser, go to:
   👉 [https://wi-re-sens-web.vercel.app/](https://wi-re-sens-web.vercel.app/)

3. **Load your sensor configuration**

   * Click **Load Config**
   * Select the appropriate configuration file (e.g., from `configs/`)

4. **Program the device (if necessary)**

   * If this is your first time using the device, or if you've re-flashed it, you may need to configure it:

     * Go to **Device Panel**

     * Click the ✎ (Edit icon)

     * Click **Program Device**

     * A message saying **"Programmed!"** will appear on success

   > ✅ If you received a plug-and-play kit, this step is usually already done unless you're reconfiguring.

5. **Start recording**

   * Click **Record**
   * Visualization and recording will begin simultaneously
   * Data will be saved in the `recordings/` folder in HDF5 format

---

## **To Record Without the Visualizer**

You can record data headlessly (e.g., for batch experiments or logging) using the following method:

1. Open `recordOnly.py`

2. Update line 2 to point to your config file:

   ```python
   myReceiver = MultiProtocolReceiver("./configs/oneGloveSerialReceiverLeftSmall.json")
   ```

3. Run the script:

   ```
   python recordOnly.py
   ```

---

## **To Replay and Visualize Recordings**

You can replay a tactile recording using the spatial visualization tool.

Run the `createViz.py` script with your desired inputs. For example:

```python
create_video(
    left_h5="./recordings/recentLeft.hdf5",
    mapping_json="point_weight_mappings_large.json",
    svg_file="voronoi_regions_large.svg",
    output_mp4="glove_viz.mp4",
    use_normalized=False
)
```

> 🎥 This will render a video (`glove_viz.mp4`) showing your tactile data projected over a spatial layout.

---

## **Data Processing and Accessing Raw Pressure Values**

All tactile data is stored in `.hdf5` format.

To extract and analyze it, use the `tactile_reading` function in `utils.py`:

```python
frames, num_frames, timestamps = tactile_reading("recordings/myRecording.hdf5")
```

* `frames` → A NumPy array of shape `(frame_count, rows, cols)` with pressure values
* `num_frames` → Total number of frames
* `timestamps` → Unix timestamps (in seconds) for each frame
