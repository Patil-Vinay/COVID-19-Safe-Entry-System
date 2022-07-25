# COVID-19-Safe-Entry-System

Because of the mask and body temperature rules at public places, staff have to check every person by hand before letting them in.
Doing this manually is slow and mistakes happen easily.

This project is a smart entry system that checks the face mask and body temperature of a person, makes them sanitize their hands
and only then opens the door. It also keeps count of how many people are inside the building.

Our project is named "THE SENTINEL" - which means a guard who stands at the entrance and keeps watch.

![Block Diagram](Block%20Diagram.png)

## How it works

1. An IR sensor at the entrance detects a person standing at the check point.
2. The Pi Camera takes frames. OpenCV's SSD face detector finds the face and a MobileNetV2 model (TFLite) decides mask or no mask.
3. The MLX90614 reads the temperature. Anything above 37.5 C is refused.
4. The person puts a hand under the sanitizer. The HC-SR04 checks for a hand closer than 10 cm.
5. If everything passes the servo opens the door. An IR sensor in the doorway sees the person walk through and the door closes again.
6. An IR sensor on the inside opens the door for people going out.

The system refuses entry once `MAX_PEOPLE` are inside. Every check is saved in `logs/entries.csv`.

Buzzer:
- one beep for every check that passes
- two short beeps when the door opens
- three long beeps when entry is refused

## Hardware

- Raspberry Pi 4 Model B
- Pi Camera
- MLX90614 IR temperature sensor
- 3 IR obstacle sensors
- HC-SR04 ultrasonic sensor
- Servo motor for the door
- Buzzer

| Part | Pi pin (BCM) |
|---|---|
| IR at check point | GPIO 17 |
| IR in doorway | GPIO 27 |
| IR on the inside (exit) | GPIO 22 |
| HC-SR04 TRIG | GPIO 23 |
| HC-SR04 ECHO | GPIO 24 (through a voltage divider) |
| Servo | GPIO 18 |
| Buzzer | GPIO 25 |
| MLX90614 SDA / SCL | GPIO 2 / GPIO 3 |

Pins and limits can be changed in `config.py`.

![Circuit Diagram](Circuit%20Diagram.png)

## Files

| File | What it does |
|---|---|
| `sentinel.py` | Main program that runs on the Pi |
| `config.py` | Pins, temperature limit, timings |
| `mask_detector.py` | Face detection + mask model. Run it alone to test with a camera |
| `train_mask_model.py` | Trains the mask model on a PC and saves `.h5` and `.tflite` |
| `mlx90614_sensor_testing.py` | Prints readings from the MLX90614 to test the wiring |
| `face_detector/deploy.prototxt` | Config for OpenCV's face detector |

## Training the mask model

Done on a PC, not on the Pi. Put the images in two folders:

```
dataset/with_mask/
dataset/without_mask/
```

```
pip install -r requirements-train.txt
python train_mask_model.py --dataset dataset --epochs 10
```

This saves `mask_model.h5`, `mask_model.tflite` and `training_plot.png`. Only the `.tflite` file is needed on the Pi.
The dataset and model files are not in the repo because of their size.

## Running on the Raspberry Pi

1. Enable I2C and the legacy camera in `sudo raspi-config`.
2. Download the face detector weights into `face_detector/`:
   [res10_300x300_ssd_iter_140000.caffemodel](https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)
3. Copy `mask_model.tflite` into the project folder.
4. Install the libraries:
   ```
   pip3 install -r requirements.txt
   ```
5. Test the parts:
   ```
   python3 mlx90614_sensor_testing.py
   python3 mask_detector.py
   ```
6. Start the system:
   ```
   python3 sentinel.py
   ```

Press Ctrl+C to stop. The door closes and the GPIO pins are cleaned up.

## Prototype

![Initial Circuit](Initial%20Circuit.png)
