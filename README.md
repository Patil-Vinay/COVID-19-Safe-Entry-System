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
| `mask_model.tflite` | Trained mask model used on the Pi |
| `hardware_test.py` | Tests buzzer, servo, IR sensors, ultrasonic and temperature sensor one by one |
| `mlx90614_sensor_testing.py` | Prints readings from the MLX90614 to test the wiring |
| `get_face_detector.py` | Downloads the face detector weights into `face_detector/` |
| `split_dataset.py` | Splits the images into train, val and test folders, keeping copies of the same photo together |
| `train_mask_model.py` | Trains the mask model on a PC and saves `.h5` and `.tflite` |
| `evaluate_model.py` | Tests the `.tflite` model on the test images |
| `results/` | Training plot and confusion matrix |

## Mask model

### Dataset

We used the face mask dataset by Prajna Bhandary ([prajnasb/observations](https://github.com/prajnasb/observations), folder `experiements/data`).
It has 690 images with a mask and 686 without.

This dataset has many copies of the same photo. 97 files are exact copies, the `augmented_image_*` files are mirrored or edited
versions of other photos and the "with mask" images are the "without mask" photos with a mask pasted on.
If copies of a photo end up in both train and test, the model has already seen the test photos and the accuracy looks better than it is.

`split_dataset.py` handles this. It removes the exact copies, then compares every image using a 256 bit image hash
(also of the mirrored image and of the top half only, which a mask does not cover) and puts similar images in one group.
A whole group always goes into the same folder. The limit of 44 different bits was picked by looking at image pairs:
up to 44 they were all the same photo, above it different people with the same background started to show up.

```
python split_dataset.py --source observations/experiements/data
```

This gives 1279 images in 679 groups:

| Folder | with_mask | without_mask |
|---|---|---|
| train | 453 | 442 |
| val | 62 | 66 |
| test | 129 | 127 |

### Training

Done on a PC, not on the Pi. MobileNetV2 trained on ImageNet is used as the base and only the new top layers are trained
(Adam, learning rate 0.001). Training stops when the val loss has not gone down by at least 0.001 for 5 epochs
and the weights of the best epoch are kept. In our run it stopped after 26 epochs and kept epoch 21.

```
pip install -r requirements-train.txt
python train_mask_model.py
```

This saves `mask_model.h5` (not kept in the repo), `mask_model.tflite` and `results/training_plot.png`.
The TFLite model is saved with float16 weights. The default 8 bit version was smaller but took over a second per image.

![Training plot](results/training_plot.png)

### Results

`evaluate_model.py` runs the `.tflite` model on the 256 test images. None of them, or any copy of them,
was used for training or for deciding when to stop.

```
python evaluate_model.py
```

| | Precision | Recall |
|---|---|---|
| with_mask (129 images) | 1.000 | 0.992 |
| without_mask (127 images) | 0.992 | 1.000 |

Test accuracy: **99.61%** (255 of 256).
No person without a mask was classified as wearing one. 1 person with a mask was classified as no mask and would be asked to try again.

![Confusion matrix](results/confusion_matrix.png)

All masks in this dataset are the same pasted blue mask, so the test is easy for the model.
Real masks of other colours and shapes can give lower accuracy. Adding real photos to the dataset and training again will help.

## Running on the Raspberry Pi

Written for Raspberry Pi OS Bullseye (Python 3.9) with the legacy camera stack.

1. Enable I2C and the legacy camera in `sudo raspi-config`.
2. Install the libraries. numpy and OpenCV from piwheels need the ATLAS library on the 32 bit OS:
   ```
   sudo apt install libatlas-base-dev
   pip3 install -r requirements.txt
   ```
   If OpenCV still says a `.so` file is missing, the opencv-python page on piwheels.org lists the other system packages it needs.
3. Download the face detector weights:
   ```
   python3 get_face_detector.py
   ```
4. Test the parts:
   ```
   python3 hardware_test.py
   python3 mask_detector.py
   ```
5. Compare the MLX90614 reading with a normal thermometer and set `TEMP_OFFSET` in `config.py`.
   The sensor reads skin temperature, which is lower than body temperature.
6. Start the system:
   ```
   python3 sentinel.py
   ```

The model, face detector and log paths are taken from the project folder, so `sentinel.py` can also be started from
another folder, for example at boot. Press Ctrl+C to stop. The door closes and the GPIO pins are cleaned up.

## Prototype

![Initial Circuit](Initial%20Circuit.png)
