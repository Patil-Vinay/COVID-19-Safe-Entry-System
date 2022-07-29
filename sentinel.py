# THE SENTINEL - COVID-19 safe entry system
# Runs on the Raspberry Pi: mask check, temperature check, hand sanitizer,
# then opens the door and keeps count of how many people are inside.
import csv
import os
import time
from datetime import datetime

import cv2
import RPi.GPIO as GPIO
from smbus2 import SMBus
from mlx90614 import MLX90614

import config
from mask_detector import MaskDetector


def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in (config.IR_CHECK_PIN, config.IR_DOOR_PIN, config.IR_EXIT_PIN, config.ECHO_PIN):
        GPIO.setup(pin, GPIO.IN)
    GPIO.setup(config.TRIG_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(config.BUZZER_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(config.SERVO_PIN, GPIO.OUT)
    servo = GPIO.PWM(config.SERVO_PIN, 50)
    servo.start(0)
    return servo


def ir_blocked(pin):
    # the IR modules give LOW when something is in front of them
    return GPIO.input(pin) == GPIO.LOW


def beep(times=1, length=0.2):
    for _ in range(times):
        GPIO.output(config.BUZZER_PIN, GPIO.HIGH)
        time.sleep(length)
        GPIO.output(config.BUZZER_PIN, GPIO.LOW)
        time.sleep(0.1)


def get_distance():
    GPIO.output(config.TRIG_PIN, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(config.TRIG_PIN, GPIO.LOW)

    # give up after 40 ms so a missed echo does not hang the loop
    timeout = time.time() + 0.04
    start = time.time()
    while GPIO.input(config.ECHO_PIN) == 0:
        start = time.time()
        if start > timeout:
            return None
    stop = start
    while GPIO.input(config.ECHO_PIN) == 1:
        stop = time.time()
        if stop > timeout:
            return None
    return (stop - start) * 34300 / 2  # cm


def set_door(servo, angle):
    servo.ChangeDutyCycle(2.5 + angle / 18)
    time.sleep(0.5)
    servo.ChangeDutyCycle(0)  # stops the servo from jittering


def read_temperature(sensor):
    readings = []
    for _ in range(5):
        readings.append(sensor.get_object_1())
        time.sleep(0.1)
    return sum(readings) / len(readings) + config.TEMP_OFFSET


def check_mask(cam, detector):
    # returns the average no mask score, or None if no face was seen
    # throw away old frames sitting in the camera buffer
    for _ in range(5):
        cam.read()
    scores = []
    end = time.time() + config.MASK_CHECK_TIME
    while time.time() < end and len(scores) < 10:
        ok, frame = cam.read()
        if not ok:
            continue
        box, score = detector.check(frame)
        if box is not None:
            scores.append(score)
        if config.SHOW_CAMERA:
            if box is not None:
                cv2.rectangle(frame, box[:2], box[2:], (255, 255, 0), 2)
            cv2.imshow("Sentinel", frame)
            cv2.waitKey(1)
    if len(scores) < 3:
        return None
    return sum(scores) / len(scores)


def wait_for_hand():
    end = time.time() + config.SANITIZE_WAIT
    while time.time() < end:
        d = get_distance()
        if d is not None and d < config.HAND_DISTANCE:
            return True
        time.sleep(0.1)
    return False


def let_through(servo):
    # open the door and close it once the person has walked through,
    # returns False if nobody went through before the time ran out
    set_door(servo, config.DOOR_OPEN_ANGLE)
    passed = False
    end = time.time() + config.DOOR_OPEN_TIME
    while time.time() < end:
        if ir_blocked(config.IR_DOOR_PIN):
            passed = True
        elif passed:
            break
        time.sleep(0.05)
    time.sleep(0.5)
    set_door(servo, config.DOOR_CLOSED_ANGLE)
    return passed


def log_entry(result, temp, score, inside):
    folder = os.path.dirname(config.LOG_FILE)
    if folder:
        os.makedirs(folder, exist_ok=True)
    new_file = not os.path.exists(config.LOG_FILE)
    with open(config.LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["time", "result", "temperature", "no_mask_score", "inside"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), result,
                         "" if temp is None else "%.1f" % temp,
                         "" if score is None else "%.2f" % score, inside])


def check_person(cam, detector, sensor, inside):
    # returns (allowed, result, temperature, no mask score)
    if inside >= config.MAX_PEOPLE:
        print("Building is full")
        return False, "full", None, None

    print("Checking mask...")
    score = check_mask(cam, detector)
    if score is None:
        print("No face found, look at the camera")
        return False, "no face", None, None
    if score > config.NO_MASK_LIMIT:
        print("No mask (score %.2f)" % score)
        return False, "no mask", None, score
    print("Mask ok (score %.2f)" % score)
    beep()

    temp = read_temperature(sensor)
    if temp > config.TEMP_LIMIT:
        print("Temperature too high: %.1f C" % temp)
        return False, "high temperature", temp, score
    print("Temperature ok: %.1f C" % temp)
    beep()

    print("Put your hand under the sanitizer")
    if not wait_for_hand():
        print("Hands not sanitized")
        return False, "not sanitized", temp, score
    print("Hands sanitized")
    return True, "allowed", temp, score


def main():
    servo = setup_gpio()
    set_door(servo, config.DOOR_CLOSED_ANGLE)
    bus = SMBus(1)
    sensor = MLX90614(bus, address=config.MLX_ADDRESS)
    detector = MaskDetector(config.MODEL_PATH, config.FACE_PROTOTXT, config.FACE_WEIGHTS)
    cam = cv2.VideoCapture(config.CAMERA_INDEX)

    inside = 0
    checked = False  # so a person standing at the check is only checked once
    print("Sentinel ready")
    try:
        while True:
            if ir_blocked(config.IR_EXIT_PIN):
                print("Person leaving")
                if let_through(servo):
                    inside = max(0, inside - 1)
                    log_entry("exit", None, None, inside)
                print("People inside:", inside)

            elif ir_blocked(config.IR_CHECK_PIN) and not checked:
                checked = True
                print("Person at the entrance")
                allowed, result, temp, score = check_person(cam, detector, sensor, inside)
                if allowed:
                    beep(2, 0.1)
                    if let_through(servo):
                        inside += 1
                    else:
                        result = "did not enter"
                else:
                    beep(3, 0.5)
                log_entry(result, temp, score, inside)
                print("People inside:", inside)

            elif not ir_blocked(config.IR_CHECK_PIN):
                checked = False

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Stopping")
    finally:
        set_door(servo, config.DOOR_CLOSED_ANGLE)
        cam.release()
        cv2.destroyAllWindows()
        bus.close()
        servo.stop()
        GPIO.cleanup()


if __name__ == "__main__":
    main()
