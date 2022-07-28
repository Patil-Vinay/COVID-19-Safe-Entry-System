# Tests the parts of the circuit one at a time before running sentinel.py.
# Camera and mask model are tested with mask_detector.py.
import time

import RPi.GPIO as GPIO
from smbus2 import SMBus
from mlx90614 import MLX90614

import config
from sentinel import beep, get_distance, ir_blocked, read_temperature, set_door, setup_gpio

servo = setup_gpio()
try:
    print("Buzzer: 2 beeps")
    beep(2)

    print("Servo: opening door")
    set_door(servo, config.DOOR_OPEN_ANGLE)
    time.sleep(1)
    print("Servo: closing door")
    set_door(servo, config.DOOR_CLOSED_ANGLE)

    print("IR sensors: put your hand in front of each one (10 s)")
    sensors = {"check": config.IR_CHECK_PIN, "door": config.IR_DOOR_PIN, "exit": config.IR_EXIT_PIN}
    seen = set()
    end = time.time() + 10
    while time.time() < end and len(seen) < len(sensors):
        for name, pin in sensors.items():
            if name not in seen and ir_blocked(pin):
                print("  %s sensor ok" % name)
                seen.add(name)
        time.sleep(0.05)
    for name in sensors:
        if name not in seen:
            print("  %s sensor did not trigger" % name)

    print("Ultrasonic: 5 readings")
    for _ in range(5):
        d = get_distance()
        print("  no echo" if d is None else "  %.1f cm" % d)
        time.sleep(0.5)

    print("Temperature sensor")
    bus = SMBus(1)
    sensor = MLX90614(bus, address=config.MLX_ADDRESS)
    print("  ambient %.1f C, object %.1f C" % (sensor.get_ambient(), read_temperature(sensor)))
    bus.close()
finally:
    servo.stop()
    GPIO.cleanup()
