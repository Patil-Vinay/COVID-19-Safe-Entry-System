# GPIO pins, BCM numbering
IR_CHECK_PIN = 17   # IR sensor where the person stands for the checks
IR_DOOR_PIN = 27    # IR sensor in the doorway
IR_EXIT_PIN = 22    # IR sensor on the inside, for people going out
TRIG_PIN = 23       # HC-SR04 at the sanitizer
ECHO_PIN = 24       # through a voltage divider, echo is 5V
SERVO_PIN = 18
BUZZER_PIN = 25

# MLX90614 on I2C bus 1
MLX_ADDRESS = 0x5A
TEMP_LIMIT = 37.5   # deg C
TEMP_OFFSET = 0.0   # change after comparing with a normal thermometer

# camera and mask model
CAMERA_INDEX = 0
MODEL_PATH = "mask_model.tflite"
NO_MASK_LIMIT = 0.5   # model score above this means no mask
MASK_CHECK_TIME = 5   # seconds to look for a face
SHOW_CAMERA = False   # set True if a screen is connected

# sanitizer
HAND_DISTANCE = 10    # cm
SANITIZE_WAIT = 10    # seconds

# door servo
DOOR_CLOSED_ANGLE = 0
DOOR_OPEN_ANGLE = 90
DOOR_OPEN_TIME = 8    # seconds to wait for the person to walk through

MAX_PEOPLE = 20
LOG_FILE = "logs/entries.csv"
