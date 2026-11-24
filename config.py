"""
Shared settings for the gesture-controlled devices prototype.

Every script imports this file, so change things here (not in the scripts).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"   # dataset/<gesture_name>/*.jpg
MODELS_DIR = PROJECT_DIR / "models"     # trained model, labels, plots
LOGS_DIR = PROJECT_DIR / "logs"         # CSV logs from detect.py

KERAS_MODEL_PATH = MODELS_DIR / "gesture_model.keras"
TFLITE_MODEL_PATH = MODELS_DIR / "gesture_model.tflite"
LABELS_PATH = MODELS_DIR / "labels.txt"

# ---------------------------------------------------------------------------
# Gestures
# ---------------------------------------------------------------------------
# One folder per gesture is created in dataset/. "none" is the background
# class (empty box, random hand movement) so the system can do nothing.
GESTURES = [
    "none",      # nothing / background        -> no action
    "palm",      # open hand, fingers together  -> selected device ON / OFF
    "one",       # index finger up              -> select the LIGHT
    "peace",     # index + middle finger (V)    -> select the FAN
    "three",     # three fingers up             -> everything OFF
    "like",      # thumbs up                    -> brighter / faster
    "dislike",   # thumbs down                  -> dimmer / slower
]

# What each gesture does. Actions are handled in appliances.py.
# "one" / "peace" choose which device the other gestures act on, so six
# gestures cover power and level for both the LED light and the fan.
GESTURE_ACTIONS = {
    "one": "select_light",
    "peace": "select_fan",
    "palm": "power_toggle",
    "like": "level_up",
    "dislike": "level_down",
    "three": "all_off",
}

# Keyboard keys that fake a gesture in detect.py (handy for demos/testing).
KEY_TO_GESTURE = {
    "1": "one",
    "2": "peace",
    "3": "palm",
    "4": "like",
    "5": "dislike",
    "6": "three",
}

# ---------------------------------------------------------------------------
# Camera and region of interest (ROI)
# ---------------------------------------------------------------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIRROR = True   # flip the image so it behaves like a mirror

# The green box the hand must be inside. Values are fractions of the frame.
# ROI_SIZE is a fraction of the frame height; the box is always square.
ROI_CENTER_X = 0.70
ROI_CENTER_Y = 0.50
ROI_SIZE = 0.70

SAVE_SIZE = 224   # size (px) of the square images saved by collect_data.py

# ---------------------------------------------------------------------------
# Model and training
# ---------------------------------------------------------------------------
# MobileNetV2 accepts 96, 128, 160, 192 or 224. 160 is a good balance between
# accuracy and speed on a Raspberry Pi 5.
IMG_SIZE = 160
BATCH_SIZE = 32
EPOCHS_FROZEN = 10      # phase 1: train only the new classifier head
EPOCHS_FINE_TUNE = 10   # phase 2: also train the top of MobileNetV2
FINE_TUNE_LAYERS = 30   # how many of MobileNetV2's last layers to unfreeze
LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
SEED = 42

# Our own CNN (train.py --model cnn), trained from scratch.
CNN_IMG_SIZE = 128        # smaller input = faster on the Pi
EPOCHS_CNN = 40           # stops earlier by itself when it stops improving
CNN_LEARNING_RATE = 1e-3

# ---------------------------------------------------------------------------
# Real-time decision logic (detect.py)
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.85  # ignore predictions less sure than this
SMOOTHING_FRAMES = 5         # look at the last N predictions
STABLE_FRACTION = 0.8        # gesture must win 80% of the last N frames
ACTION_COOLDOWN_S = 1.0      # minimum seconds between two actions

# ---------------------------------------------------------------------------
# Raspberry Pi hardware (BCM GPIO numbers, not physical pin numbers)
# ---------------------------------------------------------------------------
# Each device has a relay for POWER and a MOSFET (PWM) for its LEVEL.
LIGHT_RELAY_PIN = 17     # physical pin 11 -> relay board IN1
FAN_RELAY_PIN = 27       # physical pin 13 -> relay board IN2
LIGHT_PWM_PIN = 12       # physical pin 32 -> 220 ohm -> LED MOSFET gate
FAN_PWM_PIN = 18         # physical pin 12 -> 220 ohm -> fan MOSFET gate

# Most 4-channel PC817 relay boards switch ON when IN is pulled LOW.
# If a device turns on when it should be off, set this to True.
RELAY_ACTIVE_HIGH = False

LIGHT_PWM_FREQUENCY = 1000  # Hz. High enough that the LEDs don't flicker.
FAN_PWM_FREQUENCY = 100     # Hz. Try 25-1000 if the fan whines or stutters.

# Levels that thumbs up / down step through (PWM duty cycle, 0.0 to 1.0).
LIGHT_LEVELS = [0.10, 0.25, 0.50, 0.75, 1.00]
FAN_LEVELS = [0.60, 0.80, 1.00]   # most 12 V fans stall below ~50%
LIGHT_START_LEVEL = 2             # index used the first time it turns on
FAN_START_LEVEL = 0

# Status LEDs on the breadboard: one trio per device (green, yellow, red).
#   red = device OFF, green = device ON (glows brighter at higher levels),
#   yellow = this device is SELECTED (the one palm / like / dislike act on).
# Each LED: GPIO -> 220 ohm -> LED long leg; short leg -> ground rail.
STATUS_LEDS = {
    "Light": {"green": 5, "yellow": 6, "red": 13},    # pins 29, 31, 33
    "Fan":   {"green": 19, "yellow": 26, "red": 21},  # pins 35, 37, 40
}
STATUS_LED_MIN_GLOW = 0.15   # dimmest green (lowest level) so it never looks off
