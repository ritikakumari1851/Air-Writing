"""All tunable settings in one place."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
USER_DATA_DIR = ROOT / "user_data"

HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/latest/hand_landmarker.task")
CHAR_MODEL_PATH = MODELS_DIR / "char_cnn.npz"

# BGR colours
BLUE = (255, 120, 30)
GREEN = (60, 200, 60)
RED = (60, 60, 235)
YELLOW = (40, 220, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK = (35, 35, 35)
ACCENT = (255, 190, 60)
PALETTE = {"blue": BLUE, "green": GREEN, "red": RED, "yellow": YELLOW}


@dataclass
class Settings:
    # camera
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    mirror: bool = True

    # hand tracking (MediaPipe)
    min_detection_confidence: float = 0.6
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # gestures
    gesture_stable_frames: int = 3       # frames a gesture must last before it counts
    palm_hold_seconds: float = 0.8       # open palm held this long -> clear canvas
    fist_hold_seconds: float = 0.35      # fist held this long -> recognise now
    dwell_seconds: float = 0.45          # hover over a toolbar button this long -> press it

    # fingertip smoothing (One Euro filter)
    filter_min_cutoff: float = 1.5
    filter_beta: float = 0.01

    # drawing
    brush_thickness: int = 9
    eraser_radius: int = 40
    min_stroke_length: float = 12.0      # px; shorter strokes are treated as accidental
    max_jump_ratio: float = 0.25         # a jump bigger than this * frame diagonal breaks the stroke

    # recognition
    keep_ink: bool = True                # characters stay on screen after they are read, until you clear them
    recognize_delay: float = 1.5         # seconds without drawing before a character is read
    min_char_size_ratio: float = 0.04    # characters smaller than this * frame height are ignored
    segment_characters: bool = True      # allow several characters in one go (side by side)

    # look
    show_landmarks: bool = True
    background: str = "camera"           # camera | dim | blackboard | whiteboard

    charsets: dict = field(default_factory=lambda: {
        "A-Z 0-9": list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        "A-Z": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        "0-9": list("0123456789"),
    })
