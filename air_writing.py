#!/usr/bin/env python3
"""Air Writing - write letters and digits in the air with your index finger.

    python air_writing.py                  # webcam, write mode (characters become text)
    python air_writing.py --draw           # free drawing (air canvas)
    python air_writing.py --collect        # record your own handwriting to improve the model
    python air_writing.py --video demo.mp4 # run on a recorded video instead of the webcam

Gestures:  index finger = draw | index + middle = move without drawing |
           point at a toolbar button and hold still = press it |
           fist (hold) = read now | open palm (hold) = clear the screen.   Press H for help, Q to quit.
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from airwriting.app import AirWritingApp, Collector  # noqa: E402
from airwriting.config import CHAR_MODEL_PATH, HAND_MODEL_PATH, OUTPUT_DIR, Settings  # noqa: E402

WINDOW = "Air Writing  (H = help, Q = quit)"


def parse_args():
    ap = argparse.ArgumentParser(description="Air Writing: write in the air with your fingertip.")
    ap.add_argument("--camera", type=int, default=0, help="webcam index (try 1 if 0 is not your camera)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--video", help="use a video file instead of the webcam")
    ap.add_argument("--draw", action="store_true", help="start in free-drawing mode")
    ap.add_argument("--model", default=str(CHAR_MODEL_PATH), help="character model (.npz)")
    ap.add_argument("--hand-model", default=str(HAND_MODEL_PATH), help="MediaPipe hand_landmarker.task")
    ap.add_argument("--delay", type=float, help="seconds of pause before a character is read (default 1.5)")
    ap.add_argument("--auto-clear", action="store_true",
                    help="remove each character from the screen as soon as it is read "
                         "(by default your writing stays until you clear it with an open palm)")
    ap.add_argument("--no-mirror", action="store_true", help="do not mirror the image (e.g. a video that is already mirrored)")
    ap.add_argument("--background", choices=["camera", "dim", "blackboard", "whiteboard"], default="camera")
    ap.add_argument("--collect", nargs="?", const="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", metavar="CHARS",
                    help="record handwriting samples (default: all digits and letters)")
    ap.add_argument("--per-char", type=int, default=5, help="samples per character in --collect mode")
    ap.add_argument("--record", metavar="OUT.mp4", help="also save the session as a video")
    return ap.parse_args()


def open_camera(index: int, width: int, height: int):
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if platform.system() == "Windows" else [cv2.CAP_ANY]
    for be in backends:
        cap = cv2.VideoCapture(index, be)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # faster HD on most webcams
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, frame
            cap.release()
    return None, None


def main():
    args = parse_args()
    cfg = Settings(camera_index=args.camera, width=args.width, height=args.height,
                   mirror=not args.no_mirror, background=args.background)
    if args.delay is not None:
        cfg.recognize_delay = max(0.3, args.delay)
    cfg.keep_ink = not args.auto_clear

    # video source
    if args.video:
        cap = cv2.VideoCapture(args.video)
        ok, frame = cap.read()
        if not ok:
            sys.exit(f"Could not read video: {args.video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    else:
        cap, frame = open_camera(args.camera, args.width, args.height)
        if cap is None:
            sys.exit(f"Could not open camera {args.camera}.\n"
                     "  * try another index:  python air_writing.py --camera 1\n"
                     "  * close other apps that use the camera (Zoom, Teams, browser tabs)\n"
                     "  * macOS: allow camera access for Terminal / your IDE in System Settings > Privacy & Security")
        fps = None
    if cfg.mirror:          # writing seen by the camera is mirrored; flip it back
        frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # models
    recognizer = None
    try:
        from airwriting.recognizer import Recognizer
        recognizer = Recognizer(args.model)
    except FileNotFoundError as exc:
        print(f"Warning: {exc}\nStarting in drawing mode only.")
    from airwriting.hand_tracker import HandTracker, ensure_hand_model
    try:
        hand_model = ensure_hand_model(Path(args.hand_model))
    except RuntimeError as exc:
        sys.exit(str(exc))
    try:
        tracker = HandTracker(hand_model, min_detection_confidence=cfg.min_detection_confidence,
                              min_presence_confidence=cfg.min_presence_confidence,
                              min_tracking_confidence=cfg.min_tracking_confidence)
    except Exception as exc:
        sys.exit(f"Could not start the MediaPipe hand tracker: {exc}\n"
                 f"If the model file is damaged, delete {hand_model} and run again to re-download it.")

    collector = Collector(args.collect, args.per_char) if args.collect else None
    app = AirWritingApp(w, h, recognizer, cfg, collector=collector,
                        start_mode="DRAW" if (args.draw and not args.collect) else None)

    writer = None
    if args.record:
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), fps or 30.0, (w, h))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, w, h)
    print(f"Camera {w}x{h}. Press H in the window for help, Q to quit.")
    if collector:
        print(f"Collect mode: samples are saved to {collector.path}")

    frame_no = 0
    window_seen = False      # some OpenCV builds cannot report visibility; only trust it once it has said "visible"
    t0 = time.monotonic()
    try:
        while True:
            if frame_no > 0:
                ok, frame = cap.read()
                if not ok or frame is None:
                    if args.video:
                        break
                    time.sleep(0.01)
                    continue
                if cfg.mirror:
                    frame = cv2.flip(frame, 1)
            if args.video:
                now = frame_no / fps
                ts = int(now * 1000)
            else:
                now = time.monotonic() - t0
                ts = None
            hand = tracker.process(frame, ts)
            out = app.step(frame, hand, now)
            if writer is not None:
                writer.write(out)
            cv2.imshow(WINDOW, out)
            frame_no += 1
            key = cv2.waitKey(1)
            if key != -1 and not app.handle_key(key & 0xFF):
                break
            try:                                     # window closed with its X button?
                visible = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE)
            except cv2.error:
                break
            if visible >= 1:
                window_seen = True
            elif window_seen:
                break
    finally:
        cap.release()
        tracker.close()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        if app.text.strip():
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUTPUT_DIR / "last_session.txt").write_text(app.text + "\n", encoding="utf-8")
            print(f"Text: {app.text}\n(saved to {OUTPUT_DIR / 'last_session.txt'})")


if __name__ == "__main__":
    main()
