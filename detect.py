"""
Step 3: real-time gesture control.

Reads the camera, classifies the hand inside the green box every frame and,
once a gesture is held steadily, acts on the selected appliance:

    one   -> select LIGHT    palm  -> ON / OFF        like    -> brighter / faster
    peace -> select FAN      three -> everything OFF  dislike -> dimmer / slower

By default nothing physical is switched (simulated mode). Add --hardware on
the Raspberry Pi to drive the relays (power) and MOSFETs (brightness/speed).

Keys: Q = quit, 1-6 = fake a gesture (works even without a trained model):
1 one, 2 peace, 3 palm, 4 like, 5 dislike, 6 three.

Examples:
    python detect.py                      # laptop webcam, simulated devices
    python detect.py --hardware           # Raspberry Pi with real devices
    python detect.py --hardware --headless --log   # no screen, log to CSV
"""

import argparse
import collections
import csv
import json
import signal
import threading
import time
import urllib.request

import cv2
import numpy as np

import config
from appliances import Appliances
from camera import Camera, add_camera_arguments, crop_roi, get_roi_box
from status_leds import StatusLeds


class GestureStabiliser:
    """
    Turns noisy per-frame predictions into deliberate commands.

    A gesture only fires when it wins most of the last few frames with high
    confidence. The same gesture will not fire again until the hand is
    removed (or another gesture is shown), so holding a thumbs-up changes
    the channel once, not 30 times a second.
    """

    def __init__(self):
        self.recent = collections.deque(maxlen=config.SMOOTHING_FRAMES)
        self.last_fired = None
        self.last_fire_time = 0.0

    def update(self, label, confidence):
        """Returns the gesture to act on now, or None."""
        if confidence < config.CONFIDENCE_THRESHOLD:
            label = "none"
        self.recent.append(label)
        if len(self.recent) < self.recent.maxlen:
            return None

        winner, votes = collections.Counter(self.recent).most_common(1)[0]
        if votes < config.STABLE_FRACTION * len(self.recent):
            return None
        if winner == "none":
            self.last_fired = None      # hand gone: allow a repeat next time
            return None
        if winner == self.last_fired:
            return None
        if time.time() - self.last_fire_time < config.ACTION_COOLDOWN_S:
            return None

        self.last_fired = winner
        self.last_fire_time = time.time()
        return winner


class StatusSender:
    """
    Sends the latest prediction to webcam_stream.py on the laptop, which draws
    it on its preview window (used with --status-url when running headless).
    Only the newest status is sent, so a slow network never delays detection.
    """

    def __init__(self, url):
        self.url = url
        self.latest = None
        self.event = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def send(self, **status):
        self.latest = status
        self.event.set()

    def _run(self):
        while True:
            self.event.wait()
            self.event.clear()
            body = json.dumps(self.latest).encode()
            request = urllib.request.Request(self.url, data=body, method="POST",
                                             headers={"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(request, timeout=1).close()
            except OSError:
                time.sleep(0.5)


def draw_overlay(frame, label, confidence, probs, labels, fps, infer_ms, backend):
    x1, y1, x2, y2 = get_roi_box(frame)
    sure = confidence >= config.CONFIDENCE_THRESHOLD and label != "none"
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if sure else (0, 200, 255), 2)

    def text(msg, y, colour=(255, 255, 255), scale=0.6):
        cv2.putText(frame, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
        cv2.putText(frame, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2)

    text(f"{label}  {confidence * 100:.0f}%", 32, (0, 255, 0) if sure else (0, 200, 255), 0.9)
    text(f"FPS {fps:.1f}   model {infer_ms:.0f} ms ({backend})", 60)

    # Small bar chart of every class probability.
    if probs is not None:
        for i, (name, p) in enumerate(zip(labels, probs)):
            y = 90 + i * 20
            cv2.rectangle(frame, (10, y - 12), (10 + int(120 * p), y + 2), (0, 180, 0), -1)
            cv2.putText(frame, name, (135, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)
    text("Q quit | 1-6 fake gesture", frame.shape[0] - 12, scale=0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hardware", action="store_true",
                        help="drive the real relay and fan through the Pi GPIO")
    parser.add_argument("--headless", action="store_true",
                        help="no video window (e.g. running over SSH)")
    parser.add_argument("--keras", action="store_true",
                        help="use the Keras model instead of TFLite")
    parser.add_argument("--log", action="store_true",
                        help="save every prediction to logs/ as a CSV file")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after this many frames (0 = run forever)")
    parser.add_argument("--status-url", default=None,
                        help="POST each prediction here (webcam_stream.py preview)")
    add_camera_arguments(parser)
    args = parser.parse_args()

    classifier = None
    try:
        from classifier import GestureClassifier
        classifier = GestureClassifier(prefer_tflite=not args.keras)
        print(f"Loaded {classifier.backend} model with gestures {classifier.labels}")
    except (FileNotFoundError, ImportError) as error:
        print(f"No model ({error}).\nRunning in keyboard-only mode: press 1-6.")

    appliances = Appliances(use_hardware=args.hardware)
    leds = StatusLeds(appliances, use_hardware=args.hardware)

    # Closing the SSH window sends SIGHUP; treat it like Ctrl+C so everything
    # still switches off and the stop pattern plays.
    def stop(*_):
        raise KeyboardInterrupt
    for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if sig is not None:
            signal.signal(sig, stop)
    cam = Camera(args.camera, args.camera_index, args.video)
    print(f"Using {cam.name}. Mode: {'HARDWARE' if args.hardware else 'SIMULATED'}")

    stabiliser = GestureStabiliser()
    status = StatusSender(args.status_url) if args.status_url else None
    last_action = ""
    log_writer = log_file = None
    if args.log:
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = config.LOGS_DIR / time.strftime("detect_%Y%m%d_%H%M%S.csv")
        log_file = open(log_path, "w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow(["time", "gesture", "confidence", "inference_ms",
                             "fps", "action"])
        print(f"Logging to {log_path}")

    leds.startup()
    frames = 0
    fps = 0.0
    last_time = time.perf_counter()
    try:
        while True:
            frame = cam.read()
            if frame is None:
                break
            frames += 1

            label, confidence, probs, infer_ms = "none", 0.0, None, 0.0
            if classifier is not None:
                label, confidence, probs, infer_ms = classifier.predict(crop_roi(frame))
            gesture = stabiliser.update(label, confidence)

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_time, 1e-6))
            last_time = now

            key = -1
            if not args.headless:
                draw_overlay(frame, label, confidence, probs,
                             classifier.labels if classifier else [],
                             fps, infer_ms,
                             classifier.backend if classifier else "no model")
                panel = appliances.draw_dashboard(frame.shape[0])
                cv2.imshow("Gesture-controlled devices", np.hstack([frame, panel]))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key != 255 and chr(key) in config.KEY_TO_GESTURE:
                    gesture = config.KEY_TO_GESTURE[chr(key)]

            action_message = ""
            if gesture in config.GESTURE_ACTIONS:
                target = appliances.selected
                was_on, was_level = target.on, target.level
                action_message = appliances.perform(config.GESTURE_ACTIONS[gesture])
                leds.gesture(gesture, appliances.selected if gesture in ("one", "peace")
                             else target, was_on, was_level)
                print(f"[{time.strftime('%H:%M:%S')}] {gesture:8s} -> {action_message}")
                last_action = f"{gesture} -> {action_message}"

            if status:
                status.send(label=label, confidence=round(float(confidence), 3),
                            sure=bool(confidence >= config.CONFIDENCE_THRESHOLD
                                      and label != "none"),
                            fps=round(fps, 1), infer_ms=round(infer_ms, 1),
                            action=last_action,
                            selected=appliances.selected.name,
                            devices={d.name: (d.level_text() if d.on else "OFF")
                                     for d in (appliances.light, appliances.fan)})

            if log_writer:
                log_writer.writerow([f"{time.time():.3f}", label, f"{confidence:.3f}",
                                     f"{infer_ms:.1f}", f"{fps:.1f}", action_message])

            if args.max_frames and frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        appliances.close()
        leds.shutdown()
        leds.close()
        cam.release()
        if log_file:
            log_file.close()
        if not args.headless:
            cv2.destroyAllWindows()
        print(f"Stopped after {frames} frames. Everything switched off.")


if __name__ == "__main__":
    main()
