"""
Step 1: record your own hand-gesture dataset.

Shows the camera with a green box. Put your hand in the box, press SPACE and
the script saves a picture of the box every --interval seconds until it has
--count pictures for that gesture. Then it moves on to the next gesture.

Pictures go to dataset/<gesture>/. Running it again ADDS pictures, it never
deletes old ones, so you can record in several sessions (different rooms,
lighting, people), which makes the model much more reliable.

Examples:
    python collect_data.py                      # all gestures, 60 each
    python collect_data.py --gesture palm --count 100
"""

import argparse
import time

import cv2

import config
from camera import Camera, add_camera_arguments, crop_roi, get_roi_box

TIPS = {
    "none": "Empty box, background, arm passing by",
    "palm": "Open hand, fingers together, palm to camera",
    "one": "Index finger up",
    "peace": "Index + middle finger up (V)",
    "three": "Three fingers up",
    "like": "Thumbs up",
    "dislike": "Thumbs down",
}


def count_images(gesture):
    folder = config.DATASET_DIR / gesture
    return len(list(folder.glob("*.jpg"))) if folder.exists() else 0


def put_text(frame, text, y, colour=(255, 255, 255), scale=0.6):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 4)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                colour, 2)


def collect_gesture(cam, gesture, count, interval, session):
    """Record `count` pictures of one gesture. Returns False if user quit."""
    folder = config.DATASET_DIR / gesture
    folder.mkdir(parents=True, exist_ok=True)

    saved = 0
    recording = False
    last_save = 0.0
    while saved < count:
        frame = cam.read()
        if frame is None:
            print("Camera stopped giving frames.")
            return False

        now = time.time()
        if recording and now - last_save >= interval:
            name = f"{session}_{int(now * 1000)}_{saved:04d}.jpg"
            cv2.imwrite(str(folder / name), crop_roi(frame))
            saved += 1
            last_save = now

        x1, y1, x2, y2 = get_roi_box(frame)
        box_colour = (0, 0, 255) if recording else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_colour, 2)
        put_text(frame, f"Gesture: {gesture.upper()}", 30, (0, 255, 255), 0.8)
        put_text(frame, TIPS.get(gesture, ""), 60)
        put_text(frame, f"Saved {saved}/{count}  (total in folder: "
                        f"{count_images(gesture)})", 90)
        if recording:
            put_text(frame, "RECORDING - move your hand around a bit", 120,
                     (0, 0, 255))
        else:
            put_text(frame, "SPACE = start/pause   N = skip   Q = quit", 120)

        cv2.imshow("Collect gesture data", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False
        if key == ord(" "):
            recording = not recording
        if key == ord("n"):
            break

    print(f"  {gesture}: saved {saved} new pictures")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gesture", choices=config.GESTURES,
                        help="record only this gesture (default: all of them)")
    parser.add_argument("--count", type=int, default=60,
                        help="pictures to save per gesture (default 60)")
    parser.add_argument("--interval", type=float, default=0.4,
                        help="seconds between pictures (default 0.4)")
    parser.add_argument("--session", default=time.strftime("s%Y%m%d_%H%M"),
                        help="tag added to file names to tell sessions apart")
    add_camera_arguments(parser)
    args = parser.parse_args()

    gestures = [args.gesture] if args.gesture else config.GESTURES
    cam = Camera(args.camera, args.camera_index, args.video)
    print(f"Using {cam.name}. Saving to {config.DATASET_DIR}")

    try:
        for gesture in gestures:
            if not collect_gesture(cam, gesture, args.count, args.interval,
                                   args.session):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

    print("\nPictures per gesture:")
    for gesture in config.GESTURES:
        print(f"  {gesture:10s} {count_images(gesture)}")


if __name__ == "__main__":
    main()
