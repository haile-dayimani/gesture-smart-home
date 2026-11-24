"""
Camera helpers shared by collect_data.py and detect.py.

Works with:
  * the Raspberry Pi Camera Module 3 (through Picamera2), and
  * any USB/laptop webcam or a video file (through OpenCV).
"""

import cv2

import config


class Camera:
    """Gives BGR frames (OpenCV format) of size FRAME_WIDTH x FRAME_HEIGHT."""

    def __init__(self, source="auto", index=0, video_path=None):
        self.picam = None
        self.cap = None

        if video_path:
            self.cap = cv2.VideoCapture(str(video_path))
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open video file: {video_path}")
            self.name = f"video file {video_path}"
            return

        if source in ("auto", "picamera"):
            try:
                self._open_picamera()
                return
            except Exception as error:  # Picamera2 missing or no Pi camera
                if source == "picamera":
                    raise RuntimeError(f"Could not open the Pi camera: {error}")

        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam {index}. Is it plugged in and not used "
                "by another program?"
            )
        self.name = f"webcam {index}"

    def _open_picamera(self):
        from picamera2 import Picamera2

        self.picam = Picamera2()
        # "RGB888" in Picamera2 is stored as BGR, which is what OpenCV expects.
        cam_config = self.picam.create_preview_configuration(
            main={"size": (config.FRAME_WIDTH, config.FRAME_HEIGHT),
                  "format": "RGB888"}
        )
        self.picam.configure(cam_config)
        self.picam.start()
        try:
            # Camera Module 3 has autofocus; keep it focusing continuously.
            from libcamera import controls
            self.picam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        except Exception:
            pass
        self.name = "Raspberry Pi camera"

    def read(self):
        """Return the next frame, or None when there are no more frames."""
        if self.picam is not None:
            frame = self.picam.capture_array()
        else:
            ok, frame = self.cap.read()
            if not ok:
                return None
            if frame.shape[1] != config.FRAME_WIDTH:
                frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        if config.MIRROR:
            frame = cv2.flip(frame, 1)
        return frame

    def release(self):
        if self.picam is not None:
            self.picam.stop()
        if self.cap is not None:
            self.cap.release()


def get_roi_box(frame):
    """Return (x1, y1, x2, y2) of the square hand box inside the frame."""
    height, width = frame.shape[:2]
    side = int(config.ROI_SIZE * height)
    cx = int(config.ROI_CENTER_X * width)
    cy = int(config.ROI_CENTER_Y * height)
    x1 = max(0, min(width - side, cx - side // 2))
    y1 = max(0, min(height - side, cy - side // 2))
    return x1, y1, x1 + side, y1 + side


def crop_roi(frame):
    """Cut out the hand box and resize it to SAVE_SIZE x SAVE_SIZE (BGR)."""
    x1, y1, x2, y2 = get_roi_box(frame)
    roi = frame[y1:y2, x1:x2]
    return cv2.resize(roi, (config.SAVE_SIZE, config.SAVE_SIZE),
                      interpolation=cv2.INTER_AREA)


def add_camera_arguments(parser):
    """Add the --camera, --camera-index and --video options to a script."""
    parser.add_argument("--camera", choices=["auto", "picamera", "webcam"],
                        default="auto",
                        help="auto tries the Pi camera first, then a webcam")
    parser.add_argument("--camera-index", type=int, default=0,
                        help="webcam number (0 is usually the built-in one)")
    parser.add_argument("--video", default=None,
                        help="read frames from a video file instead of a camera")
