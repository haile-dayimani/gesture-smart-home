"""
Streams the laptop webcam as MJPEG at http://localhost:8080/video so the Pi
can use it instead of the Pi camera (stand-in until the ribbon cable arrives).

The Pi reaches it through an SSH tunnel, so no firewall change is needed:
    ssh -R 8080:127.0.0.1:8080 hailepi@raspberrypi5.local
and on the Pi:
    python detect.py --hardware --headless --video http://127.0.0.1:8080/video \
        --status-url http://127.0.0.1:8080/status
(Use 127.0.0.1, not localhost: on Windows localhost can resolve to IPv6.)

With --preview, a window shows the webcam (mirrored, like the Pi sees it), the
green hand box, and the Pi's live prediction posted back to /status.

Run:  laptop-venv\\Scripts\\python webcam_stream.py [--index 0] [--preview]
Or use run-with-laptop-webcam.ps1, which does all of the above.
"""

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

# Reuse the Pi's box position and mirror setting so the preview matches it.
sys.path.insert(0, str(Path(__file__).resolve().parent / "gesture_prototype"))
import config  # noqa: E402
from camera import get_roi_box  # noqa: E402

latest_jpeg = None
lock = threading.Lock()
pi_status = {}
pi_status_time = 0.0


def put_text(frame, text, y, colour=(255, 255, 255), scale=0.6):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2)


def draw_preview(frame):
    """Mirror like the Pi does, then draw the hand box and the Pi's prediction."""
    view = cv2.flip(frame, 1) if config.MIRROR else frame.copy()
    if view.shape[1] != config.FRAME_WIDTH:
        view = cv2.resize(view, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
    status = pi_status if time.time() - pi_status_time < 2 else {}
    sure = status.get("sure", False)
    colour = (0, 255, 0) if sure else (0, 200, 255)
    x1, y1, x2, y2 = get_roi_box(view)
    cv2.rectangle(view, (x1, y1), (x2, y2), colour, 2)
    if status:
        put_text(view, f"{status['label']}  {status['confidence'] * 100:.0f}%", 32, colour, 0.9)
        put_text(view, f"Pi: {status['fps']:.1f} FPS, model {status['infer_ms']:.0f} ms", 60)
        if status.get("action"):
            put_text(view, f"Last action: {status['action']}", 88, (255, 255, 0))
        y = 116
        for name, state in status.get("devices", {}).items():
            marker = ">" if name == status.get("selected") else " "
            put_text(view, f"{marker} {name}: {state}", y,
                     (0, 255, 0) if marker == ">" else (255, 255, 255))
            y += 26
    else:
        put_text(view, "Waiting for the Pi...", 32, (0, 200, 255), 0.8)
    put_text(view, "Hand inside the box | Q quits", view.shape[0] - 12, scale=0.5)
    return view


def capture_loop(index, width, height, preview):
    global latest_jpeg
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam {index}")
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with lock:
                latest_jpeg = jpeg.tobytes()
        if preview:
            cv2.imshow("Gesture control (laptop webcam -> Pi)", draw_preview(frame))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                raise SystemExit


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/video":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        last = None
        try:
            while True:
                with lock:
                    jpeg = latest_jpeg
                if jpeg is None or jpeg is last:
                    time.sleep(0.01)
                    continue
                last = jpeg
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                 + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                                 + jpeg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_POST(self):
        global pi_status, pi_status_time
        if self.path != "/status":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            pi_status = json.loads(self.rfile.read(length))
            pi_status_time = time.time()
        except ValueError:
            pass
        self.send_response(204)
        self.end_headers()

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=int, default=0, help="webcam number")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--preview", action="store_true",
                        help="show the webcam, hand box and Pi prediction in a window")
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), StreamHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Streaming webcam {args.index} at http://localhost:{args.port}/video  (Ctrl+C to stop)")
    try:
        capture_loop(args.index, args.width, args.height, args.preview)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
