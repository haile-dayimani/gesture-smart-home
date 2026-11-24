# Runs the gesture detector on the Pi using the laptop webcam (until the Pi camera arrives).
# Usage (from this folder):
#   powershell -ExecutionPolicy Bypass -File .\run-with-laptop-webcam.ps1             # simulated devices
#   powershell -ExecutionPolicy Bypass -File .\run-with-laptop-webcam.ps1 -Hardware   # drive the relay/fan
# A window shows what the webcam sends; the Pi's actions print here. Ctrl+C stops both.
param([switch]$Hardware, [int]$Index = 0)

Set-Location $PSScriptRoot
$stream = Start-Process -PassThru -WindowStyle Hidden .\laptop-venv\Scripts\python.exe `
    -ArgumentList "webcam_stream.py --preview --index $Index"
Start-Sleep 3
$hw = if ($Hardware) { "--hardware" } else { "" }
try {
    ssh -t -o ExitOnForwardFailure=yes -R 8080:127.0.0.1:8080 pi@raspberrypi.local `
        "cd ~/gesture/gesture_prototype && . ../venv/bin/activate && python detect.py --headless $hw --video http://127.0.0.1:8080/video --status-url http://127.0.0.1:8080/status 2>&1 | grep -vE '^(W0000|I0000|WARNING|INFO)'"
} finally {
    Stop-Process -Id $stream.Id -ErrorAction SilentlyContinue
}
