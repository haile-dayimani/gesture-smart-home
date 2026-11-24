# Copies the trained model (gesture_prototype\models) to the Pi after train.py finishes.
#   powershell -ExecutionPolicy Bypass -File .\send-model-to-pi.ps1
Set-Location $PSScriptRoot
$models = "gesture_prototype\models"
if (-not (Test-Path "$models\gesture_model.tflite")) { throw "No $models\gesture_model.tflite yet. Run train.py first." }
scp -r $models pi@raspberrypi.local:gesture/gesture_prototype/
if ($LASTEXITCODE -eq 0) { "Model copied to the Pi. Now run .\run-with-laptop-webcam.ps1" }
