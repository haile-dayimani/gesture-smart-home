# Copies the gesture_prototype code (not the dataset) from this laptop to the Pi.
#   powershell -ExecutionPolicy Bypass -File .\send-code-to-pi.ps1
Set-Location $PSScriptRoot
$files = Get-ChildItem -Path gesture_prototype\* -File -Include *.py, *.md, *.txt -Exclude train_log.txt
scp $files.FullName hailepi@raspberrypi5.local:gesture/gesture_prototype/
if ($LASTEXITCODE -eq 0) { "Code copied to the Pi." } else { "Copy failed. Is the Pi on the hotspot?" }
