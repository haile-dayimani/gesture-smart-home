# Copies fix-pi-hotspot.sh to the Pi over the Ethernet cable and runs it.
# You'll be asked for your Pi (sudo) password and the hotspot Wi-Fi password.
# Pi's link-local address on the laptop's Ethernet port (interface 8)
$pi = 'raspberrypi5.local'   # or the Pi's IP address
Write-Host "Pi found at $pi"
$sh = Join-Path $PSScriptRoot 'fix-pi-hotspot.sh'
# tr strips the CRs Windows adds (including the one the pipe appends)
Get-Content -Raw $sh | ssh "pi@$pi" 'tr -d \\r > ~/fix-pi-hotspot.sh'
ssh -t "pi@$pi" 'bash ~/fix-pi-hotspot.sh'
