#!/bin/bash
# Re-creates the Pi's Wi-Fi connection to the laptop hotspot.
# Asks for the sudo password and then the hotspot Wi-Fi password (typed by you, never stored here).
set -e
SSID='YOUR_HOTSPOT_SSID'   # set to your hotspot name
if nmcli -t -f NAME con show | grep -qx hotspot; then
    echo "A 'hotspot' connection already exists; updating it."
else
    sudo nmcli con add type wifi ifname wlan0 con-name hotspot ssid "$SSID" \
        wifi-sec.key-mgmt wpa-psk
fi
read -rsp "Hotspot Wi-Fi password: " PSK; echo
sudo nmcli con modify hotspot 802-11-wireless.ssid "$SSID" wifi-sec.key-mgmt wpa-psk \
    wifi-sec.psk "$PSK" connection.autoconnect yes connection.autoconnect-priority 10 \
    wifi.cloned-mac-address permanent
unset PSK
sudo sync
echo "Saved. Connections:"
nmcli -f NAME,TYPE,AUTOCONNECT,AUTOCONNECT-PRIORITY con show
echo "Bringing it up..."
sudo nmcli con up hotspot || echo "Could not connect yet (is the hotspot visible?). It will auto-connect when it is."
ip -4 addr show wlan0 | grep inet || true
