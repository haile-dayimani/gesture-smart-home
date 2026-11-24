# CPUT blocks NTP (UDP 123), so the Pi's clock drifts and apt rejects signatures/certificates.
# This installs a small service on the Pi that sets the clock from an HTTPS server's Date header
# at boot and every 30 minutes, then runs apt update. You'll be asked for the sudo password once.
$remote = @'
set -e
sudo tee /usr/local/sbin/https-time-sync >/dev/null <<'EOF'
#!/bin/sh
# Set the clock from the HTTP Date header (NTP is blocked on CPUT). --insecure is needed
# because a wrong clock makes certificate checks fail; it's only used to read the time.
for url in https://www.google.com https://deb.debian.org; do
  d=$(curl -sI --insecure -m 10 "$url" | sed -n 's/^[Dd]ate: //p' | tr -d '\r')
  [ -n "$d" ] && date -u -s "$d" >/dev/null && exit 0
done
exit 1
EOF
sudo chmod 755 /usr/local/sbin/https-time-sync
sudo tee /etc/systemd/system/https-time-sync.service >/dev/null <<'EOF'
[Unit]
Description=Set clock from HTTPS Date header
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/https-time-sync
EOF
sudo tee /etc/systemd/system/https-time-sync.timer >/dev/null <<'EOF'
[Unit]
Description=Periodic HTTPS clock sync

[Timer]
OnBootSec=30s
OnUnitActiveSec=30min

[Install]
WantedBy=timers.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now https-time-sync.timer
sudo systemctl start https-time-sync.service
echo "== Pi clock now: $(date)"
echo '== running apt update'
sudo apt update
'@
$b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remote.Replace("`r`n", "`n")))
ssh -t hailepi@raspberrypi5.local "echo $b64 | base64 -d > /tmp/fixtime.sh && bash /tmp/fixtime.sh; rm -f /tmp/fixtime.sh"
