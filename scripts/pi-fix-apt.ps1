# Installs the laptop's SSH key on the Pi and switches apt sources to HTTPS.
# You'll be asked for the Pi password twice: once to log in, once for sudo.
$key = (Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub" -Raw).Trim()
$remote = @"
set -e
mkdir -p ~/.ssh && chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
grep -qxF '$key' ~/.ssh/authorized_keys || echo '$key' >> ~/.ssh/authorized_keys
echo '== SSH key installed'
sudo cp -a /etc/apt /root/apt-backup-20261101 2>/dev/null || true
for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do
  [ -f "`$f" ] && sudo sed -i -E 's#http://(deb\.debian\.org|security\.debian\.org|archive\.raspberrypi\.(com|org))#https://\1#g' "`$f"
done
echo '== apt sources now:'
grep -rhE '^(URIs|deb )' /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null || true
echo '== running apt update'
sudo apt update
"@
$b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remote.Replace("`r`n", "`n")))
ssh -t hailepi@raspberrypi5.local "echo $b64 | base64 -d > /tmp/fixapt.sh && bash /tmp/fixapt.sh; rm -f /tmp/fixapt.sh"
