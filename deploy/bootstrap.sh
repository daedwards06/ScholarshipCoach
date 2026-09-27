#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 24.04 server, run as root. Safe to re-run:
# every step checks before it changes anything.
#
#   sudo bash bootstrap.sh
#
# Run it only after `tailscale up --ssh` has joined the server to the tailnet.
# The firewall step closes every public port, including SSH on 22.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/daedwards06/ScholarshipCoach.git}"
APP_DIR=/srv/scholarshipcoach
APP_USER=coach
SERVICE=scholarshipcoach

log() { printf '\n==> %s\n' "$*"; }
as_coach() { sudo -u "$APP_USER" -H "$@"; }

if [[ $EUID -ne 0 ]]; then
    echo "bootstrap.sh must run as root" >&2
    exit 1
fi
if ! ip link show tailscale0 >/dev/null 2>&1; then
    echo "tailscale0 not found. Run 'tailscale up --ssh' first, or this script's" >&2
    echo "firewall step will lock you out of the server." >&2
    exit 1
fi

log "Packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q python3.12-venv git sqlite3 restic ufw unattended-upgrades curl

log "Automatic security updates"
cat > /etc/apt/apt.conf.d/20auto-upgrades <<'EOF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
EOF

log "2 GB swap file"
if ! swapon --show=NAME --noheadings | grep -qx /swapfile; then
    [[ -f /swapfile ]] || fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
fi
grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab

log "User $APP_USER"
id "$APP_USER" >/dev/null 2>&1 || useradd --create-home --shell /bin/bash "$APP_USER"

log "Clone into $APP_DIR"
if [[ ! -d $APP_DIR/.git ]]; then
    install -d -o "$APP_USER" -g "$APP_USER" "$APP_DIR"
    as_coach git clone "$REPO_URL" "$APP_DIR"
fi

log "Virtualenv"
[[ -x $APP_DIR/.venv/bin/python ]] || as_coach python3.12 -m venv "$APP_DIR/.venv"
as_coach "$APP_DIR/.venv/bin/pip" install -q --upgrade pip
# CPU-only torch before the package, same order as CI: otherwise pip pulls the
# CUDA build (~3 GB of nvidia wheels) that this server has no GPU for.
as_coach "$APP_DIR/.venv/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
as_coach "$APP_DIR/.venv/bin/pip" install -q -e "$APP_DIR" -c "$APP_DIR/constraints-ci.txt"

log "Firewall: deny inbound except Tailscale"
ufw default deny incoming
ufw default allow outgoing
ufw allow in on tailscale0
ufw allow 41641/udp
ufw --force enable

log "Let $APP_USER restart the service (used by deploy/update.sh)"
SUDOERS=/etc/sudoers.d/scholarshipcoach
cat > "$SUDOERS.tmp" <<EOF
$APP_USER ALL=(root) NOPASSWD: /usr/bin/systemctl restart $SERVICE, /usr/bin/systemctl status $SERVICE
EOF
visudo -cf "$SUDOERS.tmp"
chmod 440 "$SUDOERS.tmp"
mv "$SUDOERS.tmp" "$SUDOERS"

log "systemd unit"
install -m 644 "$APP_DIR/deploy/$SERVICE.service" "/etc/systemd/system/$SERVICE.service"
systemctl daemon-reload
systemctl enable "$SERVICE"
systemctl restart "$SERVICE"

log "Nightly backup timer"
install -d -m 755 /etc/scholarshipcoach
install -m 644 "$APP_DIR/deploy/$SERVICE-backup.service" "/etc/systemd/system/$SERVICE-backup.service"
install -m 644 "$APP_DIR/deploy/$SERVICE-backup.timer" "/etc/systemd/system/$SERVICE-backup.timer"
systemctl daemon-reload
systemctl enable --now "$SERVICE-backup.timer"

cat <<EOF

Done. The clone has only what git tracks. Copy these from the dev PC as $APP_USER,
then run: sudo systemctl restart $SERVICE
  .streamlit/secrets.toml            (family PIN)
  data/private/                      (coach.db, student profiles; or restore from backup)
  data/processed/*.parquet           (catalog snapshots)
  data/processed/embeddings/         (optional; rebuilt on demand)
  data/processed/win_model/          (optional)
Check: curl -fsS http://127.0.0.1:8501/_stcore/health

Backups (and deploy/update.sh) fail until /etc/scholarshipcoach/restic.env exists:
  RESTIC_REPOSITORY, RESTIC_PASSWORD, B2_ACCOUNT_ID, B2_ACCOUNT_KEY
  chown $APP_USER:$APP_USER /etc/scholarshipcoach/restic.env
  chmod 600 /etc/scholarshipcoach/restic.env
then, as $APP_USER: set -a; . /etc/scholarshipcoach/restic.env; restic init
EOF
