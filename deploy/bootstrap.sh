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
PUSH_URL="${PUSH_URL:-git@github.com:daedwards06/ScholarshipCoach.git}"
APP_DIR=/srv/scholarshipcoach
APP_USER=coach
SERVICE=scholarshipcoach
SYNC_BRANCH=server
DEPLOY_KEY=/home/$APP_USER/.ssh/scholarshipcoach_deploy

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

log "Catalog edits commit to the $SYNC_BRANCH branch, pushed with a deploy key"
git_app() { as_coach git -C "$APP_DIR" "$@"; }
if ! git_app show-ref --verify --quiet "refs/heads/$SYNC_BRANCH"; then
    if git_app ls-remote --exit-code --heads origin "$SYNC_BRANCH" >/dev/null; then
        git_app switch -c "$SYNC_BRANCH" "origin/$SYNC_BRANCH"
    else
        git_app switch -c "$SYNC_BRANCH"
    fi
fi
git_app config user.name "ScholarshipCoach server"
git_app config user.email "coach@scholarshipcoach.invalid"
# Fetches stay on anonymous HTTPS; only pushes use SSH and the deploy key.
git_app config remote.origin.pushurl "$PUSH_URL"
git_app config core.sshCommand "ssh -i $DEPLOY_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
if [[ ! -f $DEPLOY_KEY ]]; then
    as_coach install -d -m 700 "$(dirname "$DEPLOY_KEY")"
    as_coach ssh-keygen -q -t ed25519 -N "" -C "scholarshipcoach server catalog sync" -f "$DEPLOY_KEY"
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

log "Timers: nightly backup, monthly re-verification, nightly catalog sync"
install -d -m 755 /etc/scholarshipcoach
for job in backup verify catalog-sync; do
    install -m 644 "$APP_DIR/deploy/$SERVICE-$job.service" "/etc/systemd/system/$SERVICE-$job.service"
    install -m 644 "$APP_DIR/deploy/$SERVICE-$job.timer" "/etc/systemd/system/$SERVICE-$job.timer"
done
systemctl daemon-reload
for job in backup verify catalog-sync; do
    systemctl enable --now "$SERVICE-$job.timer"
done

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

Catalog sync (deploy/catalog_sync.sh) cannot push until this public key is a
deploy key on the GitHub repo (Settings -> Deploy keys -> Add deploy key, tick
"Allow write access"). It is scoped to this one repo and held only by $APP_USER;
do not put a personal access token on this server.
$(cat "$DEPLOY_KEY.pub")
Check, as $APP_USER: ssh -i $DEPLOY_KEY -o IdentitiesOnly=yes -T git@github.com
EOF
