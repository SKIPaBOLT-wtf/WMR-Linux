#!/usr/bin/env bash
# One-time: build + install the sched_ext schedulers (scx_lavd + scx_loader + scxctl)
# from source. Kernel-agnostic: needs sched_ext only (CONFIG_SCHED_CLASS_EXT=y).
# Follows the official upstream INSTALL.md (Ubuntu path) — cargo-based, all Rust.
#
#   scripts/scx-setup.sh
#
# Model: your kernel default scheduler (BORE on the custom kernel; EEVDF on stock)
# stays in charge; scx_lavd is loaded ON-DEMAND for gaming (situational — can
# regress CPU-light titles; see KERNEL-BUILD-PLAN.md).
set -euo pipefail
RESEARCH=$HOME/g2-linux-research
SCX=$RESEARCH/src/scx

# tee everything (stdout+stderr) to a persistent log for post-mortem
LOG_DIR=${LOG_DIR:-$RESEARCH/logs}
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/scx-setup-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "#### scx-setup.sh   $(date)   log=$LOG"
trap 'echo "#### scx-setup.sh EXIT rc=$? at $(date)"' EXIT

[ -d /sys/kernel/sched_ext ] || { echo "ERR: running kernel lacks sched_ext" >&2; exit 1; }

echo "== deps (per upstream INSTALL.md / Ubuntu) =="
sudo apt install -y build-essential cmake cargo rustc clang llvm pkg-config \
                    libelf-dev libbpf-dev pahole protobuf-compiler libseccomp-dev

echo "== fetch scx =="
if [ -d "$SCX/.git" ]; then git -C "$SCX" pull --ff-only || true
else git clone https://github.com/sched-ext/scx.git "$SCX"; fi

# scx crates bump MSRV aggressively (e.g. sysinfo 0.39 needs rustc >= 1.95);
# apt's rustc on Ubuntu 26.04 is often 1.93. Auto-install rustup if missing.
NEED_MIN_MM=1.95
need_upgrade=1
if [ -f "$HOME/.cargo/env" ]; then . "$HOME/.cargo/env"; fi
RUSTC_VER=$(rustc --version 2>/dev/null | awk '{print $2}' || true)
if [ -n "$RUSTC_VER" ]; then
    RUSTC_MM=${RUSTC_VER%.*}
    awk -v v="$RUSTC_MM" -v m="$NEED_MIN_MM" 'BEGIN{
        split(v,a,"."); split(m,b,".");
        exit (a[1]>b[1] || (a[1]==b[1] && a[2]>=b[2]))?0:1
    }' && need_upgrade=0
fi
if [ "$need_upgrade" = 1 ]; then
    echo "== rustc ${RUSTC_VER:-missing} < $NEED_MIN_MM — installing rustup (user-local, no sudo) =="
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal
    . "$HOME/.cargo/env"
    rustup update stable
    RUSTC_VER=$(rustc --version | awk '{print $2}')
fi
echo "using rustc $RUSTC_VER"
echo "== cargo build --release (long; ~10 min first time) =="
cd "$SCX"
cargo build --release

echo "== install scheduler + tool binaries to /usr/local/bin =="
INSTALLED=()
for b in scx_lavd scx_bpfland scx_rusty scx_flash scx_loader scxctl scxtop; do
    if [ -x "target/release/$b" ]; then
        sudo install -m 755 "target/release/$b" "/usr/local/bin/$b"
        INSTALLED+=("$b")
    fi
done
echo "  installed: ${INSTALLED[*]:-(none?)}"

# scx_loader systemd unit if the repo ships one
for SVC in rust/scx_loader/scx_loader.service services/scx_loader.service; do
    if [ -f "$SCX/$SVC" ]; then
        sudo install -m 644 "$SCX/$SVC" /etc/systemd/system/scx_loader.service
        sudo systemctl daemon-reload
        echo "  installed unit: /etc/systemd/system/scx_loader.service"
        break
    fi
done

echo; echo "== verify =="
command -v scx_lavd scxctl >/dev/null && echo "  scx_lavd + scxctl on PATH" || echo "  WARN: missing binaries"
cat <<'EOF'

DONE. Use it:
  Quick test (foreground):   sudo scx_lavd               # Ctrl+C to stop; check /sys/kernel/sched_ext/state
  Via the loader (if installed): scxctl switch -s scx_lavd -m gaming  /  scxctl stop
  Persistent loader service: sudo systemctl enable --now scx_loader   (in-kernel default stays)

AUTO-SWITCH per game — add to ~/.config/gamemode.ini under [custom]:
  start=scxctl switch -s scx_lavd -m gaming
  end=scxctl stop
Launch via `gamemoderun mangohud %command%`; MangoHud's `gamemode` field confirms it's active.
Watch 1% lows — drop scx for any title it regresses.
EOF
