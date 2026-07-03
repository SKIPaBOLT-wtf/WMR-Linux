"""GPU + CPU performance tuning for VR (O-5 GPU clocks, O-4 governor)."""
import subprocess
from pathlib import Path

SAVED_GOV = Path("/tmp/g2_saved_cpu_governor")


def run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=8, check=False)
        return r.returncode == 0, r.stdout + r.stderr
    except Exception as e:
        return False, str(e)


def set_persistence(on: bool):
    return run(["sudo", "-n", "nvidia-smi", "-pm", "1" if on else "0"])


def set_clock_lock(min_mhz: int = 2820, max_mhz: int = 3105):
    """Lock the graphics clock to the P0 range. RTX 4080: 2820 MHz floor (P0),
    3105 MHz boost ceiling — keeping the floor at P0 avoids first-heavy-frame
    boost latency. (Was 2400, which sat below P0.)"""
    return run(["sudo", "-n", "nvidia-smi", "-lgc", f"{min_mhz},{max_mhz}"])


def reset_clocks():
    return run(["sudo", "-n", "nvidia-smi", "-rgc"])


def set_mem_clock_lock(min_mhz: int = 11201, max_mhz: int = 11201):
    """Lock the memory clock at the top point. Ada exposes 5 points
    (405/810/5001/10801/11201); pinning forbids memclk pstate excursions between
    light frames, whose transitions show up as frame-spike tails."""
    return run(["sudo", "-n", "nvidia-smi", "-lmc", f"{min_mhz},{max_mhz}"])


def reset_mem_clocks():
    return run(["sudo", "-n", "nvidia-smi", "-rmc"])


def set_power_limit(watts: int = 320):
    return run(["sudo", "-n", "nvidia-smi", "-pl", str(watts)])


def _cpu_governor() -> str:
    try:
        return Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    except Exception:
        return ""


def set_cpu_governor(gov: str):
    """Set the CPU frequency governor on all cores (needs root). VR wants
    'performance' — the default 'powersave' adds frame-time variance.
    Verified by sysfs readback: Ubuntu's cpupower wrapper exits 0 WITHOUT
    acting when no linux-tools build matches the running kernel (custom
    kernels) — scripts/setup-system.sh repairs that."""
    ok, out = run(["sudo", "-n", "cpupower", "frequency-set", "-g", gov])
    if _cpu_governor() != gov:
        print(f"[gpu] governor NOT set to {gov} "
              "(cpupower wrapper broken? run scripts/setup-system.sh)")
        return False, out
    return True, out


def _deep_cstates_disabled(max_latency_us: int) -> bool:
    """True iff every cpu0 idle state at/above the latency ceiling is off.
    Readback is the only trustworthy check — see set_cpu_governor."""
    states = Path("/sys/devices/system/cpu/cpu0/cpuidle")
    try:
        for st in states.iterdir():
            lat = int((st / "latency").read_text())
            if lat >= max_latency_us and (st / "disable").read_text().strip() != "1":
                return False
        return True
    except OSError:
        return False


def set_cstate_ceiling(max_latency_us: int = 200):
    """Disable CPU idle states with exit latency >= max_latency_us. Between
    11 ms frames cores idle long enough to enter deep states (i9-12900K:
    C6=220, C8=280, C10=680 us exit) and every 1 kHz pose-loop tick, USB IRQ,
    and compositor wake then pays the exit latency — a frame-tail mechanism
    no other lever touches. 200 keeps C1E (2 us), so idle power cost is
    modest. Grants + the cpupower wrapper repair: scripts/setup-system.sh."""
    _, out = run(["sudo", "-n", "cpupower", "idle-set", "-D", str(max_latency_us)])
    applied = _deep_cstates_disabled(max_latency_us)
    if not applied:
        print("[gpu] C-state ceiling NOT active "
              "(grant or linux-tools missing — run scripts/setup-system.sh)")
    return applied, out


def reset_cstates():
    return run(["sudo", "-n", "cpupower", "idle-set", "-E"])


def apply_vr_optimizations():
    """All-in performance setup for VR: persistence, P0 GPU clocks + power cap,
    the CPU performance governor (original saved for restore), and the deep
    C-state ceiling."""
    cur = _cpu_governor()
    if cur and cur != "performance":
        SAVED_GOV.write_text(cur)
    set_persistence(True)
    set_clock_lock(2820, 3105)
    set_mem_clock_lock(11201, 11201)
    set_power_limit(320)
    set_cpu_governor("performance")
    set_cstate_ceiling(200)
    return True


def revert_optimizations():
    set_persistence(False)
    reset_clocks()
    reset_mem_clocks()
    reset_cstates()
    gov = SAVED_GOV.read_text().strip() if SAVED_GOV.exists() else "powersave"
    set_cpu_governor(gov)
    if SAVED_GOV.exists():
        SAVED_GOV.unlink()
    return True
