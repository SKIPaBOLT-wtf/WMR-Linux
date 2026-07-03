#!/usr/bin/env bash
# One-time setup for the FrameTiming sidecar (core/frametiming.py).
#
# The sidecar needs the `openvr` Python binding, but the system python is
# PEP-668 (externally managed), so it lives in a repo-local venv under var/
# (runtime state, gitignored). core/perfmon.py launches the sidecar with this
# venv's interpreter and skips the channel with a loud message if it's missing.
# No sudo needed. Re-running is safe (venv is reused).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv var/venv
var/venv/bin/pip install --quiet openvr==2.12.1401
var/venv/bin/python -c "import openvr; print('perfmon venv ready: openvr', openvr.IVRCompositor_Version)"
