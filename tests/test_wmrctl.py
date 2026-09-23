# SPDX-License-Identifier: MIT
import copy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("wmrctl", ROOT / "package/wmrctl.py")
wmrctl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wmrctl)


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads((ROOT / "profiles/hp-reverb-g2-nvidia-ubuntu.json").read_text())

    def test_reference_is_valid_but_never_applied(self):
        result = wmrctl.validate_profile(self.profile)
        self.assertTrue(result["valid"])
        self.assertFalse(result["applied"])

    def test_rejects_private_fields_and_unqualified_switches(self):
        candidates = []
        for section, key, value in [("headset", "serial", "private"), ("environment", "TOKEN", "secret"),
                                    ("environment", "G2_PREDICT_WITH_VIT_BIAS", "true")]:
            p = copy.deepcopy(self.profile)
            p[section][key] = value
            candidates.append(p)
        for key, value in [("automatic_system_changes", True), ("qualification", "stable-qualified")]:
            p = copy.deepcopy(self.profile)
            p[key] = value
            candidates.append(p)
        for candidate in candidates:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                wmrctl.validate_profile(candidate)

    def test_rejects_camera_mismatch_and_bad_hash(self):
        for key, value in [("backend_sha256", "latest"), ("backend_source", "main")]:
            p = copy.deepcopy(self.profile)
            p["runtime"][key] = value
            with self.assertRaises(ValueError):
                wmrctl.validate_profile(p)
        self.profile["headset"]["slam_cameras"] = 4
        with self.assertRaises(ValueError):
            wmrctl.validate_profile(self.profile)

    def test_rejects_wrong_types_without_unhashable_errors(self):
        for section, key, value in [(None, "schema_version", True),
                                    (None, "schema_version", 1.0),
                                    (None, "qualification", []),
                                    ("headset", "family", {"private": "value"}),
                                    ("headset", "family", "   "),
                                    ("environment", "WMR_SLAM", [])]:
            p = copy.deepcopy(self.profile)
            target = p if section is None else p[section]
            target[key] = value
            with self.subTest(section=section, key=key, value=value), self.assertRaises(ValueError):
                wmrctl.validate_profile(p)

    def test_deeply_nested_profile_fails_without_path_or_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "private-profile.json"
            source.write_text("[" * 10000 + "0" + "]" * 10000)
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = wmrctl.main(["profile", str(source)])
            self.assertEqual(result, 2)
            self.assertFalse(json.loads(stdout.getvalue())["ok"])
            self.assertNotIn(tmp, stdout.getvalue())
            self.assertEqual(stderr.getvalue(), "")

    def test_doctor_has_no_user_or_path_fields_and_no_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(wmrctl.os.environ, {"XDG_DATA_HOME": tmp}), \
                 mock.patch.object(wmrctl.shutil, "which", return_value=None), \
                 mock.patch.object(wmrctl.platform, "freedesktop_os_release", return_value={"ID": "test", "PRIVATE": "secret"}):
                before = list(Path(tmp).rglob("*"))
                report = wmrctl.doctor()
                self.assertEqual(before, list(Path(tmp).rglob("*")))
                encoded = json.dumps(report)
                self.assertNotIn(tmp, encoded)
                self.assertNotIn("secret", encoded)
                self.assertEqual(report["os"], {"ID": "test"})
                self.assertFalse(report["uploads"])

    def test_doctor_does_not_echo_gpu_error_output(self):
        result = mock.Mock(returncode=1, stdout="private-device-id", stderr="private-home-path")
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.dict(wmrctl.os.environ, {"XDG_DATA_HOME": tmp}), \
             mock.patch.object(wmrctl.shutil, "which", return_value="nvidia-smi"), \
             mock.patch.object(wmrctl.subprocess, "run", return_value=result) as run:
            report = wmrctl.doctor()
            encoded = json.dumps(report)
            self.assertNotIn("private-device-id", encoded)
            self.assertNotIn("private-home-path", encoded)
            self.assertEqual(report["gpu"], "NVIDIA query unavailable")
            self.assertEqual(run.call_args.args[0],
                             ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])


if __name__ == "__main__":
    unittest.main()
