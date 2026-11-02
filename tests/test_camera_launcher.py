"""Invalid camera configuration must fail before contacting a robot."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("args, hint", [
    ([], "--host"),
    (["--host", "example-robot"], "--listen"),
    (["--host"], "missing value"),
    (["--mode", "--status"], "missing value"),
])
def test_missing_camera_settings_fail_before_ssh(tmp_path, args, hint):
    marker = tmp_path / "ssh-called"
    ssh = tmp_path / "ssh"
    ssh.write_text('#!/bin/sh\ntouch "$SSH_TEST_MARKER"\nexit 99\n')
    ssh.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}",
               SSH_TEST_MARKER=str(marker))
    for key in ("G1_HOST", "ORIN_LISTEN", "CAMERA_MODE"):
        env.pop(key, None)
    script = Path(__file__).resolve().parents[1] / "scripts/g1/start_orin_camera.sh"
    result = subprocess.run(["bash", str(script), *args], env=env,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 2
    assert hint in result.stdout + result.stderr
    assert not marker.exists()
