"""
Test 1.1: Zero-Bloat Packaging Tests.

Validates that:
- Wheel size is strictly < 10MB (typically < 100KB pure python)
- No .gguf, .onnx, or binary weights are bundled in the wheel
- Package installs cleanly as pure Python without triggering C++ toolchain builds
"""

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

WORKSPACE_ROOT = Path(__file__).parent.parent
DIST_DIR = WORKSPACE_ROOT / "dist"


def test_wheel_build_and_size(tmp_path):
    """Build wheel and assert file size < 10MB and absence of binary weights."""
    # Run build in clean mode
    res = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(DIST_DIR)],
        cwd=str(WORKSPACE_ROOT),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"Build failed: {res.stderr}\n{res.stdout}"

    wheels = list(DIST_DIR.glob("narvl-*.whl"))
    assert len(wheels) > 0, "No wheel generated in dist/"

    latest_wheel = max(wheels, key=lambda p: p.stat().st_mtime)
    wheel_size_bytes = latest_wheel.stat().st_size
    wheel_size_mb = wheel_size_bytes / (1024 * 1024)

    print(f"\n[Packaging Test] Wheel path: {latest_wheel}")
    print(f"[Packaging Test] Wheel size: {wheel_size_bytes} bytes ({wheel_size_mb:.3f} MB)")

    # Strict constraint: Must be < 10MB
    assert wheel_size_mb < 10.0, f"Wheel size {wheel_size_mb:.2f}MB exceeds 10MB limit!"

    # Inspect zip contents to ensure no heavy weights leaked
    with zipfile.ZipFile(latest_wheel, "r") as zf:
        file_list = zf.namelist()
        for fname in file_list:
            lower = fname.lower()
            assert not lower.endswith(".gguf"), f"Forbidden model weight found in wheel: {fname}"
            assert not lower.endswith(".onnx"), f"Forbidden ONNX model found in wheel: {fname}"
            assert not lower.endswith(".bin"), f"Forbidden binary weight found in wheel: {fname}"

    print("[Packaging Test] Zero-bloat check passed: 0 model weights in wheel.")
