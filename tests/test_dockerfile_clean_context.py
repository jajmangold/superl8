# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""Narrow clean-build-context regression for #243.

``docker/Dockerfile`` unconditionally COPYs ``docker/wheels/`` into the image,
but that directory is gitignored (it holds a private ``flash_attn_v100`` wheel),
so a clean checkout had no such directory and BuildKit aborted before compiling:

    failed to solve: failed to calculate checksum ... "/docker/wheels": not found

The wheel is optional (external-baseline tests soft-skip), so the build must
succeed on a clean clone. Contract: the Dockerfile keeps the optional-wheel
COPY; the clean tracked tree always contains a placeholder under docker/wheels/;
and real wheel binaries stay ignored while the placeholder stays tracked.
"""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WHEELS = REPO_ROOT / "docker" / "wheels"
PLACEHOLDER = WHEELS / ".gitkeep"


def _git(args):
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True)


@pytest.mark.regression
def test_clean_checkout_builds_without_local_wheel():
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY docker/wheels/ /tmp/wheels/" in dockerfile  # optional-wheel contract intact
    assert "if ls /tmp/wheels/flash_attn_v100*.whl" in dockerfile  # installs when present

    # Git-INDEPENDENT checks first: a git failure below must not mask a deleted
    # .gitkeep or a broken .gitignore pair. Only tracked-ness needs git.
    assert PLACEHOLDER.is_file(), "docker/wheels/.gitkeep missing on disk"
    ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "docker/wheels/*" in ignore and "!docker/wheels/.gitkeep" in ignore, (
        ".gitignore must keep docker/wheels/* ignored but .gitkeep tracked (superl8#243)"
    )

    proc = _git(["ls-files", "--", "docker/wheels/"])
    if proc.returncode != 0:
        # See test_ccache_build_inputs: rc != 0 means git could not answer (linked
        # worktree gitdir outside the container mount), not that the path is
        # untracked. Skip only on rc != 0; rc == 0 with no output still fails.
        pytest.skip(f"git cannot read this checkout (rc={proc.returncode}): "
                    f"{proc.stderr.strip()[:80]}")
    tracked = [p for p in proc.stdout.splitlines() if p]
    assert tracked, (
        "docker/wheels/ is COPYed unconditionally but has no tracked content, so "
        "a clean checkout cannot build (superl8#243); commit docker/wheels/.gitkeep."
    )
    assert "docker/wheels/.gitkeep" in tracked
    assert _git(["check-ignore", "-q", "--", "docker/wheels/flash_attn_v100-x.whl"]).returncode == 0
    assert _git(["check-ignore", "-q", "--", "docker/wheels/.gitkeep"]).returncode == 1
