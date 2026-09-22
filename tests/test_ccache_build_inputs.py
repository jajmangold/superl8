# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
"""CCache build-input regression tests for superl8#247.

PyTorch's BuildExtension (torch 2.10) emits ``--generate-dependencies-with-compile
--dependency-output <path>`` for nvcc. ccache does not recognize those flags, so
its nvcc preprocessor pass re-runs them against ``nvcc -E``, which nvcc refuses
("'-MD' is not supported when '-E' is also specified") -> every CUDA object is
counted "preprocessing failed" / uncacheable (0 hits on the shared /ccache).

``docker/nvcc-ccache.sh`` (bind-mounted read-only over the image's nvcc by
docker-compose.yml) rewrites the pair into the ccache-recognized ``-MD -MF <path>``
form while keeping the ninja depfile at the same path.

These tests pin the translation, the fail-closed behavior, the read-only compose
mount, and the end-to-end ccache hit/miss behavior. They are CPU-only (nvcc
compiles offline; no CUDA device is needed) but follow the repo convention of
running inside the GPU-enabled ``test`` service.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "docker" / "nvcc-ccache.sh"
COMPOSE = REPO_ROOT / "docker-compose.yml"

TORCH_DEPFLAGS = [
    "--generate-dependencies-with-compile",
    "--dependency-output",
    "/tmp/tmpXYZ.build-temp/csrc/kernel/attn_decode.o.d",
]
REST = [
    "-I/workspace/csrc/include",
    "-c",
    "-c",
    "/workspace/csrc/kernel/attn_decode.cu",
    "-o",
    "/tmp/tmpXYZ.build-temp/csrc/kernel/attn_decode.o",
    "-O3",
    "-std=c++17",
    "-gencode",
    "arch=compute_70,code=sm_70",
]


def _git(args):
    # -c safe.directory: the container runs as root over a host-owned checkout,
    # which git's dubious-ownership guard otherwise rejects.
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "-c", f"safe.directory={REPO_ROOT}", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _run_wrapper_print(args):
    """Invoke the wrapper in --superl8-print-translated mode; return (rc, lines, err)."""
    proc = subprocess.run(
        [str(WRAPPER), "--superl8-print-translated", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout.splitlines(), proc.stderr


@pytest.mark.regression
def test_compose_mounts_wrapper_readonly_and_tracked():
    """The exact wrapper mount is wired read-only in compose and tracked in git."""
    compose = COMPOSE.read_text(encoding="utf-8")
    assert "./docker/nvcc-ccache.sh:/usr/local/cuda/bin/nvcc:ro" in compose, (
        "docker-compose.yml must bind-mount the tracked nvcc->ccache wrapper "
        "read-only over /usr/local/cuda/bin/nvcc (superl8#247)"
    )
    assert WRAPPER.is_file(), "tracked wrapper file missing"
    assert WRAPPER.stat().st_mode & 0o111, "wrapper must be executable"
    proc = _git(["ls-files", "--", "docker/nvcc-ccache.sh"])
    if proc.returncode != 0:
        # git could not ANSWER (not: answered "untracked"). In a linked worktree
        # `.git` is a file pointing at <repo>/.git/worktrees/<name>, which lies
        # outside compose's `./:/workspace` mount, so git exits 128 inside the
        # container. Failing here reports "untracked" for a question never asked,
        # and a permanently-red test hides real regressions. Skip only on rc != 0;
        # an empty list with rc == 0 is a genuine failure and still asserts below.
        pytest.skip(f"git cannot read this checkout (rc={proc.returncode}): "
                    f"{proc.stderr.strip()[:80]}")
    tracked = proc.stdout.splitlines()
    assert tracked and tracked[0] == "docker/nvcc-ccache.sh", (
        "docker/nvcc-ccache.sh must be tracked (a clean checkout needs it)"
    )


@pytest.mark.regression
@pytest.mark.parametrize(
    ("args", "want"),
    [
        # Exact torch emission -> ccache-recognized -MD -MF at the SAME path.
        (
            TORCH_DEPFLAGS + REST,
            ["-MD", "-MF", "/tmp/tmpXYZ.build-temp/csrc/kernel/attn_decode.o.d", *REST],
        ),
        # Attached single-arg form.
        (
            ["--generate-dependencies-with-compile", "--dependency-output=/x/y.d", "-c", "a.cu"],
            ["-MD", "-MF", "/x/y.d", "-c", "a.cu"],
        ),
        # Attachment without the paired -MD is still translated.
        (["--dependency-output=/x/y.d", "-c", "a.cu"], ["-MF", "/x/y.d", "-c", "a.cu"]),
        # Order independence: --dependency-output may precede the generator flag.
        (
            ["--dependency-output", "/x/y.d", "--generate-dependencies-with-compile", "-c", "a.cu"],
            ["-MF", "/x/y.d", "-MD", "-c", "a.cu"],
        ),
    ],
)
def test_wrapper_translation(args, want):
    rc, lines, err = _run_wrapper_print(args)
    assert rc == 0, err
    assert lines == want


@pytest.mark.regression
def test_wrapper_preserves_args_with_spaces():
    """Arguments containing spaces survive translation byte-for-byte."""
    weird = ["/we ird/$dir/my file.cu", "-I/inc dir", "-o", "/out dir/a.o"]
    rc, lines, err = _run_wrapper_print(["--generate-dependencies-with-compile", *weird])
    assert rc == 0, err
    assert lines == ["-MD", *weird]


@pytest.mark.regression
def test_wrapper_dangling_dependency_output_fails_closed():
    """A trailing --dependency-output without a path must fail, not compile naked."""
    rc, lines, err = _run_wrapper_print(
        ["--generate-dependencies-with-compile", "--dependency-output"]
    )
    assert rc == 2
    assert "missing its path" in err
    assert lines == []


@pytest.mark.regression
def test_wrapper_fail_closed_takes_precedence_over_debug_print():
    rc, lines, _err = _run_wrapper_print(["--dependency-output"])
    assert rc == 2
    assert lines == []


@pytest.mark.regression
def test_wrapper_debug_argv_keeps_hash_dir():
    """-g/-lineinfo (FNI8_DEBUG) must retain hash_dir; release argv must not."""
    if not (WRAPPER.is_file() and Path("/usr/bin/ccache").exists()):
        pytest.skip("wrapper or ccache not present (tests run in the superl8-dev container)")

    import tempfile

    work = Path(tempfile.mkdtemp(prefix="superl8-ccdbg-"))
    src = work / "mini.cu"
    src.write_text("__global__ void k(float* x) { *x = 1.0f; }\n", encoding="utf-8")

    def run_series(debug_flags, label):
        cache = Path(tempfile.mkdtemp(prefix=f"superl8-cc-{label}-"))
        env = dict(os.environ, CCACHE_DIR=str(cache), CCACHE_MAXSIZE="1G")
        subprocess.run(["ccache", "--zero-stats"], env=env, check=True, capture_output=True)

        def go(cwd_dir):
            cwd_dir.mkdir(parents=True, exist_ok=True)
            obj = cwd_dir / "a.o"
            cmd = [
                str(WRAPPER),
                "--generate-dependencies-with-compile",
                "--dependency-output",
                str(obj.with_suffix(".o.d")),
                "-I/usr/local/cuda/include",
                "-c",
                "-c",
                str(src),
                "-o",
                str(obj),
                "-O3",
                "-std=c++17",
                "-gencode",
                "arch=compute_70,code=sm_70",
                *debug_flags,
            ]
            return subprocess.run(
                cmd, cwd=str(cwd_dir), env=env, check=False, capture_output=True, text=True
            )

        def hits_misses():
            r = subprocess.run(
                ["ccache", "-sv", "-v"], env=env, check=False, capture_output=True, text=True
            )
            txt = r.stdout + r.stderr
            h = re.search(r"Hits:\s+(\d+)", txt)
            m = re.search(r"Misses:\s+(\d+)", txt)
            return (int(h.group(1)) if h else -1), (int(m.group(1)) if m else -1)

        assert go(work / f"{label}Cold.build-temp").returncode == 0
        assert go(work / f"{label}Warm.build-temp").returncode == 0
        return hits_misses()

    # Debug argv: CWD is part of the key, so a different build-temp dir misses.
    hits, misses = run_series(["-g", "-lineinfo"], "dbg")
    assert hits == 0 and misses == 2, (
        f"debug argv must keep hash_dir (no cross-CWD hit): hits={hits} misses={misses}"
    )

    # Release argv: CWD excluded, so a different build-temp dir hits.
    hits, misses = run_series([], "rel")
    assert hits == 1 and misses == 1, (
        f"release argv must drop hash_dir (cross-CWD hit): hits={hits} misses={misses}"
    )


@pytest.mark.regression
def test_wrapper_end_to_end_ccache_hit_miss():
    """Real wrapper, real nvcc: cross-CWD hit, depfile restore, source/header/flag misses."""
    if not (WRAPPER.is_file() and Path("/usr/bin/ccache").exists()):
        pytest.skip("wrapper or ccache not present (tests run in the superl8-dev container)")

    import tempfile

    cache = Path(tempfile.mkdtemp(prefix="superl8-ccache-"))
    work = Path(tempfile.mkdtemp(prefix="superl8-ccwork-"))

    # Local probe header: FNI8_PROBE lives here, referenced by the source, so
    # a header-content mutation MUST invalidate the cache (issue #247 accepts
    # "headers change -> miss"). -DFNI8_PROBE on the command line overrides the
    # guarded default, so a semantic flag change also MISSES.
    probe_h = work / "probe.h"
    probe_h.write_text("#ifndef FNI8_PROBE\n#define FNI8_PROBE 1\n#endif\n", encoding="utf-8")
    src = work / "mini.cu"
    src.write_text(
        '#include "probe.h"\n__global__ void k(float* x) { *x = 1.0f * FNI8_PROBE; }\n',
        encoding="utf-8",
    )

    env = dict(os.environ, CCACHE_DIR=str(cache), CCACHE_MAXSIZE="1G")
    subprocess.run(["ccache", "--zero-stats"], env=env, check=True, capture_output=True)

    def compile_through_wrapper(cwd_dir, obj_name, source, define_probe=None):
        cwd_dir.mkdir(parents=True, exist_ok=True)
        obj = cwd_dir / obj_name
        dep = obj.with_suffix(".o.d")
        cmd = [
            str(WRAPPER),
            "--generate-dependencies-with-compile",
            "--dependency-output",
            str(dep),
            f"-I{work}",
            "-I/usr/local/cuda/include",
            "-c",
            "-c",
            str(source),
            "-o",
            str(obj),
            "-O3",
            "-std=c++17",
            "-gencode",
            "arch=compute_70,code=sm_70",
        ]
        if define_probe is not None:
            # Append after the depfile operand (inserting at index 3 would put
            # it between --dependency-output and its path, and the wrapper
            # would consume the macro as the -MF target).
            cmd.append(f"-DFNI8_PROBE={define_probe}")
        return subprocess.run(
            cmd, cwd=str(cwd_dir), env=env, check=False, capture_output=True, text=True
        )

    def stats():
        proc = subprocess.run(
            ["ccache", "-sv", "-v"], env=env, check=False, capture_output=True, text=True
        )
        txt = proc.stdout + proc.stderr

        def grab(label):
            m = re.search(rf"{label}:\s+(\d+)", txt)
            return int(m.group(1)) if m else None

        return {
            "cacheable": grab("Cacheable calls"),
            "hits": grab("Hits"),
            "misses": grab("Misses"),
            "pp_failed": grab("Preprocessing failed"),
        }

    # Cold: cwd = one (simulated) build-temp dir; cacheable miss, depfile emitted.
    cold_dir = work / "tmpCold123.build-temp"
    r = compile_through_wrapper(cold_dir, "a.o", src)
    assert r.returncode == 0, r.stderr
    assert (cold_dir / "a.o").is_file()
    assert (cold_dir / "a.o.d").is_file(), "depfile must be emitted"
    s1 = stats()
    assert s1["pp_failed"] == 0, f"preprocessing must not fail: {s1}"
    assert s1["cacheable"] == 1 and s1["misses"] == 1 and s1["hits"] == 0, s1

    # Warm from a DIFFERENT cwd/build-temp dir and output path -> hit, and both
    # the object AND the depfile are restored (ninja depfile=$out.d must hold).
    warm_dir = work / "tmpWarm456.build-temp"
    r2 = compile_through_wrapper(warm_dir, "a.o", src)
    assert r2.returncode == 0, r2.stderr
    assert (warm_dir / "a.o").is_file()
    assert (warm_dir / "a.o.d").is_file(), "ccache must restore the depfile on hit"
    s2 = stats()
    assert s2["cacheable"] == 2 and s2["hits"] == 1 and s2["misses"] == 1, s2

    # Source change -> real miss (no stale-object reuse).
    src.write_text(
        '#include "probe.h"\n'
        "__global__ void k(float* x) { *x = 2.0f * FNI8_PROBE; }\n"
        "__host__ int superl8_marker(void) { return 1; }\n",
        encoding="utf-8",
    )
    r3 = compile_through_wrapper(warm_dir, "a.o", src)
    assert r3.returncode == 0, r3.stderr
    s3 = stats()
    assert s3["cacheable"] == 3 and s3["hits"] == 1 and s3["misses"] == 2, s3

    # Referenced-macro / semantic flag change -> real miss.
    r4 = compile_through_wrapper(warm_dir, "a.o", src, define_probe=2)
    assert r4.returncode == 0, r4.stderr
    s4 = stats()
    assert s4["cacheable"] == 4 and s4["hits"] == 1 and s4["misses"] == 3, s4

    # Header content change -> real miss (acceptance: headers changing must miss).
    probe_h.write_text("#ifndef FNI8_PROBE\n#define FNI8_PROBE 9\n#endif\n", encoding="utf-8")
    r5 = compile_through_wrapper(warm_dir, "a.o", src)
    assert r5.returncode == 0, r5.stderr
    s5 = stats()
    assert s5["cacheable"] == 5 and s5["hits"] == 1 and s5["misses"] == 4, s5
