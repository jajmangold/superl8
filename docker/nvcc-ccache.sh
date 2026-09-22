#!/bin/bash
# ============================================================================
# Copyright (c) 2026, superl8 authors
# SPDX-License-Identifier: BSD-3-Clause
# ============================================================================
# superl8 nvcc -> ccache wrapper with torch dependency-flag translation.
#
# PyTorch's BuildExtension emits this nvcc dependency form (torch 2.10):
#
#   --generate-dependencies-with-compile --dependency-output <path>
#
# ccache does NOT recognize those flags (verified on the production image's
# ccache 4.9.1 and in ccache 4.11.1 upstream source), so it forwards them
# unchanged into its preprocessor pass:
#
#   nvcc ... --generate-dependencies-with-compile --dependency-output <p> -E ...
#
# nvcc refuses dependency generation together with -E ("'-MD' is not
# supported when '-E' is also specified"), the preprocessor exits non-zero,
# and ccache marks EVERY invocation "preprocessing failed" (uncacheable).
# Result: 0 hits / 149 cacheable misses and ~4,350 uncacheable calls per
# required-CI build (superl8#247).
#
# This wrapper rewrites that dependency pair into the classic nvcc form
# ccache understands, "-MD -MF <path>", which:
#   * still makes nvcc emit a ninja-compatible depfile at the SAME <path>
#     (native nvcc -MD -MF includes device/CUDA headers, so ninja header
#     invalidation keeps working), and
#   * lets ccache cache the object; on a hit ccache restores both the .o and
#     the .d (verified against ccache 4.9.1), so ninja's depfile=$out.d rule
#     is satisfied in both cache-miss and cache-hit builds.
#
# The ccache input hash is unaffected by the -MF/<path> argument (the object
# and dependency paths are not part of the hash), so the randomized per-build
# /tmp/tmp*.build-temp paths stay stable, safe keys.
#
# hash_dir (CWD hashing) is disabled for NVCC ONLY, scoped to this wrapper:
# torch runs ninja with cwd = the setuptools build-temp directory, which is a
# NEW RANDOM /tmp/tmp*.build-temp on every build. With ccache's default
# hash_dir=true the CWD is part of the key, so even byte-identical compiles
# from two clean checkouts never hit (0/26 warm). This is safe here because:
#   * the standard superl8 nvcc argv has NO debug-info flag (setup.py NVCC_FLAGS
#     carries no -g/-lineinfo; only FNI8_DEBUG adds them), so the object does
#     not embed the compilation CWD;
#   * source and include paths are absolute and stable (/workspace/...) across
#     all runner checkouts, so no other component varies;
#   * the only CWD-dependent quantity was the throwaway build-temp dir itself.
# See ccache(1) "Compiling in different directories": hash_dir may be disabled
# to get hits when compiling the same source in different directories if an
# incorrect CWD in debug info is acceptable (no debug info is emitted here).
# FNI8_DEBUG (-g -lineinfo) is the exception: it DOES embed the compilation
# CWD, so for those invocations the wrapper keeps the default hash_dir (the
# object is then keyed on the real build-temp CWD, and misses when it varies,
# exactly as it should for debug builds).
# The gcc/c++ compile path is NOT affected: ninja invokes it through the
# /usr/lib/ccache masquerade with -g in its flags, where hash_dir stays on.
#
# The wrapper is bind-mounted over the image's /usr/local/cuda/bin/nvcc by
# docker-compose.yml, so it is tracked in git and no Docker image rebuild is
# required. ccache is invoked as /usr/bin/ccache with the REAL nvcc binary
# (nvcc.real) as the cached compiler, so compiler_check hashes the real
# toolchain, not this script.

set -u

# Scope hash_dir=false to nvcc (see above) ONLY when no debug-info flag is
# present: -g/-lineinfo (FNI8_DEBUG adds both) embed the compilation CWD in
# the object, so those builds must keep the default hash_dir to stay correct.
# Standard superl8 builds carry no such flag, so the randomized torch build-temp
# CWD is excluded from the key and cross-checkout builds hit.
#
# Matching is EXACT-TOKEN on purpose: a prefix match like -g* would also match
# "-gencode" (used by every superl8 build) and wrongly keep hash_dir for release
# objects. nvcc does not combine short options, and torch emits these flags as
# separate argv tokens (setup.py FNI8_DEBUG appends ["-g", "-lineinfo", ...]),
# so the exact tokens cover every real invocation.
has_debug_flag=0
for arg in "$@"; do
  case "$arg" in
    -g|-G|-lineinfo|--generate-line-info|--device-debug|--debug)
      has_debug_flag=1
      ;;
  esac
done
if [ "$has_debug_flag" -eq 0 ]; then
  export CCACHE_NOHASHDIR=1
fi

translated=()
consume_as_mf=0
debug_print=0

for arg in "$@"; do
  if [ "$consume_as_mf" -eq 1 ]; then
    translated+=("-MF" "$arg")
    consume_as_mf=0
    continue
  fi
  case "$arg" in
    --superl8-print-translated)
      # Test-only: print the translated argv (one arg per line) and exit,
      # so the translation logic is unit-testable without invoking nvcc.
      debug_print=1
      ;;
    --generate-dependencies-with-compile)
      # nvcc treats this as -MD (compile and emit a depfile); ccache needs
      # the classic form so its preprocessor pass can strip it before -E.
      translated+=("-MD")
      ;;
    --dependency-output)
      consume_as_mf=1
      ;;
    --dependency-output=*)
      translated+=("-MF" "${arg#--dependency-output=}")
      ;;
    *)
      translated+=("$arg")
      ;;
  esac
done

# Fail closed: a dangling --dependency-output must never silently produce an
# object without its depfile (ninja's depfile=$out.d rule would break).
if [ "$consume_as_mf" -eq 1 ]; then
  echo "superl8 nvcc-ccache wrapper: --dependency-output is missing its path" >&2
  exit 2
fi

if [ "$debug_print" -eq 1 ]; then
  printf '%s\n' "${translated[@]}"
  exit 0
fi

exec /usr/bin/ccache /usr/local/cuda/bin/nvcc.real "${translated[@]}"
