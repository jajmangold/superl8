# Qwen3.6/Qwen3.8 performance provenance — fni8#299

Status: provenance locked, 2026-08-15. The claimed `45.8 tok/s` is **not a reproducible
local measurement**; the verified historical local Qwen3.6 result is 21.3 steady decode
tok/s / 7.2 end-to-end tok/s.

## Evidence classification

| Row | Status | What the artifact actually proves |
|---|---|---|
| `45.8 tok/s` baseline | **unproven reference** | Appears twice in `qwen38-tq3/llama.cpp-tq3/artifacts/perf_m3/results{,_retry}.md` only as `Baseline reference`; there is no command, host/GPU, commit, checksum, run output, or raw timing attached. |
| 47.21 tok/s MTP | **external hardware** | `/srv/nvme-data/qwen38-tq3/llama.cpp-tq3/artifacts/perf_p0_baseline_3090/production_server_probes_20260610.md`: RTX 3090 24 GB, Qwen3.6-27B-MTP-TQ3_4S, `--spec-type draft-mtp`, `n-max 2`, `p-min 1.0`; not a fleet result. |
| 53.17 tok/s MTP | **external hardware** | Same RTX 3090 probe, 2,210-token prompt and 64-token generation; also not transferable to CMP/V100-labelled cards. |
| 21.3 tok/s steady / 7.2 tok/s E2E | **verified local fleet artifact** | Qwen3.6 Q3_K_S, GPU 11 V100-labelled CMP fleet card, graphs on, spec off, prompt 256/new 128, peak 14.33 GiB; exact fni8/fni8-serve commits captured. The checkpoint path is now absent, so this is provenance-locked but not replayable yet. |
| 38.64 tok/s MTP | **local artifact, incomplete provenance** | `artifacts/perf_m3/results_retry.md`: three runs 38.76/38.69/38.47, 259/280 accepted drafts (92.5%). It is stored on this machine and restored the local server, but the artifact omits GPU UUID/label, power state, exact launch command, and raw server log. |
| 128.3 prompt / 7.76 decode tok/s | **local fni8 control** | Current Qwen3.8 baked-image qualification recorded in WORKSTREAMS and fni8-serve#412/#295 evidence; exact 2k, K8V8, graphs/eager qualification regime. This is Qwen3.8 and is not a Qwen3.6 45.8 comparison. |
| 127.1 prompt / 7.36 decode tok/s | **fresh local fni8 control** | Qwen3.8, GPU 14 V100-labelled card, K8V8, prompt 512, chunked prefill 256, graphs off; captured with the current `fni8-final-qual` image. |

## Research findings

- The local TurboQuant fork report confirms MTP can be high-acceptance, but its M3
  experiment measured 38.64 tok/s and rejected the proposed verify fusion. It does not
  establish the 45.8 reference.
- The local Qwen3.6 TQ3 GGUF used by the old report is not present as a directly runnable
  GGUF in the current NVMe model tree. The 24 TB archive contains the Qwen3.6 HF staging
  files and fni8-native weight artifacts, but these are not automatically equivalent to
  the old llama.cpp TQ3 server binary/model pair.
- Current community research is consistent with separating the levers: qengine uses
  compact Q8/DP4A GEMMs, split-K attention, and pipelined cross-GPU transfers; 1Cat-vLLM
  uses SM70-specific Flash-V100 attention, FP8 KV, and MTP4; neither community number is
  a valid CMP result without an identical local reproduction.

## Primary sources inspected

- Local: `qwen38-tq3/llama.cpp-tq3/artifacts/perf_m3/results_retry.md`
- Local: `qwen38-tq3/llama.cpp-tq3/artifacts/perf_p0_baseline_3090/production_server_probes_20260610.md`
- Community: https://github.com/Haru-neo/qengine
- Community: https://github.com/1CatAI/1Cat-vLLM/blob/main/RELEASE.md
- Community: https://github.com/1CatAI/1Cat-vLLM/blob/main/README.md
- Upstream issue context: https://github.com/ggml-org/llama.cpp/issues/19858

## Required next capture

The 45.8 row is rejected as a local baseline until the exact Qwen3.6 GGUF/runtime pair is
found or recreated and run on a free V100-labelled card. The historical local Qwen3.6
artifact is 21.3 steady / 7.2 E2E; the current Qwen3.8 control is 127.1 prompt / 7.36 decode
under the fresh captured command. These are separate model/runtime rows.
