#!/usr/bin/env bash
set -euo pipefail

# #299 replay contract. Fill MODEL and SERVER before execution; this script intentionally
# refuses to guess model/runtime settings so an unattributed number cannot recur.
: "${MODEL:?set MODEL to the exact local GGUF/native checkpoint}"
: "${SERVER:?set SERVER to the exact binary or container entrypoint}"
: "${GPU_UUID:?set GPU_UUID to a free V100-labelled card UUID}"
: "${PORT:=18099}"
: "${CONTEXT:=32768}"
: "${BATCH:=8}"
: "${N_PREDICT:=256}"
: "${PROMPT_FILE:?set PROMPT_FILE to a fixed prompt artifact}"

printf 'model=%s\nserver=%s\ngpu_uuid=%s\ncontext=%s\nbatch=%s\nn_predict=%s\nprompt_file=%s\n' \
  "$MODEL" "$SERVER" "$GPU_UUID" "$CONTEXT" "$BATCH" "$N_PREDICT" "$PROMPT_FILE"
printf '%s\n' 'This is a contract-only replay stub; use the runtime-specific launcher after provenance is complete.'
