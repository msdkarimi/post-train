#!/usr/bin/env bash
# run.sh - run bootstrap.sh first; only if it exits 0 (success), run train.sh.
# Exit code 0 = success, anything else = failure (standard shell convention).
#
# Usage:  bash run.sh                      # bootstrap, then train
#         WITH_VLLM=1 bash run.sh          # extra env vars are passed through
set -uo pipefail
cd "$(dirname "$0")"

if bash bootstrap.sh; then
  echo ">>> bootstrap succeeded (exit 0) - starting train.sh"
  bash train.sh
  rc=$?
  echo ">>> train.sh finished with exit code $rc"
  exit "$rc"
else
  rc=$?
  echo ">>> bootstrap FAILED (exit $rc) - NOT starting train.sh. Fix the error above, then re-run."
  exit "$rc"
fi
