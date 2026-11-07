#!/usr/bin/env bash
# Run once by the workspace owner, in a terminal of their own Noppe session. Puts the
# demonstration's Python and the model in the shared folder, where participants can read them.
# Safe to run again, for example after changing requirements-cpu.txt.
#   bash ~/persona-drift-lens/deploy/noppe/conda/setup-shared.sh
# PVD_SHARED sets another location (for a test off Noppe).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
SHARED="${PVD_SHARED:-$HOME/shared/persona-drift-lens}"
MODEL="${PVD_MODEL:-Qwen/Qwen3-1.7B}"
ENV_DIR="${PVD_ENV_DIR:-$SHARED/env}"

mkdir -p "$SHARED"

if command -v mamba >/dev/null; then CONDA=mamba; else CONDA=conda; fi
if [ -x "$ENV_DIR/bin/python" ]; then
  echo "Updating the environment in $ENV_DIR"
  "$ENV_DIR/bin/python" -m pip install --quiet -r "$REPO/requirements-cpu.txt" \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
else
  echo "Creating the environment in $ENV_DIR (a few minutes)"
  "$CONDA" env create --yes -p "$ENV_DIR" -f "$HERE/environment.yml"
fi

echo "Downloading $MODEL to $SHARED/hf (about 2 GB, once)"
(cd "$REPO" && HF_HOME="$SHARED/hf" HF_HUB_OFFLINE=0 "$ENV_DIR/bin/python" -m app.llama_runner "$MODEL")

# Participants read these files as other users on some systems; let everyone read them.
chmod -R a+rX "$SHARED"

du -sh "$SHARED"
echo "Done. Start a new session to see the demonstration."
