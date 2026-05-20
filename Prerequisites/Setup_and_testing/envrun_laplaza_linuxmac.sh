#!/usr/bin/env bash
set -euo pipefail

# ===== Configuration =====
ENV_NAME="gm-tutorial"
# =========================

# Directory of this script (Prerequisites/Setup_and_testing/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Laplaza tutorial directory: go up twice, then Sessions/Laplaza
LAPLAZA_DIR="$(cd "${SCRIPT_DIR}/../../Sessions/Laplaza" && pwd)"

echo "Script directory: ${SCRIPT_DIR}"
echo "Laplaza directory: ${LAPLAZA_DIR}"

# 1) Create the conda environment if it does not exist
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Conda environment '${ENV_NAME}' already exists. Skipping creation."
else
    echo "Creating conda environment '${ENV_NAME}' from environment.yaml..."
    (cd "${LAPLAZA_DIR}" && conda env create -f environment.yaml)
fi

# 2) Execute the notebook using the environment
echo "Running gm.ipynb using conda environment '${ENV_NAME}'..."

conda run -n "${ENV_NAME}" python -m nbconvert \
    --to notebook \
    --execute "${LAPLAZA_DIR}/gm.ipynb" \
    --output "${LAPLAZA_DIR}/gm.executed.ipynb"

echo "Done. Executed notebook saved as gm.executed.ipynb in:"
echo "  ${LAPLAZA_DIR}"

