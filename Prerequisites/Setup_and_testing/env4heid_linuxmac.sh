#!/usr/bin/env bash
set -euo pipefail

# ===== Configuration =====
ENV_NAME="chemtorch"
# =========================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Correctly resolve to the Heid folder
HEID_DIR="$(cd "${SCRIPT_DIR}/../Sessions/Heid" 2>/dev/null || cd "${SCRIPT_DIR}/../../Sessions/Heid" && pwd)"
CHEMTORCH_DIR="${HEID_DIR}/chemtorch"

echo "===================================================="
echo "Script directory: ${SCRIPT_DIR}"
echo "Heid directory:   ${HEID_DIR}"
echo "===================================================="

# 0) Fix the missing/broken chemtorch folder
if [ ! -d "${CHEMTORCH_DIR}" ] || [ -z "$(ls -A "${CHEMTORCH_DIR}" 2>/dev/null)" ] || [ ! -f "${CHEMTORCH_DIR}/setup.py" ] && [ ! -f "${CHEMTORCH_DIR}/pyproject.toml" ]; then
    echo "⚠️  'chemtorch' folder is empty, missing, or a broken submodule link."
    echo "--> Cloning a fresh copy of chemtorch directly into ${CHEMTORCH_DIR}..."
    
    # Remove the broken directory/submodule link safely
    rm -rf "${CHEMTORCH_DIR}"
    
    # Clone the repository. 
    # NOTE: If chemtorch is a private workshop repo, replace this URL with the actual git URL
    git clone https://github.com/heid-lab/ChemTorch.git "${CHEMTORCH_DIR}" #t clone https://github.com/YUR_ORGANIZATION_OR_USER/chemtorch.git "${CHEMTORCH_DIR}"
fi

# 1) Create the conda environment if it does not exist
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "--> Conda environment '${ENV_NAME}' already exists. Skipping creation."
else
    echo "--> Creating conda environment '${ENV_NAME}' from environment.yml..."
    cd "${HEID_DIR}"
    conda env create -f environment.yml
fi

# 2) Install Python packages inside the environment
echo "--> Installing core Python packages in '${ENV_NAME}'..."
conda run -n "${ENV_NAME}" pip install \
    rdkit \
    numpy==1.26.4 \
    scikit-learn \
    pandas \
    torch==2.10.0 \
    hydra-core \
    wandb \
    ipykernel \
    jupyter \
    matplotlib

echo "--> Installing PyTorch Geometric dependencies..."
conda run -n "${ENV_NAME}" pip install \
    torch_scatter \
    torch_sparse \
    torch_cluster \
    torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.10.0+cpu.html

# 3) Local editable chemtorch install
echo "--> Installing local chemtorch in editable mode..."
cd "${HEID_DIR}"
conda run -n "${ENV_NAME}" pip install -e chemtorch

# 4) Install chemprop
echo "--> Installing chemprop..."
conda run -n "${ENV_NAME}" pip install chemprop

echo "===================================================="
echo "Success! Environment '${ENV_NAME}' is ready."
echo "===================================================="
