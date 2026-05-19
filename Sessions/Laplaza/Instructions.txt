# Genetic algorithms for molecules (CAMLC 2024)

Tutorial notebook and environment for SMILES-based genetic algorithms with RDKit and [mol-ga](https://github.com/AustinT/mol_ga).

## Setup

From this directory:

```bash
conda env create -f environment.yaml
conda activate gm-tutorial
```

RDKit comes from conda-forge. `mol-ga` is installed from a vendored wheel in `packaging/vendor/` whose PyPI dependency pins are stripped so it does not pull a conflicting RDKit wheel (see `packaging/patch_mol_ga_wheel.py`).

## Run the notebook

```bash
python -m nbconvert --to notebook --execute gm.ipynb --output gm.executed.ipynb
```

Or open `gm.ipynb` in Jupyter/VS Code with the `gm-tutorial` kernel.

A static PDF export is included as `gm.pdf`.

## Files

| File | Purpose |
|------|---------|
| `gm.ipynb` | Main tutorial |
| `simple_ga.py` | Simple string-based GA helpers |
| `SMILES_ZINC_5000.csv` | Starting population for graph GA |
| `solubility_aqsoldb.tab` | Solubility data for ML fitness |
| `SA_Score/` | Synthetic accessibility scorer (RDKit contrib fallback) |
| `packaging/` | Patched `mol_ga` wheel for conda-compatible install |

## Updating mol-ga

```bash
pip download mol-ga==NEW_VERSION --no-deps -d packaging/vendor
# Update paths in packaging/patch_mol_ga_wheel.py
python packaging/patch_mol_ga_wheel.py
# Point environment.yaml pip entry at the new +conda wheel
```
