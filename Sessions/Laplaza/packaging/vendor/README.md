# Vendored mol_ga wheel

`mol_ga` on PyPI depends on `rdkit`, which would install a second RDKit and break the
conda-forge build. The wheel `mol_ga-0.2.1+conda-py3-none-any.whl` has runtime
`Requires-Dist` entries removed (numpy, rdkit, joblib); those come from `environment.yaml`.

To bump the mol_ga version:

```bash
pip download mol-ga==NEW_VERSION --no-deps -d packaging/vendor
# Update SRC/OUT paths in packaging/patch_mol_ga_wheel.py
python packaging/patch_mol_ga_wheel.py
# Point environment.yaml pip entry at the new +conda wheel
```
