To setup your environment:

```
conda env create -f environment.yml  
conda activate chemtorch
pip install rdkit numpy==1.26.4 scikit-learn pandas torch==2.10.0 hydra-core wandb ipykernel jupyter matplotlib
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.10.0+cpu.html
pip install -e chemtorch
pip install chemprop
```

Then, launch `Workshop_ChemTorch_Demo_Jupyter.ipynb` as jupyter notebook or in VS Code.
