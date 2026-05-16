# Introduction to ML and data science in chemistry

## Learning outcomes
By the end of this session, you should be able to:

1. **Explain** the role and key applications of machine learning in chemistry.
2. **Assess** the quality and limitations of chemical datasets, including bias and FAIR principles.
3. **Interpret** chemical data types and distributions, and their impact on model selection and performance.
4. **Distinguish** between regression and classification approaches for chemical problems.
5. **Apply** core data preprocessing (e.g., scaling, feature selection, dataset splitting) steps in a simple ML workflow.
6. **Construct and evaluate** basic linear regression and classification models using chemical data.
7. **Critically evaluate** the reliablity and applicability of machine learning models in chemical research.

## Datasets
There are four example datasets which are in the `data` folder, two of which we will look at in the class, the other two you can have a play with afterwards!

### Ni-catalyzed homo-Diels–Alder
This data can be found in `data/homo_diels_alder.csv`. Reference: https://pubs.acs.org/doi/10.1021/jacs.5c09948.

This data models the yield of a Ni-catalyzed homo-Diels–Alder cycloaddition -  the output in the CSV is `mvk_hda_yield`

### Amide coupling

This data can be found in `data/amide_coupling.csv`. For information on where this data came from, this is the link to the original paper: https://doi.org/10.1073/pnas.2118451119.

This data models rates of amide couplings - the output in the CSV is `ln rate`.

### BONUS: Enantiodivergent Pd-catalyzed C-C coupling

This data can be found in `data/biscoe_kraken_data`. Reference: https://doi.org/10.1126/science.aat2299.

This data models the enantiospecificity of a Pd-catalyzed cross-coupling -  the output in the CSV is `ddG_rem` or `ddG`, depending if you want to use the data before or after the removal described in the paper.

### BONUS: Hydrogen bond donor catalysts

This data can be found in `data/hydrogen_bond_donors.csv`. Reference: https://pubs.acs.org/doi/10.1021/jacs.0c06905.

This data models enantioselectivity data for a series of bifunctional hydrogen bond donor catalysts - the output in the CSV is `∆∆G‡ (kcal/mol)`.