"""Utility helpers for plotting reaction-path analyses."""

import numpy as np
import py3Dmol
from ase import Atoms
from ase.constraints import FixInternals
from ase.io import read
from ase.optimize import BFGS
from IPython.display import HTML
from rdkit import Chem
from rdkit.Chem import Draw, rdDetermineBonds
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import SimilarityMaps
from sella import IRC, Sella

EV_TO_KCAL_MOL = 23.06054783061903


def ev_to_kcal_mol(values):
    """Convert energies from eV to kcal/mol."""
    return np.asarray(values, dtype=float) * EV_TO_KCAL_MOL


def rdkit_mol_to_ase_atoms(mol, conf_id=0):
    """Convert an RDKit molecule conformer into an ASE Atoms object."""
    conf = mol.GetConformer(conf_id)
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = np.asarray(conf.GetPositions(), dtype=float)
    return Atoms(symbols=symbols, positions=positions)


def draw_similarity_map_png_from_xyz(
    xyz_path,
    weights,
    *,
    charge=0,
    width=900,
    height=500,
):
    """Render a similarity map from XYZ + atom weights and return it as PNG."""
    # Convert the XYZ to an RDKit Mol object
    mol = Chem.MolFromXYZFile(xyz_path)
    rdDetermineBonds.DetermineBonds(mol, charge=charge)

    # Draw the similarity map
    d = Draw.MolDraw2DCairo(width, height)
    AllChem.Compute2DCoords(mol)

    for at, sd in zip(mol.GetAtoms(), weights):
        at.SetProp("atomNote", f"{sd:+.3f}")

    SimilarityMaps.GetSimilarityMapFromWeights(
        mol,
        list(weights),
        draw2d=d,
    )

    return d.GetDrawingText()


def relaxed_2d_scan(
    *,
    seed_atoms,
    x_vals,
    y_vals,
    x_bond,
    y_bond,
    calculator,
    charge,
    spin,
    fmax=0.05,
    steps=200,
    col_label="x",
    row_label="y",
):
    """Run a warm-started relaxed 2D scan over two constrained bond distances."""
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    x_i, x_j = x_bond
    y_i, y_j = y_bond

    e_grid = np.empty((len(y_vals), len(x_vals)), dtype=float)
    optimized_grid = [[None for _ in range(len(x_vals))] for _ in range(len(y_vals))]
    prior_structures = []

    seed_x = seed_atoms.get_distance(x_i, x_j)
    seed_y = seed_atoms.get_distance(y_i, y_j)

    # Visit grid points as a flood fill outward from the cell nearest the seed
    cells = [(r, c) for r in range(len(y_vals)) for c in range(len(x_vals))]
    cells.sort(
        key=lambda rc: (x_vals[rc[1]] - seed_x) ** 2 + (y_vals[rc[0]] - seed_y) ** 2
    )

    # Already-relaxed grid points, used as warm-start templates.
    total = len(cells)
    for done, (row_idx, col_idx) in enumerate(cells, start=1):
        x_val = x_vals[col_idx]
        y_val = y_vals[row_idx]

        if prior_structures:
            _, _, template = min(
                prior_structures,
                key=lambda p: (p[0] - x_val) ** 2 + (p[1] - y_val) ** 2,
            )
        else:
            template = seed_atoms  # reactant geometry seeds the first point

        atoms = template.copy()
        atoms.info.update({"charge": charge, "spin": spin})
        atoms.calc = calculator
        # Move both bonds onto their target lengths before relaxing.
        atoms.set_distance(x_i, x_j, x_val, fix=0.5)
        atoms.set_distance(y_i, y_j, y_val, fix=0.5)
        atoms.set_constraint(
            FixInternals(bonds=[[float(x_val), [x_i, x_j]], [float(y_val), [y_i, y_j]]])
        )
        converged = BFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)

        energy = atoms.get_potential_energy()
        e_grid[row_idx, col_idx] = energy
        optimized_grid[row_idx][col_idx] = atoms
        prior_structures.append((x_val, y_val, atoms))

        status = "optimized" if converged else "NOT converged"
        energy_kcal = energy * EV_TO_KCAL_MOL
        print(
            f"[{done}/{total}] {status} {col_label}={x_val:.3f} "
            f"{row_label}={y_val:.3f} -> E = {energy_kcal:.3f} kcal/mol "
            f"({col_label}/{row_label} scan)"
        )

    e_grid -= e_grid.min()

    return e_grid, optimized_grid, prior_structures


def draw_pes_meshgrid(
    fig,
    ax,
    cf_vals,
    ccl_vals,
    e_grid,
    *,
    cmap="viridis",
    vmax_cap=3.0,
    levels_count=25,
    colorbar_label="Relative energy (kcal/mol)",
    convert_from_ev=True,
):
    """Draw a 2D PES contour map on bond-distance axes and attach a colorbar."""
    e_grid_plot = np.asarray(e_grid, dtype=float)
    vmax_use = float(vmax_cap)
    if convert_from_ev:
        e_grid_plot = ev_to_kcal_mol(e_grid_plot)
        vmax_use *= EV_TO_KCAL_MOL

    c_fg, c_clg = np.meshgrid(cf_vals, ccl_vals)
    vmax = min(float(np.max(e_grid_plot)), vmax_use)
    levels = np.linspace(0.0, vmax, int(levels_count))

    cs = ax.contourf(c_fg, c_clg, e_grid_plot, levels=levels, cmap=cmap)
    ax.contour(
        c_fg,
        c_clg,
        e_grid_plot,
        colors="k",
        linewidths=0.5,
        levels=levels,
    )
    fig.colorbar(cs, ax=ax, label=colorbar_label)

    return c_fg, c_clg, levels


def reoptimize_ts(
    start_atoms,
    calculator,
    traj_name,
    label,
    *,
    charge,
    spin,
    fmax=0.02,
):
    """Reoptimize a TS guess with Sella and report the final energy."""
    atoms = start_atoms.copy()
    atoms.info.update({"charge": charge, "spin": spin})
    atoms.calc = calculator
    opt = Sella(atoms, trajectory=traj_name)
    opt.run(fmax=fmax)
    energy = atoms.get_potential_energy()
    print(f"{label}: {energy * EV_TO_KCAL_MOL:.6f} kcal/mol")
    return atoms, energy


def ordered_irc_path(
    ts_atoms,
    prefix,
    charge,
    spin,
    dx=0.1,
    fmax=0.01,
    steps=300,
    keep_going=False,
    log_to_notebook=True,
):
    """Run forward and reverse IRC from a TS and return a stitched path."""
    calc_use = ts_atoms.calc

    for direction in ("forward", "reverse"):
        run_atoms = ts_atoms.copy()
        run_atoms.calc = calc_use
        logfile_target = "-" if log_to_notebook else f"{prefix}_{direction}.log"
        print(f"Running {prefix} ({direction}) with logfile={logfile_target}")
        IRC(
            run_atoms,
            trajectory=f"{prefix}_{direction}.traj",
            logfile=logfile_target,
            dx=dx,
            eta=1e-4,
            gamma=0.4,
            keep_going=keep_going,
        ).run(fmax=fmax, steps=steps, direction=direction)

    fwd_images = read(f"{prefix}_forward.traj", index=":")
    rev_images = read(f"{prefix}_reverse.traj", index=":")
    return list(reversed(rev_images)) + fwd_images[1:]


def atoms_to_xyz(atoms):
    """Convert an ASE Atoms object to an XYZ string for py3Dmol."""
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    return f"{len(atoms)}\n\n" + "\n".join(
        f"{sym} {x} {y} {z}" for sym, (x, y, z) in zip(symbols, positions, strict=True)
    )


def snapshot_row(atoms_list, labels=None, *, width=250, height=250):
    """Render a horizontal row of 3D structure snapshots with optional labels."""
    html_columns = []
    for i, atoms in enumerate(atoms_list):
        view = py3Dmol.view(width=width, height=height)
        view.addModel(atoms_to_xyz(atoms), "xyz")
        view.setStyle({"stick": {"radius": 0.2}, "sphere": {"radius": 0.6}})
        view.zoomTo()

        label_text = labels[i] if labels else f"Frame {i}"
        html_columns.append(f"""
<div style='display:flex;flex-direction:column;align-items:center;margin-right:10px'>
    <div style='font-weight:bold;margin-bottom:4px;text-align:center'>{label_text}</div>
    {view._make_html()}
</div>""")

    return HTML(f"<div style='display:flex;gap:10px'>{''.join(html_columns)}</div>")
