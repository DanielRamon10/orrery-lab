# Notebooks

Three notebooks, each telling a story the scripts do not. The scripts produce figures;
these show the reasoning, the intermediate quantities, and the checks along the way.

They are committed **with their outputs**, so GitHub renders them without anyone needing
to run anything.

| | What it covers | Needs network |
|---|---|---|
| [**1 · Where is Mars tonight?**](01-solving-keplers-equation.ipynb) | From a date to a 3D position. Orbital elements, Kepler's equation and why it has no closed-form solution, Newton–Raphson converging in three iterations, the perifocal frame, the three rotations. Validated against conservation laws and then against published values. | no |
| [**2 · Why the integrator matters more than its accuracy**](02-why-integrators-matter.ipynb) | The two-body problem gives way to N-body. Four integrators measured side by side, the symplectic crossover at 544 orbits, convergence orders recovered from the data, time-reversibility, and a diagnostic that separates real planetary perturbation from discretisation error. | no |
| [**3 · What the data does not say**](03-what-the-data-does-not-say.ipynb) | Four ways to be confidently wrong: an R² of 0.99 hiding 21% errors, a pattern-search that finds resonances in anything, a classifier scoring 99.6% by reading the answer key, and a correlation that reverses sign because of how the sample was collected. | last two sections only |

## Running them

```bash
pip install -e ".[viz,science,dev,notebooks]"
jupyter lab notebooks/
```

Notebooks 1 and 2 run entirely offline. Notebook 3's last two sections need catalogue
snapshots and say so if they are missing:

```bash
python scripts/fetch_exoplanets.py --koi
python scripts/fetch_gaia.py
```

## A note on what is in them

Every mistake described in notebook 3 was made while building this project, and each was
caught by something in the repository — a test that failed, a figure that looked wrong, a
CI run on the first push. They are reproduced rather than summarised, because the
reproduction is the part worth having.
