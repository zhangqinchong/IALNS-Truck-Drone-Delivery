# IALNS for Truck-Drone Collaborative Delivery

Reproducibility materials for an improved adaptive large neighborhood search
(IALNS) for one truck collaborating with multiple drones. The model includes
heterogeneous parcel weights, truck-only customers, multi-customer drone sorties,
customer time windows, exposure-sensitive parcels, and carbon-emission costs.

This repository provides the final IALNS implementation, modified Solomon
instances, instance-generation programs, parameter settings, fixed random seeds,
and experimental results.

[Chinese introduction](README.zh-CN.md) | [Experimental protocol](docs/experiments.md)
| [Data and generation rules](docs/instances.md)

## Contents

| Directory | Contents |
| --- | --- |
| `src/ialns.py` | Unmodified frozen IALNS implementation |
| `instances/base/` | 30 actual base Excel inputs used in the final experiments |
| `instances/exposure_ratio/` | 18 nested exposure-ratio Excel inputs |
| `instances/coordinates/` | Original, unscaled coordinates for generation |
| `settings/` | Full parameter values, ten solver seeds, input checksums, generation metadata |
| `scripts/` | Portable single-run, batch, generation, and result-validation tools |
| `results/` | Archived IALNS costs, times, routes, schedules, and convergence histories |
| `tests/` | Input, generator, configuration, and short solver regression tests |

The file hashes and dataset counts are recorded in `settings/release_manifest.json`.

## Quick Start

Python 3.10 or newer is required; the archived runs used Python 3.12.14.
From the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/run_ialns.py --instance instances/base/c101改10.xlsx --seed 104729
python -m unittest discover -s tests -v
```

The default run uses 4,500 iterations, all final IALNS components, and coordinate
scale **0.6**. Coordinates in the supplied Excel files are not pre-scaled.
Customer and depot coordinates are scaled once in memory by the public runner.
Output includes five cost components, runtime, routes, schedules, feasibility,
and iteration-cost history. Run results are written under `run_outputs/`.

Run all 240 main jobs (six families, four sizes, ten fixed seeds):

```bash
python scripts/run_experiments.py --suite main --workers 8
```

Batch execution is resumable: existing successful job files are skipped only when
their instance checksum, parameters, seed, iteration count, and code checksum agree.
An eight-worker queue is not itself a guarantee of eight distinct physical cores;
optional `--cpu-ids` pins each worker to explicitly selected logical CPUs.

## Model and Algorithm

The frozen implementation uses an initial-solution portfolio, 21 composite
operators, adaptive operator weighting, simulated-annealing acceptance, a tabu
list, stagnation-adaptive reconstruction and intensification, and plateau/final
polishing. The class name `ALNSOptimizer` inside the source is a historical name
for this full IALNS implementation.

Truck travel distances are Euclidean distances multiplied by the road factor;
drone distances are Euclidean. A drone may be launched after truck arrival while
the truck waits for a customer's earliest service time. Exposure is evaluated at
drone customer service start. All constraints and cost details are implemented in
the shared full-solution evaluator in `src/ialns.py`.

| Baseline parameter | Value |
| --- | --- |
| Available drones / payload / endurance | 4 / 2.5 kg / 0.5 h |
| Truck / drone speed | 30 / 60 km/h |
| Coordinate scale / road factor | 0.6 / 1.5 |
| Truck / used-drone fixed cost | 100 / 20 CNY |
| Truck / drone travel cost | 90 / 18 CNY/h |
| Waiting / tardiness cost | 18 / 40 CNY/h |
| Carbon price | 0.2 CNY/kg |
| Truck / drone emissions | 0.2155 / 0.01378 kg/km |
| Iterations / reconstruction stagnation / plateau polish | 4500 / 220 / 900 |

The complete settings, including all algorithm parameters, are in
`settings/parameters.json`. The compatibility field `M` in this heuristic's
configuration is not a MILP big-M derivation and is not used to claim exactness.

## Reproducibility Notes

The fixed solver seeds, in order, are:

```text
104729, 130363, 155921, 181081, 206369,
231779, 257053, 282427, 307831, 333287
```

The package preserves 240 main runs, 60 supplementary 5/15-customer runs, and
810 sensitivity records. Sensitivity reference rows reused from the main batch
are explicitly identified; they are not 810 new independent runs.

The 102-series attribute generator reproduces the archived customer attributes.
The historical **101-series attribute-generation seed was not archived**.
Exact replication of those instances therefore uses the supplied fixed Excel
files and SHA-256 hashes, not a falsely reconstructed seed. Solver seeds and
attribute-generation seeds are separate quantities.

For the main and sensitivity batches, the original workstation was an AMD Ryzen
9 5900HX (8 physical cores, 16 logical processors), 16 GB RAM, 64-bit Windows 11.
Eight workers used one logical processor from each physical core. Archived wall
times are machine- and load-dependent; costs are not global-optimality claims.

## Data Attribution

Customer locations originate from the Solomon C101/R101/RC101 and
C102/R102/RC102 VRPTW benchmarks. Parcel attributes and customer time windows
were regenerated for this truck-drone model; these are **modified inputs**, not
the original Solomon VRPTW instances or their original objective values.

Source: [SINTEF Solomon benchmark](https://www.sintef.no/projectweb/top/vrptw/100-customers/).

Reference: Solomon, M. M. (1987). Algorithms for the vehicle routing and
scheduling problems with time window constraints. *Operations Research*,
35(2), 254-265. https://doi.org/10.1287/opre.35.2.254

No license is assigned here to third-party benchmark material. Refer to the
original sources for applicable terms. A software/data reuse license should be
chosen by the repository owner; public availability alone is not a license.
