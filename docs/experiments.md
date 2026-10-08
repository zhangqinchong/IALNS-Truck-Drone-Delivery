# Experimental Protocol and Archived Results

## Final code and settings

Frozen source SHA-256:
`33cae732436e27f2e125a4b852688a3264f8017f2ac4c96951d0a47651527cfc`.
All published runs used 4,500 iterations and coordinate scale 0.6. The source
file is preserved byte-for-byte under the portable name `src/ialns.py`.
The publication wrapper changes input paths and output handling, not algorithm
operators, random draws, acceptance, reconstruction, polishing, or evaluation.

## Jobs

| Suite | Inputs | Seeds | Records |
| --- | --- | --- | --- |
| `main` | 101/102 x C/R/RC x 10/20/50/100 | 10 | 240 |
| `small` | 101 x C/R/RC x 5/15 | 10 | 60 |
| `sensitivity` | C101/R101/RC101-50, five analyses | 10 | 810 |

Sensitivity levels are 0-5 drones; endurance 0.2/0.3/0.4/0.5/0.6 h;
capacity 1.0/1.5/2.0/2.5/3.0 kg; road factor 1.1/1.3/1.5/1.7/1.9;
and exposure-sensitive ratios 0.0-0.5 in steps of 0.1. Four parameter baselines
reuse the same 30 main-suite records, for 120 reused rows and 690 new sensitivity
runs. All exposure-ratio levels were solved separately. The public batch runner
reuses main reference results when present and compatible, or solves them if
no compatible local reference exists.

At zero available drones, the applicable operator pool is
OP1/2/3/4/5/13/14/17/18/20/21, as in the archived sensitivity experiment.
All other configurations retain the complete 21-operator pool.

## Timing

`runtime` is wall time inside `optimizer.run`: it includes plateau and final
polishing, but excludes input loading, initialization, and optimizer construction.
`total_sec` includes initialization, optimizer construction, and `optimizer.run`;
it excludes importing, input I/O, post-run validation, and export. Process CPU
times are recorded separately. Supplementary 5/15 records did not separately
archive optimizer-construction time; their missing total times remain blank.
Do not pool missing values with zeros or directly treat their native runtimes as
the main batch's total times.

Original main/sensitivity environment: AMD Ryzen 9 5900HX, 8 physical cores,
16 logical processors, 16 GB installed RAM, 64-bit Windows 11, Python 3.12.14,
openpyxl 3.1.5. Eight concurrent jobs were affined to one logical processor on
each of eight physical cores, with numerical-library thread counts set to one.
Supplementary small runs used eight workers without archived per-worker affinity.

Fixed seeds do not guarantee matching wall times on other machines. They also
do not guarantee identical solutions under every Python version/platform; use
the tested Python version for the closest replication and report deviations.

## Result fields and validation

CSV files retain instance, distribution, series, seed, iterations, initial and
best objective, wall/CPU time, fixed/travel/waiting/tardiness/carbon costs,
truck/drone distances, emissions, used drones, sorties, lateness, and route data.
Compressed JSONL files store feasible final routes, time schedules,
full configurations, objective histories, and best-update logs.

`scripts/validate_results.py` re-evaluates all 1,110 archived route records using
the released evaluator and their configurations, checks exactly-once customer
coverage, and reconciles all five cost components. `tests/` additionally checks
input hashes, regenerated attributes, and a short full-iteration seeded run.
Recorded best costs are heuristic incumbent values, not certified optima.

The source is marked `-text` in `.gitattributes` so Git does not normalize its
line endings; its frozen checksum stays valid after cloning on Windows or Linux.

The final benchmarking inputs overlap with development experiments; the
experiments do not constitute a strictly held-out tuning/evaluation split.
