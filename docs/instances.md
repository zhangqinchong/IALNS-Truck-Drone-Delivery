# Modified Benchmark Instances

## Published inputs

Base 101-series: C/R/RC, each with 5, 10, 15, 20, 50, and 100 customers.
Base 102-series: C/R/RC, each with 10, 20, 50, and 100 customers.
Exposure-ratio inputs: C101/R101/RC101 with 50 customers, each at ratios
0.0, 0.1, 0.2, 0.3, 0.4, and 0.5.
Only inputs used by the released final-version experiments are included.

Files contain the original depot and first n Solomon customer coordinates.
Apply the 0.6 factor once to both customers and depots in the experiment runner.
The `case_config_overrides` in `settings/parameters.json` describe the effective,
already-scaled archived depot configuration; do not multiply these values again.
The public runner instead reads the raw depot from the input before scaling.

## Columns

| Column | Definition |
| --- | --- |
| `id` | Depot 0; customers 1 through n; implicit return depot n+1 |
| `x`, `y` | Original spatial coordinates |
| `demand` | Parcel weight in kg |
| `can_drone` | Input eligibility flag; 1 iff base demand is at most 2.5 kg |
| `ready`, `due` | Earliest/latest service time in h; tardiness is penalized |
| `max_outside` | Maximum exposure duration in h; 100 is the ordinary-parcel sentinel |

Depot weight/eligibility/window/exposure cells are not customer constraints;
the input reader uses the depot row for its coordinates. The original depot due
value may retain Solomon units and is not a customer horizon for this model.

## Attribute rules

Weights: floor(0.50n) in Uniform(0.1,0.7) kg, floor(0.36n) in
Uniform(0.8,2.5) kg, and the remaining parcels in Uniform(2.6,10.0) kg.
Round all weights to one decimal. Base exposure-sensitive count is
n-floor(0.90n), with limits in Uniform(0.3,1.0) h, rounded to one decimal.
Other parcels use 100 h.

| n | Ready range (h) | Due upper limit (h) | Minimum width (h) |
| --- | --- | --- | --- |
| 5 | 0-0.5 | 5 | 1 |
| 10 | 0-1 | 6 | 1 |
| 15 | 0-1 | 7 | 1 |
| 20 | 0-1 | 8 | 1 |
| 30 | 0-2 | 9 | 2 |
| 50 | 0-2 | 10 | 2 |
| 75 | 0-3 | 12 | 2 |
| 100 | 0-3 | 15 | 2 |

For the archived 102 inputs, sample the ready time first, round it, then sample
due in [ready+minimum width, due upper limit], with a minimum-width guard.
Weight categories and exposure-sensitive IDs are independently shuffled in the
recorded draw order. Use Mulberry32 seeded by FNV-1a uint32 of
`20260929:{family_lowercase}:{n}`. The generator implements the original
JavaScript one-decimal rounding and unsigned integer operations.

```bash
python scripts/generate_instances.py --family c102 --sizes 10 20 50 100
python scripts/generate_instances.py --exposure-ratio
```

The generator writes CSV with the same customer values as the archived Excel
inputs; it does not reproduce Excel styling or ZIP-container bytes. All twelve
published 102 base inputs and eighteen exposure inputs are tested cellwise.
For fresh 101 attributes an explicit `--master-seed` is required. These are new
instances, not reconstructions of the unknown historical 101 generation seed.

Exposure-ratio experiments use round(50r) sensitive customers, nested selections,
and archived quantiles mapped to Uniform(0.1,0.3) h. Weight, eligibility, locations,
and service windows are held fixed. Ratio 0.1 is therefore not the ordinary
base input with Uniform(0.3,1.0) exposure limits.

Changing drone capacity does not change the fixed `can_drone` input flag: both
eligibility and the varied capacity constraint must hold.
