"""Check all released routes with the frozen IALNS model evaluator."""

from collections import Counter
import gzip
import json
import math

from run_ialns import ROOT, load_engine


def validate(payload, module):
    row = payload["row"]
    cfg = module.Config.from_dict(payload["config"])
    path = ROOT / payload["input_file"]
    customers = module.load_customers_excel(path, cfg)
    scale = float(row["coordinate_scale"])
    for customer in customers.values():
        customer.x *= scale
        customer.y *= scale
    # Archived depot config is already scaled; the input loader overwrites only
    # the start coordinates because the archived return-depot ID is explicit.
    cfg.depot_start_x *= scale
    cfg.depot_start_y *= scale
    data = module.ProblemData(customers=customers, cfg=cfg,
        depot_start=(cfg.depot_start_x, cfg.depot_start_y),
        depot_end=(cfg.depot_end_x, cfg.depot_end_y))
    sorties = {int(u): [module.Sortie(drone=int(u), index=s["index"], launch=s["launch"],
               recovery=s["recovery"], customers=list(s["customers"])) for s in group]
               for u, group in payload["sorties"].items()}
    solution = module.Solution(truck_route=list(payload["truck_route"]), sorties_by_drone=sorties)
    evaluation = module.evaluate_solution(data, solution)
    served = [c for c in solution.truck_route if c in customers]
    served += [c for group in sorties.values() for s in group for c in s.customers]
    if not evaluation.feasible or Counter(served) != Counter(data.C):
        raise ValueError(f"Invalid route: {row['instance']} / {row['seed']}: {evaluation.violations}")
    if not math.isclose(evaluation.objective, float(row["best_cost"]), abs_tol=1e-7, rel_tol=0):
        raise ValueError(f"Cost mismatch: {row['instance']} / {row['seed']}")
    for name, value in zip(("fixed_cost", "transport_cost", "waiting_cost", "tardiness_cost", "carbon_cost"),
                           (evaluation.c1, evaluation.c2, evaluation.c3, evaluation.c4, evaluation.c5)):
        if not math.isclose(value, float(row[name]), abs_tol=1e-7, rel_tol=0):
            raise ValueError(f"Component mismatch: {name}")


def main():
    module = load_engine()
    counts = {}
    for path in sorted((ROOT / "results").glob("*.jsonl.gz")):
        count = 0
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                validate(json.loads(line), module)
                count += 1
        counts[path.name] = count
    if sum(counts.values()) != 1110:
        raise ValueError(f"Expected 1110 records, found {counts}")
    print(json.dumps(dict(passed=True, route_records=counts)))


if __name__ == "__main__":
    main()
