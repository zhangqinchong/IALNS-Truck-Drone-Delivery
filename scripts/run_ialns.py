"""Portable entry point around the byte-for-byte frozen full IALNS."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "ialns.py"
EXPECTED_SHA = "33cae732436e27f2e125a4b852688a3264f8017f2ac4c96951d0a47651527cfc"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_engine():
    if digest(SOURCE) != EXPECTED_SHA:
        raise RuntimeError("Frozen source checksum changed; review before replication")
    spec = importlib.util.spec_from_file_location("frozen_ialns", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_problem(module, instance, scale=0.6, iterations=4500, overrides=None):
    settings = json.loads((ROOT / "settings" / "parameters.json").read_text(encoding="utf-8"))
    config = dict(settings["common_ialns_config"])
    config.update(overrides or {})
    config.update(max_iter=iterations, depot_end_id=-1, plot_route_map=False,
                  plot_objective_history=False, enable_early_stop=False)
    cfg = module.Config.from_dict(config)
    path = Path(instance)
    loader = module.load_customers_excel if path.suffix.lower() == ".xlsx" else module.load_customers_csv
    customers = loader(path, cfg)
    for customer in customers.values():
        customer.x *= scale
        customer.y *= scale
    for key in ("depot_start_x", "depot_start_y", "depot_end_x", "depot_end_y"):
        setattr(cfg, key, getattr(cfg, key) * scale)
    return module.ProblemData(customers=customers, cfg=cfg,
        depot_start=(cfg.depot_start_x, cfg.depot_start_y),
        depot_end=(cfg.depot_end_x, cfg.depot_end_y))


def solve(instance, seed, scale=0.6, iterations=4500, overrides=None):
    module = load_engine()
    data = build_problem(module, instance, scale, iterations, overrides)
    cpu, wall = time.process_time(), time.perf_counter()
    initial = module.create_initial_solution(data, random.Random(seed))
    initial_eval = module.evaluate_solution(data, initial)
    init_sec, init_cpu = time.perf_counter() - wall, time.process_time() - cpu
    if not initial_eval.feasible:
        raise RuntimeError(f"Infeasible initialization: {initial_eval.violations}")
    cpu, wall = time.process_time(), time.perf_counter()
    optimizer = module.ALNSOptimizer(data=data, seed=seed)
    if not data.U:
        ids = (1, 2, 3, 4, 5, 13, 14, 17, 18, 20, 21)
        optimizer.operator_funcs = [getattr(optimizer, f"op{i}") for i in ids]
        optimizer.operator_stats = [module.OperatorStat(name=f"OP{i}") for i in ids]
    setup_sec, setup_cpu = time.perf_counter() - wall, time.process_time() - cpu
    cpu, wall = time.process_time(), time.perf_counter()
    solution, evaluation = optimizer.run(init_solution=initial)
    runtime, cpu_sec = time.perf_counter() - wall, time.process_time() - cpu
    native_cost = evaluation.objective
    solution.normalize(module.route_positions(solution.truck_route))
    evaluation = module.evaluate_solution(data, solution)
    sorties = [s for u in data.U for s in solution.sorties_by_drone.get(u, [])]
    served = [c for c in solution.truck_route if c in data.customers]
    served += [c for s in sorties for c in s.customers]
    parts = [evaluation.c1, evaluation.c2, evaluation.c3, evaluation.c4, evaluation.c5]
    if (not evaluation.feasible or Counter(served) != Counter(data.C)
            or not math.isclose(native_cost, evaluation.objective, abs_tol=1e-7, rel_tol=0)
            or not math.isclose(sum(parts), evaluation.objective, abs_tol=1e-7, rel_tol=0)
            or len(optimizer.history_obj) != iterations):
        raise RuntimeError("Final feasibility, coverage, cost, or iteration validation failed")
    row = dict(instance=Path(instance).name, algorithm="IALNS-final-frozen-20261002", seed=seed,
        coordinate_scale=scale, customer_count=len(data.C), iterations=iterations,
        initial_objective=initial_eval.objective, best_cost=evaluation.objective,
        initial_feasible=initial_eval.feasible, best_feasible=evaluation.feasible,
        runtime=runtime, cpu_sec=cpu_sec, initialization_sec=init_sec, initialization_cpu_sec=init_cpu,
        optimizer_setup_sec=setup_sec, optimizer_setup_cpu_sec=setup_cpu,
        total_sec=init_sec + setup_sec + runtime, total_cpu_sec=init_cpu + setup_cpu + cpu_sec,
        fixed_cost=evaluation.c1, transport_cost=evaluation.c2, waiting_cost=evaluation.c3,
        tardiness_cost=evaluation.c4, carbon_cost=evaluation.c5,
        truck_distance=evaluation.truck_distance, drone_distance=evaluation.drone_distance,
        carbon_emissions_kg=evaluation.carbon_emissions,
        number_of_drones_used=sum(bool(solution.sorties_by_drone.get(u)) for u in data.U),
        number_of_sorties=len(sorties), drone_customer_count=sum(len(s.customers) for s in sorties),
        truck_customer_count=len(data.C) - sum(len(s.customers) for s in sorties),
        late_customers=evaluation.late_customers, total_tardiness=evaluation.total_tardiness,
        truck_route=solution.truck_route, drone_routes=[asdict(s) for s in sorties],
        algorithm_source_sha256=digest(SOURCE), input_sha256=digest(instance))
    payload = module.serialize_solution(solution, evaluation, data, seed=seed,
        runtime_sec=runtime, initial_evaluation=initial_eval)
    payload.update(row=row, config=asdict(data.cfg), objective_history=optimizer.history_obj,
        best_update_log=optimizer.best_update_log, operator_stats=[asdict(s) for s in optimizer.operator_stats])
    return payload


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=104729)
    parser.add_argument("--scale", type=float, default=0.6)
    parser.add_argument("--iterations", type=int, default=4500)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON_VALUE")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or not math.isfinite(args.scale) or args.scale <= 0:
        parser.error("iterations and scale must be positive")
    overrides = {}
    for value in args.set:
        key, raw = value.split("=", 1)
        overrides[key] = json.loads(raw)
    unknown = set(overrides) - set(asdict(load_engine().Config()))
    if unknown:
        parser.error(f"Unknown configuration keys: {sorted(unknown)}")
    output = args.output or ROOT / "run_outputs" / f"{args.instance.stem}_seed_{args.seed}.json"
    if output.exists():
        parser.error(f"Refusing to overwrite an existing result: {output}")
    payload = solve(args.instance, args.seed, args.scale, args.iterations, overrides)
    atomic_json(output, payload)
    print(json.dumps(dict(best_cost=payload["objective"], runtime=payload["runtime_sec"],
                         feasible=payload["feasible"], output=str(output)), ensure_ascii=False))


if __name__ == "__main__":
    main()
