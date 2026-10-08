"""Resume a main, small, or sensitivity suite in a bounded parallel queue."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path

from run_ialns import ROOT, EXPECTED_SHA, atomic_json, digest, solve


def jobs_for(suite, settings):
    catalog = settings["input_catalog"]
    seeds = settings["solver_seeds_in_order"]
    if suite in ("main", "small"):
        cases = [c for c in catalog if "/base/" in c["input_file"] and
                 (c["customer_count"] in (10, 20, 50, 100) if suite == "main"
                  else c["series"] == 101 and c["customer_count"] in (5, 15))]
        for case in cases:
            for seed in seeds:
                yield dict(id=f"{Path(case['input_file']).stem}_seed_{seed}",
                           input_file=case["input_file"], seed=seed, overrides={})
    else:
        for spec in settings["sensitivity"]["analyses"]:
            for value in spec["values"]:
                for base in settings["sensitivity"]["base_instances"]:
                    filename = (f"{Path(base).stem}_exposure_ratio_{value:.1f}_tight.xlsx"
                                if spec["analysis"] == "exposure_ratio" else base)
                    folder = "exposure_ratio" if spec["analysis"] == "exposure_ratio" else "base"
                    for seed in seeds:
                        yield dict(id=f"{spec['analysis']}_{value}_{Path(base).stem}_seed_{seed}",
                            input_file=f"instances/{folder}/{filename}", seed=seed,
                            overrides={spec["field"]: value} if spec["field"] else {},
                            analysis=spec["analysis"], parameter_value=value,
                            baseline=spec["baseline"], base_instance=base,
                            can_reuse=spec["analysis"] != "exposure_ratio" and value == spec["baseline"])


def signature(job, iterations, settings):
    value = dict(input_sha256=digest(ROOT / job["input_file"]), seed=job["seed"],
                 iterations=iterations, overrides=job["overrides"],
                 algorithm_source_sha256=EXPECTED_SHA, scale=settings["coordinate_scale"],
                 config=settings["common_ialns_config"])
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_compatible(path, expected):
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("request_sha256") != expected or not payload.get("feasible"):
        raise RuntimeError(f"Existing result is incompatible; use another output directory: {path}")
    return payload


def initialize(queue):
    cpu = queue.get()
    if cpu is None:
        return
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), 1 << cpu):
            raise OSError(ctypes.get_last_error(), "Cannot set worker affinity")
    elif hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu})
    else:
        raise RuntimeError("Explicit CPU affinity is unsupported on this platform")


def execute(job, iterations, expected, output_dir):
    payload = solve(ROOT / job["input_file"], job["seed"], iterations=iterations,
                    overrides=job["overrides"])
    payload.update(request_sha256=expected, experiment=job)
    atomic_json(output_dir / f"{job['id']}.json", payload)
    return job["id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("main", "small", "sensitivity"), required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=4500)
    parser.add_argument("--limit", type=int, help="Smoke-test only the first N jobs")
    parser.add_argument("--cpu-ids", help="One logical CPU ID per worker; verify physical-core topology first")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "run_outputs")
    args = parser.parse_args()
    if args.workers < 1 or args.iterations < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers, iterations, and limit must be positive")
    ids = [int(v) for v in args.cpu_ids.split(",")] if args.cpu_ids else [None] * args.workers
    if (len(ids) != args.workers or (args.cpu_ids and len(set(ids)) != len(ids))
            or any(i is not None and (i < 0 or i >= os.cpu_count()) for i in ids)):
        parser.error("Provide distinct valid CPU IDs, one per worker")
    settings = json.loads((ROOT / "settings" / "parameters.json").read_text(encoding="utf-8"))
    for key, value in settings["execution"]["thread_environment"].items():
        os.environ[key] = value
    jobs = list(jobs_for(args.suite, settings))
    suite_count = len(jobs)
    if args.limit is not None:
        jobs = jobs[:args.limit]
    folder = args.output_dir / args.suite
    folder.mkdir(parents=True, exist_ok=True)
    todo, reused = [], 0
    for job in jobs:
        expected = signature(job, args.iterations, settings)
        target = folder / f"{job['id']}.json"
        if read_compatible(target, expected) is not None:
            continue
        if job.get("can_reuse"):
            reference = args.output_dir / "main" / f"{Path(job['base_instance']).stem}_seed_{job['seed']}.json"
            base_job = dict(input_file=job["input_file"], seed=job["seed"], overrides={})
            base_sig = signature(base_job, args.iterations, settings)
            payload = read_compatible(reference, base_sig)
            if payload is not None:
                payload.update(request_sha256=expected, experiment=job, result_source="reused_main_baseline")
                atomic_json(target, payload)
                reused += 1
                continue
        todo.append((job, expected))
    queue = multiprocessing.get_context("spawn").Queue()
    for cpu in ids:
        queue.put(cpu)
    print(f"{args.suite}: {len(jobs)} jobs; {len(todo)} pending; {reused} reused")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize, initargs=(queue,)) as pool:
        futures = {pool.submit(execute, job, args.iterations, expected, folder): job for job, expected in todo}
        for future in as_completed(futures):
            print(f"Completed: {future.result()}", flush=True)
    rows = []
    for job in jobs:
        payload = read_compatible(folder / f"{job['id']}.json", signature(job, args.iterations, settings))
        row = dict(payload["row"])
        row.update({key: job[key] for key in ("analysis", "parameter_value") if key in job})
        row["result_source"] = payload.get("result_source", "new_run")
        for key, value in row.items():
            if isinstance(value, (dict, list)):
                row[key] = json.dumps(value, ensure_ascii=False)
        rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (folder / "runs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(folder / "completed.json", dict(suite=args.suite, successful=len(rows),
        total_suite_jobs=suite_count, full_suite_completed=len(rows) == suite_count))


if __name__ == "__main__":
    main()
