from __future__ import annotations

# IALNS variant with arrival-based drone launch timing. At a truck-served
# customer, a drone may launch after the truck arrives while the truck waits
# for the customer's ready time. The validated baseline remains unchanged.

import argparse
import csv
import heapq
import itertools
import json
import math
import random
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple


@dataclass(slots=True)
class Customer:
    cid: int
    x: float
    y: float
    demand: float
    can_drone: bool
    ready: float
    due: float
    max_outside: float


@dataclass(slots=True)
class Config:
    experiment_disable_full_structural_group: bool = False
    ablation_disable_history_tabu: bool = False
    experiment_launch_bound_cache: bool = False
    experiment_regret_exact_bound: bool = True
    experiment_selective_post: str = "always"
    experiment_post_radius_scale: float = 1.0
    experiment_honor_reheat_pulse: bool = False
    experiment_disable_structural_directed: bool = False
    experiment_remove_op20: bool = False
    # Experiment switches are off by default; the original file is untouched.
    experiment_analytic_launch: bool = True
    experiment_deferred_regret_copy: bool = True
    experiment_standard_regret: bool = False
    experiment_legal_blocks: bool = False
    experiment_reconstruct_guard: bool = False
    experiment_noop_failure: bool = False
    experiment_causal_wait: bool = False
    experiment_polish_budget: bool = False
    experiment_cpu_budget: float = 0.0
    experiment_final_reserve: float = 0.15
    uend: int = 4
    drone_capacity: float = 2.5
    drone_endurance: float = 0.5

    road_factor: float = 1.5
    truck_speed: float = 30.0
    drone_speed: float = 60.0
    truck_fixed_cost: float = 100.0
    drone_fixed_cost: float = 20.0
    truck_unit_cost_per_hour: float = 90.0
    drone_unit_cost_per_hour: float = 18.0
    wait_cost_per_hour: float = 18.0
    tardiness_cost_per_hour: float = 40.0

    # Carbon-tax extension. Distances are measured in km and emissions in kgCO2e.
    carbon_price_per_kg: float = 0.2
    truck_emission_kg_per_km: float = 0.2155
    drone_emission_kg_per_km: float = 0.01378

    M: float = 100000.0

    # A single ALNS trajectory: exploration is extended moderately, while the
    # final 10% of the budget is kept for low-temperature intensification.
    max_iter: int = 4500
    stage_cut1: float = 0.08
    stage_cut2: float = 0.26
    stage_cut3: float = 0.55
    sa_start_ratio: float = 0.065
    sa_end_ratio: float = 0.00012
    sa_freeze_ratio: float = 0.72

    # Adaptive-search control parameters.
    cycle_update: int = 50
    cycle_no_improve_reconstruct: int = 220
    cycle_no_improve_finish: int = 2200
    cycle_no_improve_polish: int = 900

    enable_large_reconstruct: bool = True
    enable_early_stop: bool = False
    enable_final_polish: bool = True
    enable_plateau_polish: bool = True
    polish_passes: int = 3
    plateau_polish_passes: int = 1

    # Every instance receives deterministic VND intensification. Search depth
    # follows (1 + n / reference_customers)^(-decay_exponent).
    enable_global_polish: bool = True
    global_polish_reference_customers: float = 25.0
    global_polish_decay_exponent: float = 1.0
    global_polish_passes: int = 6
    global_polish_beam_width: int = 8
    global_polish_top_insertions: int = 3
    global_sortie_permutation_max_customers: int = 5
    global_polish_candidate_limit: int = 8
    global_polish_vnd_passes: int = 5

    # Standard single-trajectory ALNS diversification controls. On a plateau,
    # the destroy size is enlarged for a short pulse, the SA temperature is
    # reheated, and one adaptive destroy-repair restart is applied.
    adaptive_destroy_scale: float = 1.75
    adaptive_destroy_max_fraction: float = 0.65
    adaptive_destroy_pulse: int = 110
    adaptive_reheat_ratio: float = 0.20
    adaptive_reheat_stop_ratio: float = 0.88
    large_reconstruct_attempts: int = 2
    large_reconstruct_min_fraction: float = 0.24
    large_reconstruct_max_fraction: float = 0.58
    reconstruct_growth_scale: float = 1.0
    reconstruct_best_base_min_probability: float = 0.20
    reconstruct_best_base_max_probability: float = 0.85
    reconstruct_pulse_min_strength: float = 0.35
    reconstruct_weight_smoothing_min: float = 0.08
    reconstruct_weight_smoothing_max: float = 0.20
    force_final_sa_freeze: bool = True
    reconstruct_intensify_ratio: float = 0.58
    runtime_destroy_scale: float = 1.0

    # Immediately exploit a promising basin after a feasible large reconstruction.
    # The intensification remains inside the single ALNS trajectory and accepts
    # strict improvements only.
    enable_reconstruct_immediate_intensification: bool = True
    reconstruct_intensification_passes: int = 1
    tardiness_directed_top_pool: int = 5
    tardiness_directed_min_remove: int = 2
    tardiness_directed_max_remove: int = 5

    # Apply every search stage across the entire iteration budget. Size-based
    # adaptation remains in the destroy and local-polish operators.
    runtime_schedule_horizon: int = 0
    enable_schedule_horizon_polish: bool = False

    # Evaluation caches are performance-only and do not alter candidate order,
    # random calls, acceptance decisions, or operator scores.
    sortie_metric_cache_size: int = 20000
    sortie_schedule_cache_size: int = 12000
    evaluation_cache_size: int = 512

    # Distribution-robust candidate generation and construction controls.
    endpoint_candidate_min: int = 12
    endpoint_candidate_max: int = 14
    endpoint_candidate_route_factor: float = 0.45
    regret_endpoint_candidate_limit: int = 8
    endpoint_candidate_cache_size: int = 1024
    endpoint_distance_share: float = 0.60
    endpoint_span_share: float = 0.20
    initial_portfolio_size: int = 5
    two_opt_extra_candidates: int = 6
    exposure_uses_service_time: bool = True
    # When enabled, a sortie may launch as soon as both the truck and the
    # assigned drone are present. Customer readiness constrains truck service
    # and departure, but no longer delays an otherwise feasible launch.
    allow_launch_during_truck_ready_wait: bool = True
    shaw_distance_weight: float = 0.27
    shaw_time_weight: float = 0.34
    shaw_demand_weight: float = 0.10
    shaw_route_weight: float = 0.19
    shaw_mode_weight: float = 0.10
    shaw_same_sortie_bonus: float = 0.25
    shaw_random_pick_probability: float = 0.35

    plot_route_map: bool = True
    plot_objective_history: bool = True

    tabu_len: int = 20
    alpha: float = 0.3
    m: int = 8

    score1: float = 30.0
    score2: float = 25.0
    score3: float = 10.0
    score4: float = 2.0
    score5: float = 0.0

    depot_start_id: int = 0
    depot_end_id: int = -1
    depot_start_x: float = 0.0
    depot_start_y: float = 0.0
    depot_end_x: float = 0.0
    depot_end_y: float = 0.0

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Config":
        cfg = Config()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


@dataclass(slots=True)
class Sortie:
    drone: int
    index: int
    launch: int
    recovery: int
    customers: List[int] = field(default_factory=list)

    def clone(self) -> "Sortie":
        return Sortie(
            drone=self.drone,
            index=self.index,
            launch=self.launch,
            recovery=self.recovery,
            customers=list(self.customers),
        )

    def route_nodes(self) -> List[int]:
        return [self.launch] + self.customers + [self.recovery]


@dataclass(slots=True)
class Solution:
    truck_route: List[int]
    sorties_by_drone: Dict[int, List[Sortie]]

    def clone(self) -> "Solution":
        return Solution(
            truck_route=list(self.truck_route),
            sorties_by_drone={u: [s.clone() for s in lst] for u, lst in self.sorties_by_drone.items()},
        )

    def normalize(self, route_pos: Optional[Dict[int, int]] = None) -> None:
        for u in list(self.sorties_by_drone.keys()):
            sorties = self.sorties_by_drone[u]
            if not sorties:
                continue
            if route_pos is not None:
                sorties.sort(
                    key=lambda s: (route_pos.get(s.launch, 10 ** 9), route_pos.get(s.recovery, 10 ** 9), s.index))
            else:
                sorties.sort(key=lambda s: s.index)
            for idx, s in enumerate(sorties, start=1):
                s.index = idx
            self.sorties_by_drone[u] = sorties

        active_u_list = sorted([u for u, lst in self.sorties_by_drone.items() if lst])
        all_u_list = sorted(list(self.sorties_by_drone.keys()))
        new_sorties_by_drone = {u: [] for u in all_u_list}

        for new_idx, old_u in enumerate(active_u_list):
            target_u = all_u_list[new_idx]
            sorties = self.sorties_by_drone[old_u]
            for s in sorties:
                s.drone = target_u
            new_sorties_by_drone[target_u] = sorties

        self.sorties_by_drone = new_sorties_by_drone


@dataclass(slots=True)
class Evaluation:
    feasible: bool
    objective: float
    c1: float
    c2: float
    c3: float
    c4: float
    c5: float
    truck_distance: float
    drone_distance: float
    carbon_emissions: float
    total_tardiness: float
    late_customers: int
    violations: List[str]
    missing_violation_customers: Set[int]
    non_missing_violation_count: int
    truck_arrival: Dict[int, float]
    truck_departure: Dict[int, float]
    truck_wait: Dict[int, float]
    truck_tardiness: Dict[int, float]
    drone_tardiness: Dict[int, float]
    drone_customer_service_time: Dict[Tuple[int, int, int], float]
    drone_launch_time: Dict[Tuple[int, int], float]
    drone_recovery_time: Dict[Tuple[int, int], float]
    customer_mode: Dict[int, str]
    fingerprint: str


@dataclass(slots=True)
class OperatorStat:
    name: str
    weight: float = 12.0
    score_sum: float = 0.0
    use_count: int = 0


class ProblemData:
    def __init__(
            self,
            customers: Dict[int, Customer],
            cfg: Config,
            depot_start: Tuple[float, float],
            depot_end: Tuple[float, float],
    ) -> None:
        self.customers = customers
        self.cfg = cfg

        self.depot_start_id = cfg.depot_start_id
        self.depot_end_id = cfg.depot_end_id if cfg.depot_end_id >= 0 else (
            max(customers.keys()) + 1 if customers else 1)
        self.cfg.depot_end_id = self.depot_end_id

        self.node_xy: Dict[int, Tuple[float, float]] = {
            self.depot_start_id: depot_start,
            self.depot_end_id: depot_end,
        }
        for cid, c in customers.items():
            self.node_xy[cid] = (c.x, c.y)

        self.C = sorted(customers.keys())
        self.V = [self.depot_start_id] + self.C + [self.depot_end_id]
        self.VL = [self.depot_start_id] + self.C
        self.VR = self.C + [self.depot_end_id]
        self.U = list(range(1, cfg.uend + 1))
        self.Nk = {cid for cid, c in customers.items() if not c.can_drone}
        self.Nku = {cid for cid, c in customers.items() if c.can_drone}
        self.V_set = set(self.V)

        self._sortie_flight_time_cache: Dict[Tuple[int, Tuple[int, ...], int], float] = {}
        self._sortie_distance_cache: Dict[Tuple[int, Tuple[int, ...], int], float] = {}
        self._sortie_payload_cache: Dict[Tuple[int, ...], float] = {}
        self._sortie_simulation_cache: Dict[Tuple[Any, ...], Tuple[Any, ...]] = {}
        self._sortie_launch_cache: Dict[Tuple[Any, ...], Tuple[Any, ...]] = {}
        self._launch_bound_cache: Dict[Tuple[Any, ...], Tuple[bool, float]] = {}
        self._regret_bound_pruned = 0
        self._regret_bound_checked = 0
        self._sortie_schedule_cache_hits = 0
        self._sortie_schedule_cache_misses = 0
        self._evaluation_cache: OrderedDict[Tuple[Any, ...], Evaluation] = OrderedDict()
        self._evaluation_cache_hits = 0
        self._evaluation_cache_misses = 0
        self._endpoint_candidate_cache: OrderedDict[
            Tuple[Tuple[int, ...], int], Tuple[Tuple[int, int, float], ...]
        ] = OrderedDict()

        self.truck_dist: Dict[Tuple[int, int], float] = {}
        self.drone_dist: Dict[Tuple[int, int], float] = {}
        self.truck_tt: Dict[Tuple[int, int], float] = {}
        self.drone_tt: Dict[Tuple[int, int], float] = {}
        matrix_size = max(self.V, default=-1) + 1
        self._use_dense_matrices = (
            matrix_size > 0
            and min(self.V, default=0) >= 0
            and matrix_size <= max(16, 4 * len(self.V))
        )
        if self._use_dense_matrices:
            self.truck_dist_dense = [[0.0] * matrix_size for _ in range(matrix_size)]
            self.drone_dist_dense = [[0.0] * matrix_size for _ in range(matrix_size)]
            self.truck_tt_dense = [[0.0] * matrix_size for _ in range(matrix_size)]
            self.drone_tt_dense = [[0.0] * matrix_size for _ in range(matrix_size)]
        else:
            self.truck_dist_dense = None
            self.drone_dist_dense = None
            self.truck_tt_dense = None
            self.drone_tt_dense = None
        self._build_matrices()
        self._build_scale_metrics()

    def _build_matrices(self) -> None:
        for i in self.V:
            for j in self.V:
                if i == j:
                    d_eu = 0.0
                else:
                    xi, yi = self.node_xy[i]
                    xj, yj = self.node_xy[j]
                    d_eu = math.hypot(xi - xj, yi - yj)
                d_truck = self.cfg.road_factor * d_eu
                d_drone = d_eu
                self.truck_dist[(i, j)] = d_truck
                self.drone_dist[(i, j)] = d_drone
                truck_tt = d_truck / self.cfg.truck_speed if self.cfg.truck_speed > 0 else float("inf")
                drone_tt = d_drone / self.cfg.drone_speed if self.cfg.drone_speed > 0 else float("inf")
                self.truck_tt[(i, j)] = truck_tt
                self.drone_tt[(i, j)] = drone_tt
                if self._use_dense_matrices:
                    self.truck_dist_dense[i][j] = d_truck
                    self.drone_dist_dense[i][j] = d_drone
                    self.truck_tt_dense[i][j] = truck_tt
                    self.drone_tt_dense[i][j] = drone_tt

    def _build_scale_metrics(self) -> None:
        self.max_customer_truck_dist = max(
            (self.truck_dist[(i, j)] for i in self.C for j in self.C if i != j),
            default=1.0,
        )
        ready_values = [self.customers[c].ready for c in self.C]
        due_values = [self.customers[c].due for c in self.C]
        self.max_customer_demand = max(
            (self.customers[c].demand for c in self.C),
            default=1.0,
        )
        self.ready_span = max(ready_values, default=1.0) - min(ready_values, default=0.0)
        self.due_span = max(due_values, default=1.0) - min(due_values, default=0.0)
        self.cfg.runtime_schedule_horizon = self.cfg.max_iter
def _validate_customer_values(
        row_number: int,
        cid: int,
        x: float,
        y: float,
        demand: float,
        ready: float,
        due: float,
        max_outside: float,
) -> None:
    values = {
        "x": x,
        "y": y,
        "demand": demand,
        "ready": ready,
        "due": due,
        "max_outside": max_outside,
    }
    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"Row {row_number}: {name} must be finite for customer {cid}")
    if demand < 0.0:
        raise ValueError(f"Row {row_number}: demand must be non-negative for customer {cid}")
    if ready > due:
        raise ValueError(f"Row {row_number}: ready must not exceed due for customer {cid}")
    if max_outside < 0.0:
        raise ValueError(f"Row {row_number}: max_outside must be non-negative for customer {cid}")


def load_customers_csv(path: Path, cfg: Config) -> Dict[int, Customer]:
    customers: Dict[int, Customer] = {}
    depot_found = False
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "x", "y", "demand", "can_drone", "ready", "due", "max_outside"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"customers.csv missing columns: {sorted(required)}")
        for r_idx, row in enumerate(reader, start=2):
            id_raw = row["id"]
            if id_raw is None or str(id_raw).strip() == "":
                continue
            try:
                cid = int(float(id_raw))
            except Exception as e:
                raise ValueError(f"Row {r_idx}: invalid id={id_raw!r}") from e

            x = float(row["x"])
            y = float(row["y"])
            if cid == cfg.depot_start_id:
                if depot_found:
                    raise ValueError(f"Row {r_idx}: duplicated depot id={cid}")
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError(f"Row {r_idx}: depot coordinates must be finite")
                cfg.depot_start_x = x
                cfg.depot_start_y = y
                depot_found = True
                continue
            if cid <= 0:
                raise ValueError(f"Row {r_idx}: customer id must be > 0 (only depot is id=0)")
            if cid in customers:
                raise ValueError(f"Row {r_idx}: duplicated customer id={cid}")

            demand = float(row["demand"])
            ready = float(row["ready"])
            due = float(row["due"])
            max_outside = float(row["max_outside"])
            _validate_customer_values(r_idx, cid, x, y, demand, ready, due, max_outside)
            customers[cid] = Customer(
                cid=cid,
                x=x,
                y=y,
                demand=demand,
                can_drone=_parse_bool(row["can_drone"]),
                ready=ready,
                due=due,
                max_outside=max_outside,
            )
    if not depot_found:
        raise ValueError("CSV must include one row with id=0 as depot start.")
    if not customers:
        raise ValueError("No customer rows found (id>0).")
    if cfg.depot_end_id < 0:
        cfg.depot_end_x = cfg.depot_start_x
        cfg.depot_end_y = cfg.depot_start_y
    return customers


def _parse_bool(v: Any) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}


def load_customers_excel(path: Path, cfg: Config) -> Dict[int, Customer]:
    try:
        from openpyxl import load_workbook
    except Exception as e:
        raise ImportError("openpyxl is required for --excel input. Install with: pip install openpyxl") from e

    wb = load_workbook(path, data_only=True, read_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    header = next(rows, None)
    if header is None:
        wb.close()
        raise ValueError(f"Excel is empty: {path}")

    header_map: Dict[str, int] = {}
    for idx, h in enumerate(header):
        if h is not None:
            header_map[str(h).strip().lower()] = idx

    aliases = {
        "can": "can_drone",
        "e": "ready",
        "l": "due",
        "out": "max_outside",
    }
    for src, dst in aliases.items():
        if src in header_map and dst not in header_map:
            header_map[dst] = header_map[src]

    required = ["id", "x", "y", "demand", "can_drone", "ready", "due", "max_outside"]
    missing = [c for c in required if c not in header_map]
    if missing:
        wb.close()
        raise ValueError(f"Missing columns in Excel: {missing}")

    def getv(row: Tuple[Any, ...], col: str) -> Any:
        idx = header_map[col]
        return row[idx] if idx < len(row) else None

    customers: Dict[int, Customer] = {}
    depot_found = False

    for r_idx, row in enumerate(rows, start=2):
        if row is None:
            continue

        id_raw = getv(row, "id")
        if id_raw is None or str(id_raw).strip() == "":
            continue

        try:
            cid = int(float(id_raw))
        except Exception as e:
            wb.close()
            raise ValueError(f"Row {r_idx}: invalid id={id_raw!r}") from e

        x_raw = getv(row, "x")
        y_raw = getv(row, "y")
        if x_raw is None or y_raw is None:
            wb.close()
            raise ValueError(f"Row {r_idx}: x or y is empty")
        x = float(x_raw)
        y = float(y_raw)

        if cid == cfg.depot_start_id:
            if depot_found:
                wb.close()
                raise ValueError(f"Row {r_idx}: duplicated depot id={cid}")
            if not math.isfinite(x) or not math.isfinite(y):
                wb.close()
                raise ValueError(f"Row {r_idx}: depot coordinates must be finite")
            cfg.depot_start_x = x
            cfg.depot_start_y = y
            depot_found = True
            continue

        if cid <= 0:
            wb.close()
            raise ValueError(f"Row {r_idx}: customer id must be > 0 (only depot is id=0)")
        if cid in customers:
            wb.close()
            raise ValueError(f"Row {r_idx}: duplicated customer id={cid}")

        demand_raw = getv(row, "demand")
        can_drone_raw = getv(row, "can_drone")
        ready_raw = getv(row, "ready")
        due_raw = getv(row, "due")
        max_outside_raw = getv(row, "max_outside")
        if None in {demand_raw, can_drone_raw, ready_raw, due_raw, max_outside_raw}:
            wb.close()
            raise ValueError(f"Row {r_idx}: one or more customer fields are empty")

        demand = float(demand_raw)
        ready = float(ready_raw)
        due = float(due_raw)
        max_outside = float(max_outside_raw)
        try:
            _validate_customer_values(r_idx, cid, x, y, demand, ready, due, max_outside)
        except Exception:
            wb.close()
            raise
        customers[cid] = Customer(
            cid=cid,
            x=x,
            y=y,
            demand=demand,
            can_drone=_parse_bool(can_drone_raw),
            ready=ready,
            due=due,
            max_outside=max_outside,
        )

    wb.close()

    if not depot_found:
        raise ValueError("Excel must include one row with id=0 as depot start.")
    if not customers:
        raise ValueError("No customer rows found (id>0).")

    if cfg.depot_end_id < 0:
        cfg.depot_end_x = cfg.depot_start_x
        cfg.depot_end_y = cfg.depot_start_y

    return customers


def load_config_json(path: Path) -> Config:
    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    return Config.from_dict(d)


def create_synthetic_problem(seed: Optional[int] = None, n_customers: int = 25,
                             cfg: Optional[Config] = None) -> ProblemData:
    if seed is None:
        seed = time.time_ns() & 0xFFFFFFFF
    rng = random.Random(seed)
    cfg = cfg or Config()
    customers: Dict[int, Customer] = {}
    for cid in range(1, n_customers + 1):
        ready = rng.uniform(0.0, 4.0)
        customers[cid] = Customer(
            cid=cid,
            x=rng.uniform(0, 35),
            y=rng.uniform(0, 35),
            demand=rng.uniform(0.5, 8.0),
            can_drone=(rng.random() < 0.65),
            ready=ready,
            due=ready + rng.uniform(1.0, 5.0),
            max_outside=rng.uniform(0.2, 1.8),
        )
    return ProblemData(
        customers=customers,
        cfg=cfg,
        depot_start=(cfg.depot_start_x, cfg.depot_start_y),
        depot_end=(cfg.depot_end_x, cfg.depot_end_y),
    )


def fingerprint_solution(sol: Solution) -> str:
    parts: List[str] = []
    parts.append("T:" + ",".join(map(str, sol.truck_route)))
    for u in sorted(sol.sorties_by_drone.keys()):
        seg = []
        for s in sorted(sol.sorties_by_drone[u], key=lambda x: x.index):
            seg.append(f"({s.launch}|{'-'.join(map(str, s.customers))}|{s.recovery})")
        parts.append(f"U{u}:{';'.join(seg)}")
    return " || ".join(parts)


def evaluation_cache_key(sol: Solution, data: ProblemData) -> Tuple[Any, ...]:
    return (
        tuple(sol.truck_route),
        tuple(
            (
                u,
                tuple(
                    (s.launch, tuple(s.customers), s.recovery)
                    for s in sol.sorties_by_drone.get(u, [])
                ),
            )
            for u in data.U
        ),
    )


def route_positions(route: List[int]) -> Dict[int, int]:
    return {node: idx for idx, node in enumerate(route)}


def normalized_solution_copy(sol: Solution, route_pos: Dict[int, int]) -> Solution:
    """Build the same canonical copy as clone()+normalize(), in one pass."""
    all_drones = sorted(sol.sorties_by_drone)
    active_drones = [u for u in all_drones if sol.sorties_by_drone[u]]
    normalized = {u: [] for u in all_drones}
    for target_drone, source_drone in zip(all_drones, active_drones):
        ordered = sorted(
            sol.sorties_by_drone[source_drone],
            key=lambda s: (
                route_pos.get(s.launch, 10 ** 9),
                route_pos.get(s.recovery, 10 ** 9),
                s.index,
            ),
        )
        normalized[target_drone] = [
            Sortie(
                drone=target_drone,
                index=index,
                launch=s.launch,
                recovery=s.recovery,
                customers=list(s.customers),
            )
            for index, s in enumerate(ordered, start=1)
        ]
    return Solution(truck_route=list(sol.truck_route), sorties_by_drone=normalized)


def normalized_solution_view(sol: Solution, route_pos: Dict[int, int]) -> Solution:
    """Reuse a canonical solution; copy only candidates that actually need normalization."""
    all_drones = sorted(sol.sorties_by_drone)
    active_drones = [u for u in all_drones if sol.sorties_by_drone[u]]
    if active_drones != all_drones[:len(active_drones)]:
        return normalized_solution_copy(sol, route_pos)

    for u in active_drones:
        previous_key: Optional[Tuple[int, int, int]] = None
        for expected_index, sortie in enumerate(sol.sorties_by_drone[u], start=1):
            key = (
                route_pos.get(sortie.launch, 10 ** 9),
                route_pos.get(sortie.recovery, 10 ** 9),
                sortie.index,
            )
            if (
                    sortie.drone != u
                    or sortie.index != expected_index
                    or (previous_key is not None and key < previous_key)
            ):
                return normalized_solution_copy(sol, route_pos)
            previous_key = key
    return sol


def launch_recovery_nodes(sol: Solution) -> Set[int]:
    out: Set[int] = set()
    for sorties in sol.sorties_by_drone.values():
        for s in sorties:
            out.add(s.launch)
            out.add(s.recovery)
    return out


def customers_in_sorties(sol: Solution) -> Set[int]:
    out: Set[int] = set()
    for sorties in sol.sorties_by_drone.values():
        for s in sorties:
            out.update(s.customers)
    return out


def sortie_flight_time(sortie: Sortie, data: ProblemData) -> float:
    key = (sortie.launch, tuple(sortie.customers), sortie.recovery)
    cached = data._sortie_flight_time_cache.get(key)
    if cached is not None:
        return cached
    nodes = sortie.route_nodes()
    if data.drone_tt_dense is not None:
        matrix = data.drone_tt_dense
        value = sum(matrix[i][j] for i, j in zip(nodes[:-1], nodes[1:]))
    else:
        value = sum(data.drone_tt[(i, j)] for i, j in zip(nodes[:-1], nodes[1:]))
    if data.cfg.sortie_metric_cache_size > 0:
        if len(data._sortie_flight_time_cache) >= data.cfg.sortie_metric_cache_size:
            data._sortie_flight_time_cache.clear()
        data._sortie_flight_time_cache[key] = value
    return value


def sortie_flight_distance(sortie: Sortie, data: ProblemData) -> float:
    key = (sortie.launch, tuple(sortie.customers), sortie.recovery)
    cached = data._sortie_distance_cache.get(key)
    if cached is not None:
        return cached
    nodes = sortie.route_nodes()
    if data.drone_dist_dense is not None:
        matrix = data.drone_dist_dense
        value = sum(matrix[i][j] for i, j in zip(nodes[:-1], nodes[1:]))
    else:
        value = sum(data.drone_dist[(i, j)] for i, j in zip(nodes[:-1], nodes[1:]))
    if data.cfg.sortie_metric_cache_size > 0:
        if len(data._sortie_distance_cache) >= data.cfg.sortie_metric_cache_size:
            data._sortie_distance_cache.clear()
        data._sortie_distance_cache[key] = value
    return value


def sortie_payload(sortie: Sortie, data: ProblemData) -> float:
    key = tuple(sortie.customers)
    cached = data._sortie_payload_cache.get(key)
    if cached is not None:
        return cached
    value = sum(data.customers[c].demand for c in sortie.customers)
    if data.cfg.sortie_metric_cache_size > 0:
        if len(data._sortie_payload_cache) >= data.cfg.sortie_metric_cache_size:
            data._sortie_payload_cache.clear()
        data._sortie_payload_cache[key] = value
    return value


def has_duplicates(items: Iterable[int]) -> bool:
    seen: Set[int] = set()
    for x in items:
        if x in seen:
            return True
        seen.add(x)
    return False


def is_sortie_feasible_fast(sortie: Sortie, data: ProblemData) -> bool:
    if sortie_payload(sortie, data) > data.cfg.drone_capacity + 1e-9:
        return False
    if sortie_flight_time(sortie, data) > data.cfg.drone_endurance + 1e-9:
        return False
    return True


def roulette_select(stats: List[OperatorStat], rng: random.Random) -> int:
    total = sum(max(1e-9, s.weight) for s in stats)
    r = rng.random() * total
    acc = 0.0
    for idx, s in enumerate(stats):
        acc += max(1e-9, s.weight)
        if acc >= r:
            return idx
    return len(stats) - 1


def continuous_size_strength(
        customer_count: int,
        reference_customers: float,
        decay_exponent: float = 1.0,
) -> float:
    """Return a smooth (0, 1] search-depth multiplier for every instance size."""
    n = max(0.0, float(customer_count))
    reference = max(1.0, float(reference_customers))
    exponent = max(1e-6, float(decay_exponent))
    return (1.0 + n / reference) ** -exponent


def stage_of_iteration(it: int, max_it: int, cfg: Optional[Config] = None) -> int:
    schedule_horizon = max_it
    if cfg is not None and cfg.runtime_schedule_horizon > 0:
        schedule_horizon = min(max_it, cfg.runtime_schedule_horizon)
    ratio = it / max(1, schedule_horizon)
    cut1 = cfg.stage_cut1 if cfg is not None else 0.1
    cut2 = cfg.stage_cut2 if cfg is not None else 0.2
    cut3 = cfg.stage_cut3 if cfg is not None else 0.4
    if ratio < cut1: return 0
    if ratio < cut2: return 1
    if ratio < cut3: return 2
    return 3


def sample_remove_count(total: int, stage: int, ranges: List[Tuple[float, float]], rng: random.Random) -> int:
    if total <= 0:
        return 0
    lo_p, hi_p = ranges[stage]
    lo = max(1, int(math.floor(total * lo_p)))
    hi = max(1, int(math.ceil(total * hi_p)))

    lo = min(lo, total)
    hi = min(max(hi, lo), total)
    return rng.randint(lo, hi)


def operator_remove_counts(total: int, it: int, max_it: int, ranges: List[Tuple[float, float]],
                           rng: random.Random, cfg: Optional[Config] = None) -> int:
    stage = stage_of_iteration(it, max_it, cfg)
    if cfg is None or cfg.runtime_destroy_scale <= 1.0 + 1e-12:
        return sample_remove_count(total, stage, ranges, rng)

    scale = max(1.0, cfg.runtime_destroy_scale)
    cap = min(1.0, max(0.05, cfg.adaptive_destroy_max_fraction))
    scaled_ranges = [
        (min(cap, lo * scale), min(cap, hi * scale))
        for lo, hi in ranges
    ]
    return sample_remove_count(total, stage, scaled_ranges, rng)


def get_customer_modes(sol: Solution, data: ProblemData) -> Dict[int, str]:
    drone_set = customers_in_sorties(sol)
    truck_set = set(sol.truck_route)
    out: Dict[int, str] = {}
    for c in data.C:
        if c in drone_set:
            out[c] = "drone"
        elif c in truck_set:
            out[c] = "truck"
        else:
            out[c] = "none"
    return out


def evaluate_solution(
        data: ProblemData,
        sol: Solution,
        compact_missing_violations: bool = False,
) -> Evaluation:
    cfg = data.cfg
    violations: List[str] = []
    missing_violation_customers: Set[int] = set()
    missing_violation_count = 0
    pos = route_positions(sol.truck_route)
    sol = normalized_solution_view(sol, pos)

    cache_size = max(0, int(data.cfg.evaluation_cache_size))
    cache_enabled = cache_size > 0
    cache_key: Optional[Tuple[Any, ...]] = None
    if cache_enabled:
        cache_key = (
            compact_missing_violations,
            bool(cfg.allow_launch_during_truck_ready_wait),
            evaluation_cache_key(sol, data),
        )
        cached = data._evaluation_cache.get(cache_key)
        if cached is not None:
            data._evaluation_cache.move_to_end(cache_key)
            data._evaluation_cache_hits += 1
            return cached
        data._evaluation_cache_misses += 1

    if not sol.truck_route:
        violations.append("Truck route is empty.")
    else:
        if sol.truck_route[0] != data.depot_start_id:
            violations.append("Truck route does not start at depot start.")
        if sol.truck_route[-1] != data.depot_end_id:
            violations.append("Truck route does not end at depot end.")

    route_customers = [n for n in sol.truck_route if n in data.customers]
    truck_customer_set = set(route_customers)
    if has_duplicates(route_customers):
        violations.append("Truck route has duplicated customer visits.")
    for n in sol.truck_route:
        if n not in data.V_set:
            violations.append(f"Truck route contains unknown node {n}.")

    all_sorties = [
        (u, s)
        for u in data.U
        for s in sol.sorties_by_drone.get(u, [])
    ]
    drone_customer_set = {
        c
        for _, sortie in all_sorties
        for c in sortie.customers
    }
    modes = {
        c: "drone" if c in drone_customer_set else "truck" if c in truck_customer_set else "none"
        for c in data.C
    }
    for i in data.Nk:
        if modes.get(i) != "truck":
            if not compact_missing_violations:
                violations.append(f"Nk customer {i} is not served by truck.")
            missing_violation_customers.add(i)
            missing_violation_count += 1
    for i in data.Nku:
        if modes.get(i) not in {"truck", "drone"}:
            if not compact_missing_violations:
                violations.append(f"Nku customer {i} is not served.")
            missing_violation_customers.add(i)
            missing_violation_count += 1
    sortie_flight_times: Dict[int, float] = {}
    sortie_signatures: Dict[int, Tuple[int, Tuple[int, ...], int]] = {}
    seen_drone_customers: Set[int] = set()
    for u in data.U:
        sorties = sol.sorties_by_drone.get(u, [])
        if not sorties: continue
        for s in sorties:
            if s.launch == s.recovery:
                violations.append(f"Sortie U{u}-P{s.index}: launch equals recovery.")
            if s.launch not in pos or s.recovery not in pos:
                violations.append(f"Sortie U{u}-P{s.index}: launch/recovery not on truck route.")
            elif pos[s.launch] >= pos[s.recovery]:
                violations.append(f"Sortie U{u}-P{s.index}: launch is not before recovery.")
            if not s.customers:
                violations.append(f"Sortie U{u}-P{s.index}: empty customer list.")
            for c in s.customers:
                if c not in data.Nku:
                    violations.append(f"Sortie U{u}-P{s.index} serves non-Nku customer {c}.")
                if c in truck_customer_set:
                    violations.append(f"Customer {c} is served by both truck and drone.")
                if c in seen_drone_customers:
                    violations.append(f"Customer {c} appears in multiple drone sorties.")
                seen_drone_customers.add(c)
            if sortie_payload(s, data) > cfg.drone_capacity + 1e-9:
                violations.append(f"Sortie U{u}-P{s.index} exceeds drone capacity Q.")
            flight_time = sortie_flight_time(s, data)
            sortie_flight_times[id(s)] = flight_time
            sortie_signatures[id(s)] = (s.launch, tuple(s.customers), s.recovery)
            if flight_time > cfg.drone_endurance + 1e-9:
                violations.append(f"Sortie U{u}-P{s.index} exceeds drone endurance BU.")

    for u in data.U:
        sorties = sol.sorties_by_drone.get(u, [])
        for a, b in zip(sorties[:-1], sorties[1:]):
            if a.recovery in pos and b.launch in pos and pos[a.recovery] > pos[b.launch]:
                violations.append(f"Drone {u}: recovery of sortie {a.index} is after launch of {b.index}.")

    if compact_missing_violations and violations:
        return Evaluation(
            feasible=False,
            objective=float("inf"),
            c1=0.0,
            c2=0.0,
            c3=0.0,
            c4=0.0,
            c5=0.0,
            truck_distance=0.0,
            drone_distance=0.0,
            carbon_emissions=0.0,
            total_tardiness=0.0,
            late_customers=0,
            violations=violations,
            missing_violation_customers=missing_violation_customers,
            non_missing_violation_count=len(violations),
            truck_arrival={},
            truck_departure={},
            truck_wait={},
            truck_tardiness={},
            drone_tardiness={},
            drone_customer_service_time={},
            drone_launch_time={},
            drone_recovery_time={},
            customer_mode={},
            fingerprint="",
        )

    route_length = len(sol.truck_route)
    launches_by_pos: List[List[Sortie]] = [[] for _ in range(route_length)]
    for _, s in all_sorties:
        if s.launch in pos:
            launches_by_pos[pos[s.launch]].append(s)
    for launches in launches_by_pos:
        launches.sort(key=lambda s: (s.drone, s.index))

    truck_arrival: Dict[int, float] = {}
    truck_departure: Dict[int, float] = {}
    truck_wait: Dict[int, float] = defaultdict(float)
    truck_tard: Dict[int, float] = defaultdict(float)
    drone_tard: Dict[int, float] = defaultdict(float)
    drone_service_time: Dict[Tuple[int, int, int], float] = {}
    drone_launch_time: Dict[Tuple[int, int], float] = {}
    drone_recovery_time: Dict[Tuple[int, int], float] = {}
    drone_available: Dict[int, float] = {u: 0.0 for u in data.U}
    pending_recovery: List[List[float]] = [[] for _ in range(route_length)]
    customer_data = data.customers
    drone_tt = data.drone_tt
    drone_tt_dense = data.drone_tt_dense
    truck_tt_dense = data.truck_tt_dense
    schedule_cache_size = max(0, int(cfg.sortie_schedule_cache_size))
    simulation_cache = data._sortie_simulation_cache
    launch_cache = data._sortie_launch_cache

    def simulate_sortie(s: Sortie, launch_time: float) -> Tuple[Dict[int, float], float, float, float]:
        cache_key = (
            sortie_signatures[id(s)],
            launch_time,
            cfg.exposure_uses_service_time,
        )
        cached = simulation_cache.get(cache_key)
        if cached is not None:
            data._sortie_schedule_cache_hits += 1
            return cached
        data._sortie_schedule_cache_misses += 1
        arrivals: Dict[int, float] = {}
        current_departure = launch_time
        prev = s.launch
        pure_flight = sortie_flight_times[id(s)]
        max_outside_excess = float("-inf")

        for c in s.customers:
            leg = drone_tt_dense[prev][c] if drone_tt_dense is not None else drone_tt[(prev, c)]
            physical_arrival = current_departure + leg
            service_time = (
                max(physical_arrival, customer_data[c].ready)
                if cfg.exposure_uses_service_time
                else physical_arrival
            )
            arrivals[c] = service_time
            max_outside_excess = max(
                max_outside_excess,
                service_time - launch_time - customer_data[c].max_outside,
            )
            current_departure = max(physical_arrival, customer_data[c].ready)
            prev = c

        back_leg = (
            drone_tt_dense[prev][s.recovery]
            if drone_tt_dense is not None
            else drone_tt[(prev, s.recovery)]
        )
        recovery_time = current_departure + back_leg
        result = (arrivals, recovery_time, pure_flight, max_outside_excess)
        if schedule_cache_size > 0:
            if len(simulation_cache) >= schedule_cache_size:
                simulation_cache.clear()
            simulation_cache[cache_key] = result
        return result

    def max_outside_excess_only(s: Sortie, launch_time: float) -> float:
        current_departure = launch_time
        prev = s.launch
        max_outside_excess = float("-inf")
        for c in s.customers:
            leg = drone_tt_dense[prev][c] if drone_tt_dense is not None else drone_tt[(prev, c)]
            physical_arrival = current_departure + leg
            service_time = (
                max(physical_arrival, customer_data[c].ready)
                if cfg.exposure_uses_service_time
                else physical_arrival
            )
            max_outside_excess = max(
                max_outside_excess,
                service_time - launch_time - customer_data[c].max_outside,
            )
            current_departure = max(physical_arrival, customer_data[c].ready)
            prev = c
        return max_outside_excess

    def earliest_word_launch_time(s: Sortie, lower_bound: float) -> Tuple[float, Dict[int, float], float, float, float]:
        cache_key = (
            sortie_signatures[id(s)],
            lower_bound,
            cfg.exposure_uses_service_time,
        )
        cached = launch_cache.get(cache_key)
        if cached is not None:
            data._sortie_schedule_cache_hits += 1
            return cached
        if cfg.experiment_analytic_launch and cfg.exposure_uses_service_time:
            bound_key = (s.launch, tuple(s.customers))
            bound = data._launch_bound_cache.get(bound_key) if cfg.experiment_launch_bound_cache else None
            if bound is None:
                prefix = 0.0
                ready_barrier = float('-inf')
                threshold = float('-inf')
                previous = s.launch
                impossible = False
                for customer in s.customers:
                    prefix += drone_tt_dense[previous][customer] if drone_tt_dense is not None else drone_tt[(previous, customer)]
                    ready_barrier = max(ready_barrier, customer_data[customer].ready - prefix)
                    allowance = customer_data[customer].max_outside
                    if prefix > allowance + 1e-9:
                        impossible = True
                    threshold = max(threshold, ready_barrier + prefix - allowance)
                    previous = customer
                if cfg.experiment_launch_bound_cache and cfg.sortie_metric_cache_size > 0:
                    if len(data._launch_bound_cache) >= cfg.sortie_metric_cache_size:
                        data._launch_bound_cache.clear()
                    data._launch_bound_cache[bound_key] = (impossible, threshold)
            else:
                impossible, threshold = bound
            threshold = max(lower_bound, threshold)
            launch = lower_bound if impossible else threshold
            arrivals, recovery_time, pure_flight, excess = simulate_sortie(s, launch)
            result = (launch, arrivals, recovery_time, pure_flight, excess)
            if schedule_cache_size > 0:
                if len(launch_cache) >= schedule_cache_size:
                    launch_cache.clear()
                launch_cache[cache_key] = result
            return result
        arrivals, recovery_time, pure_flight, excess = simulate_sortie(s, lower_bound)
        if excess <= 1e-9:
            result = (lower_bound, arrivals, recovery_time, pure_flight, excess)
            if schedule_cache_size > 0:
                if len(launch_cache) >= schedule_cache_size:
                    launch_cache.clear()
                launch_cache[cache_key] = result
            return result

        lo = lower_bound
        hi = max(lower_bound + 1.0, 1.0)
        hi_excess = max_outside_excess_only(s, hi)
        while hi_excess > 1e-9 and hi < cfg.M:
            hi = lower_bound + 2.0 * (hi - lower_bound)
            hi_excess = max_outside_excess_only(s, hi)

        if hi_excess > 1e-9:
            result = (lower_bound, arrivals, recovery_time, pure_flight, excess)
            if schedule_cache_size > 0:
                if len(launch_cache) >= schedule_cache_size:
                    launch_cache.clear()
                launch_cache[cache_key] = result
            return result

        for _ in range(15):
            mid = (lo + hi) / 2.0
            mid_excess = max_outside_excess_only(s, mid)
            if mid_excess <= 1e-9:
                hi = mid
            else:
                lo = mid

        best_arrivals, best_recovery, best_pure, best_excess = simulate_sortie(s, hi)
        result = (hi, best_arrivals, best_recovery, best_pure, best_excess)
        if schedule_cache_size > 0:
            if len(launch_cache) >= schedule_cache_size:
                launch_cache.clear()
            launch_cache[cache_key] = result
        return result

    def process_launches_at_index(ridx: int, t: float) -> float:
        node_available_time = t
        launch_completion = t
        for s in launches_by_pos[ridx]:
            # Different drones have independent zero-duration launch events in
            # the MILP. A delayed launch for one drone must not postpone the
            # other drones launched from the same truck stop.
            launch_lb = max(node_available_time, drone_available[s.drone])
            if ridx == 0:
                arrivals, recovery_time, pure_flight, _ = simulate_sortie(s, launch_lb)
                launch_time = launch_lb
            else:
                launch_time, arrivals, recovery_time, pure_flight, _ = earliest_word_launch_time(s, launch_lb)
            for c, arrival in arrivals.items():
                drone_service_time[(s.drone, s.index, c)] = arrival
                outside = arrival - launch_time
                if outside > data.customers[c].max_outside + 1e-9:
                    violations.append(f"Customer {c}: max outside time violated in U{s.drone}-P{s.index}.")
            if pure_flight > cfg.drone_endurance + 1e-9:
                violations.append(f"Sortie U{s.drone}-P{s.index}: BU violated in schedule.")

            drone_launch_time[(s.drone, s.index)] = launch_time
            drone_recovery_time[(s.drone, s.index)] = recovery_time

            if s.recovery not in pos:
                violations.append(f"Sortie U{s.drone}-P{s.index}: recovery not found on truck route.")
            else:
                rec_idx = pos[s.recovery]
                pending_recovery[rec_idx].append(recovery_time)
                if rec_idx == ridx:
                    launch_completion = max(launch_completion, recovery_time)

            drone_available[s.drone] = recovery_time
            launch_completion = max(launch_completion, launch_time)
        return launch_completion

    def process_truck_stop(ridx: int, node: int, arrival_time: float) -> float:
        recoveries = pending_recovery[ridx]
        recovery_completion = max(recoveries) if recoveries else arrival_time
        service_completion = arrival_time
        if node in data.customers and modes[node] == "truck":
            service_completion = max(arrival_time, data.customers[node].ready)

        if cfg.allow_launch_during_truck_ready_wait:
            # Launches use the truck-arrival event, independently of customer
            # readiness. A launch may still be delayed by drone availability or
            # by the maximum-outside-time feasibility calculation.
            departure_time = process_launches_at_index(ridx, arrival_time)
            return max(departure_time, recovery_completion, service_completion)

        # Retained compatibility branch for controlled regression comparisons.
        departure_time = max(arrival_time, recovery_completion, service_completion)
        return process_launches_at_index(ridx, departure_time)

    if sol.truck_route:
        start_node = sol.truck_route[0]
        truck_arrival[start_node] = 0.0
        t0 = process_truck_stop(0, start_node, 0.0)
        truck_departure[start_node] = t0
        if start_node in data.customers:
            truck_wait[start_node] = max(0.0, t0 - truck_arrival[start_node])
            if modes[start_node] == "truck":
                truck_tard[start_node] = max(0.0, truck_arrival[start_node] - data.customers[start_node].due)

    for ridx in range(1, len(sol.truck_route)):
        prev_node = sol.truck_route[ridx - 1]
        node = sol.truck_route[ridx]

        leg = (
            truck_tt_dense[prev_node][node]
            if truck_tt_dense is not None
            else data.truck_tt[(prev_node, node)]
        )
        arr = truck_departure[prev_node] + leg
        t = process_truck_stop(ridx, node, arr)

        truck_arrival[node] = arr
        truck_departure[node] = t
        if node in data.customers:
            truck_wait[node] = max(0.0, t - arr)
            if modes[node] == "truck":
                truck_tard[node] = max(0.0, arr - data.customers[node].due)

    last_idx = len(sol.truck_route) - 1
    final_recoveries = pending_recovery[last_idx] if last_idx >= 0 else []
    if final_recoveries:
        req = max(final_recoveries)
        if truck_departure.get(sol.truck_route[-1], 0.0) + 1e-9 < req:
            violations.append("Truck ends before all drone recoveries complete.")

    for u, s in all_sorties:
        for c in s.customers:
            ts = drone_service_time.get((u, s.index, c))
            if ts is None:
                violations.append(f"Missing drone service time for customer {c}.")
                continue
            drone_tard[c] = max(0.0, ts - data.customers[c].due)

    c1 = cfg.truck_fixed_cost + cfg.drone_fixed_cost * sum(1 for u in data.U if sol.sorties_by_drone.get(u))
    if truck_tt_dense is not None:
        truck_time = sum(truck_tt_dense[i][j] for i, j in zip(sol.truck_route[:-1], sol.truck_route[1:]))
    else:
        truck_time = sum(data.truck_tt[(i, j)] for i, j in zip(sol.truck_route[:-1], sol.truck_route[1:]))
    drone_time = sum(sortie_flight_times[id(s)] for _, s in all_sorties)
    c2 = truck_time * cfg.truck_unit_cost_per_hour + drone_time * cfg.drone_unit_cost_per_hour
    c3 = sum(truck_wait.get(c, 0.0) for c in data.C if modes.get(c) == "truck") * cfg.wait_cost_per_hour
    truck_t = sum(truck_tard.values())
    drone_t = sum(drone_tard.values())
    c4 = (truck_t + drone_t) * cfg.tardiness_cost_per_hour
    if data.truck_dist_dense is not None:
        truck_distance = sum(
            data.truck_dist_dense[i][j] for i, j in zip(sol.truck_route[:-1], sol.truck_route[1:])
        )
    else:
        truck_distance = sum(
            data.truck_dist[(i, j)] for i, j in zip(sol.truck_route[:-1], sol.truck_route[1:])
        )
    drone_distances: List[float] = []
    for _, s in all_sorties:
        drone_distances.append(sortie_flight_distance(s, data))
    drone_distance = sum(drone_distances)
    carbon_emissions = (
        cfg.truck_emission_kg_per_km * truck_distance
        + cfg.drone_emission_kg_per_km * drone_distance
    )
    c5 = cfg.carbon_price_per_kg * carbon_emissions
    total_tardiness = truck_t + drone_t
    late_customers = len(
        {
            c
            for c, tardiness in list(truck_tard.items()) + list(drone_tard.items())
            if tardiness > 1e-9
        }
    )
    objective = c1 + c2 + c3 + c4 + c5

    materialized_missing_count = 0 if compact_missing_violations else missing_violation_count
    result = Evaluation(
        feasible=(len(violations) == 0 and missing_violation_count == 0),
        objective=objective,
        c1=c1,
        c2=c2,
        c3=c3,
        c4=c4,
        c5=c5,
        truck_distance=truck_distance,
        drone_distance=drone_distance,
        carbon_emissions=carbon_emissions,
        total_tardiness=total_tardiness,
        late_customers=late_customers,
        violations=violations,
        missing_violation_customers=missing_violation_customers,
        non_missing_violation_count=len(violations) - materialized_missing_count,
        truck_arrival={} if compact_missing_violations else truck_arrival,
        truck_departure={} if compact_missing_violations else truck_departure,
        truck_wait={} if compact_missing_violations else dict(truck_wait),
        truck_tardiness={} if compact_missing_violations else dict(truck_tard),
        drone_tardiness={} if compact_missing_violations else dict(drone_tard),
        drone_customer_service_time={} if compact_missing_violations else drone_service_time,
        drone_launch_time={} if compact_missing_violations else drone_launch_time,
        drone_recovery_time={} if compact_missing_violations else drone_recovery_time,
        customer_mode={} if compact_missing_violations else modes,
        fingerprint=(
            fingerprint_solution(sol)
            if not compact_missing_violations and not violations and missing_violation_count == 0
            else ""
        ),
    )
    if cache_key is not None:
        if len(data._evaluation_cache) >= cache_size:
            data._evaluation_cache.popitem(last=False)
        data._evaluation_cache[cache_key] = result
    return result


def evaluation_feasible_with_allowed_missing(ev: Evaluation, allowed_missing: Optional[Set[int]] = None) -> bool:
    """Treat an incomplete repair state as feasible except for customers still waiting to be reinserted."""
    if ev.feasible:
        return True
    if ev.non_missing_violation_count > 0 or not allowed_missing:
        return False
    return ev.missing_violation_customers.issubset(allowed_missing)


def nearest_neighbor_route_for_customers(
        data: ProblemData,
        customer_ids: Iterable[int],
        allow_wait: bool = True,
) -> List[int]:
    current = data.depot_start_id
    unvisited = set(customer_ids)
    route = [data.depot_start_id]
    cur_t = 0.0

    while unvisited:
        feasible: List[Tuple[float, int]] = []
        fallback: List[Tuple[float, int]] = []
        for c in unvisited:
            tt = data.truck_tt[(current, c)]
            arr = cur_t + tt
            fallback.append((tt, c))
            within_window = (
                max(arr, data.customers[c].ready) <= data.customers[c].due
                if allow_wait
                else data.customers[c].ready <= arr <= data.customers[c].due
            )
            if within_window:
                feasible.append((tt, c))
        if feasible:
            feasible.sort()
            _, nxt = feasible[0]
        else:
            fallback.sort()
            _, nxt = fallback[0]
        route.append(nxt)
        cur_t += data.truck_tt[(current, nxt)]
        if cur_t < data.customers[nxt].ready:
            cur_t = data.customers[nxt].ready
        current = nxt
        unvisited.remove(nxt)

    route.append(data.depot_end_id)
    return route


def nearest_neighbor_route_all_customers(data: ProblemData) -> List[int]:
    return nearest_neighbor_route_for_customers(data, data.C)


def historical_nearest_neighbor_route_all_customers(data: ProblemData) -> List[int]:
    """Retain the former deterministic backbone as one portfolio candidate."""
    return nearest_neighbor_route_for_customers(data, data.C, allow_wait=False)


def initial_truck_backbone(data: ProblemData, strategy: str) -> List[int]:
    required = list(data.Nk)
    if strategy == "historical":
        route_all = historical_nearest_neighbor_route_all_customers(data)
        return [data.depot_start_id] + [
            c for c in route_all[1:-1] if c in data.Nk
        ] + [data.depot_end_id]
    if strategy == "legacy":
        route_all = nearest_neighbor_route_all_customers(data)
        return [data.depot_start_id] + [
            c for c in route_all[1:-1] if c in data.Nk
        ] + [data.depot_end_id]
    if strategy == "nearest":
        return nearest_neighbor_route_for_customers(data, required)
    if strategy == "due":
        ordered = sorted(required, key=lambda c: (data.customers[c].due, data.customers[c].ready, c))
    elif strategy == "sweep":
        depot_x, depot_y = data.node_xy[data.depot_start_id]
        ordered = sorted(
            required,
            key=lambda c: (
                math.atan2(data.customers[c].y - depot_y, data.customers[c].x - depot_x),
                data.truck_dist[(data.depot_start_id, c)],
                c,
            ),
        )
    else:
        raise ValueError(f"Unknown initial truck strategy: {strategy}")
    return [data.depot_start_id, *ordered, data.depot_end_id]


def best_truck_insert_pos_by_distance(route: List[int], customer: int, data: ProblemData) -> Tuple[int, float]:
    best_pos, best_delta = 1, float("inf")
    for p in range(1, len(route)):
        a = route[p - 1]
        b = route[p]
        delta = data.truck_dist[(a, customer)] + data.truck_dist[(customer, b)] - data.truck_dist[(a, b)]
        if delta < best_delta:
            best_delta = delta
            best_pos = p
    return best_pos, best_delta


def sorted_two_nearest_route_nodes(route: List[int], customer: int, data: ProblemData) -> List[Tuple[int, int, float]]:
    pos = route_positions(route)
    out: List[Tuple[int, int, float]] = []
    for i in route:
        for j in route:
            if i == j:
                continue
            if pos[i] < pos[j]:
                out.append((i, j, data.drone_dist[(i, customer)] + data.drone_dist[(customer, j)]))
    out.sort(key=lambda x: x[2])
    return out


def candidate_drones_for_new_sortie(sol: Solution, data: ProblemData) -> List[int]:
    """Keep all active drones and one representative from the symmetric unused drones."""
    first_unused = next((u for u in data.U if not sol.sorties_by_drone.get(u)), None)
    return [
        u
        for u in data.U
        if sol.sorties_by_drone.get(u) or u == first_unused
    ]


def endpoint_candidate_pairs(
        route: List[int],
        customer: int,
        data: ProblemData,
) -> List[Tuple[int, int, float]]:
    key = (tuple(route), customer)
    cached = data._endpoint_candidate_cache.get(key)
    if cached is not None:
        data._endpoint_candidate_cache.move_to_end(key)
        return list(cached)

    pairs = sorted_two_nearest_route_nodes(route, customer, data)
    if not pairs:
        return []

    cfg = data.cfg
    route_adaptive_budget = int(math.ceil(cfg.endpoint_candidate_route_factor * len(route)))
    limit = min(
        len(pairs),
        max(
            int(cfg.endpoint_candidate_min),
            min(int(cfg.endpoint_candidate_max), route_adaptive_budget),
        ),
    )
    distance_count = max(1, min(limit, int(math.ceil(limit * cfg.endpoint_distance_share))))
    span_count = max(0, min(limit - distance_count, int(math.ceil(limit * cfg.endpoint_span_share))))
    time_count = max(0, limit - distance_count - span_count)
    pos = route_positions(route)
    target = data.customers[customer]

    # Truck-only timing estimate used only to rank endpoint candidates. Under
    # arrival-based launch timing, the launch anchor is the estimated arrival
    # at the node, not that node's customer-ready time.
    approximate_arrival: Dict[int, float] = {}
    approximate_departure = 0.0
    if route:
        approximate_arrival[route[0]] = 0.0
        if route[0] in data.customers:
            approximate_departure = max(0.0, data.customers[route[0]].ready)
        for previous, node in zip(route[:-1], route[1:]):
            arrival = approximate_departure + data.truck_tt[(previous, node)]
            approximate_arrival[node] = arrival
            approximate_departure = (
                max(arrival, data.customers[node].ready)
                if node in data.customers
                else arrival
            )

    selected: Set[Tuple[int, int]] = set()
    ordered_selected: List[Tuple[int, int, float]] = []

    def add_pair(item: Tuple[int, int, float]) -> None:
        key_pair = (item[0], item[1])
        if key_pair not in selected:
            selected.add(key_pair)
            ordered_selected.append(item)

    for item in pairs[:distance_count]:
        add_pair(item)
    if span_count > 0:
        by_span = sorted(
            pairs,
            key=lambda item: (pos[item[1]] - pos[item[0]], item[2], pos[item[0]]),
        )
        for item in by_span:
            add_pair(item)
            if len(selected) >= distance_count + span_count:
                break
    if time_count > 0:
        def time_alignment(item: Tuple[int, int, float]) -> Tuple[float, float, int]:
            launch, recovery, distance = item
            if cfg.allow_launch_during_truck_ready_wait:
                launch_anchor = approximate_arrival.get(launch, 0.0)
                recovery_anchor = approximate_arrival.get(
                    recovery,
                    target.due + data.drone_tt[(customer, recovery)],
                )
            else:
                launch_anchor = data.customers[launch].ready if launch in data.customers else 0.0
                recovery_anchor = (
                    data.customers[recovery].due
                    if recovery in data.customers
                    else target.due + data.drone_tt[(customer, recovery)]
                )
            expected_service = launch_anchor + data.drone_tt[(launch, customer)]
            expected_recovery = max(expected_service, target.ready) + data.drone_tt[(customer, recovery)]
            mismatch = (
                abs(expected_service - target.ready)
                + max(0.0, expected_service - target.due)
                + abs(expected_recovery - recovery_anchor)
            )
            return mismatch, distance, pos[launch]

        by_time = sorted(pairs, key=time_alignment)
        target_count = min(limit, len(selected) + time_count)
        for item in by_time:
            add_pair(item)
            if len(selected) >= target_count:
                break

    if len(selected) < limit:
        for item in pairs:
            add_pair(item)
            if len(selected) >= limit:
                break

    result = ordered_selected[:limit]
    cache_size = max(0, int(cfg.endpoint_candidate_cache_size))
    if cache_size > 0:
        if len(data._endpoint_candidate_cache) >= cache_size:
            data._endpoint_candidate_cache.popitem(last=False)
        data._endpoint_candidate_cache[key] = tuple(result)
    return result


def insert_sortie_by_order(sol: Solution, sortie: Sortie) -> None:
    pos = route_positions(sol.truck_route)
    lst = sol.sorties_by_drone.setdefault(sortie.drone, [])
    lst.append(sortie)
    lst.sort(key=lambda s: (pos.get(s.launch, 10 ** 9), pos.get(s.recovery, 10 ** 9), s.index))
    for idx, s in enumerate(lst, start=1):
        s.index = idx


def try_insert_customer_into_existing_sortie(
        sol: Solution,
        customer: int,
        data: ProblemData,
        metric: str,
        allowed_missing: Optional[Set[int]] = None,
        excluded_sortie: Optional[Tuple[int, int]] = None,
) -> Optional[Solution]:
    best_sol = None
    best_val = float("inf")
    allowed_after = set(allowed_missing or set())
    allowed_after.discard(customer)
    base = evaluate_solution(data, sol, compact_missing_violations=True)
    for u in data.U:
        for s in sol.sorties_by_drone.get(u, []):
            if excluded_sortie is not None and (u, s.index) == excluded_sortie:
                continue
            for k in range(len(s.customers) + 1):
                temp_customers = list(s.customers)
                temp_customers.insert(k, customer)
                temp_sortie = Sortie(u, s.index, s.launch, s.recovery, temp_customers)
                if not is_sortie_feasible_fast(temp_sortie, data):
                    continue

                s.customers.insert(k, customer)  # Do
                ev = evaluate_solution(data, sol, compact_missing_violations=True)
                if evaluation_feasible_with_allowed_missing(ev, allowed_after):
                    if metric == "distance":
                        prev_n = s.launch if k == 0 else s.customers[k - 1]
                        next_n = s.recovery if k == len(s.customers) - 1 else s.customers[k + 1]
                        val = data.drone_dist[(prev_n, customer)] + data.drone_dist[(customer, next_n)] - \
                              data.drone_dist[(prev_n, next_n)]
                    else:
                        val = ev.objective - base.objective
                    if val < best_val:
                        best_val = val
                        best_sol = sol.clone()
                s.customers.pop(k)  # Undo

    return best_sol


def try_create_new_sortie_for_customer(
        sol: Solution,
        customer: int,
        data: ProblemData,
        metric: str,
        allowed_missing: Optional[Set[int]] = None,
) -> Optional[Solution]:
    pairs = endpoint_candidate_pairs(sol.truck_route, customer, data)
    if metric == "distance":
        pairs = sorted_two_nearest_route_nodes(sol.truck_route, customer, data)[:len(pairs)]
    base = evaluate_solution(data, sol, compact_missing_violations=True)
    best_sol = None
    best_val = float("inf")
    allowed_after = set(allowed_missing or set())
    allowed_after.discard(customer)
    candidate_drones = candidate_drones_for_new_sortie(sol, data)

    for launch, recovery, pair_score in pairs:
        for u in candidate_drones:
            new_s = Sortie(drone=u, index=999, launch=launch, recovery=recovery, customers=[customer])
            if not is_sortie_feasible_fast(new_s, data):
                continue

            sol.sorties_by_drone[u].append(new_s)  # Do
            ev = evaluate_solution(data, sol, compact_missing_violations=True)
            if evaluation_feasible_with_allowed_missing(ev, allowed_after):
                if metric == "distance":
                    val = pair_score
                else:
                    val = ev.objective - base.objective
                if val < best_val:
                    best_val = val
                    best_sol = sol.clone()

            sol.sorties_by_drone[u].remove(new_s)  # Undo

        if metric == "distance" and best_sol is not None:
            break
    return best_sol


def _create_initial_solution_by_order(
        data: ProblemData,
        unassigned_order: List[int],
        truck_strategy: str = "legacy",
) -> Solution:
    route = initial_truck_backbone(data, truck_strategy)
    sol = Solution(truck_route=route, sorties_by_drone={u: [] for u in data.U})
    unassigned = list(unassigned_order)

    while unassigned:
        progress_this_round = False
        failed_customers = []

        for c in list(unassigned):
            allowed_after = set(unassigned) - {c}
            cand1 = try_insert_customer_into_existing_sortie(sol, c, data, metric="distance",
                                                             allowed_missing=allowed_after)
            if cand1 is not None:
                sol = cand1
                unassigned.remove(c)
                progress_this_round = True
                continue

            cand2 = try_create_new_sortie_for_customer(sol, c, data, metric="distance",
                                                       allowed_missing=allowed_after)
            if cand2 is not None:
                sol = cand2
                unassigned.remove(c)
                progress_this_round = True
                continue

            failed_customers.append(c)

        if not progress_this_round and failed_customers:
            hard: List[Tuple[float, int]] = []
            for c in failed_customers:
                pairs = endpoint_candidate_pairs(sol.truck_route, c, data)
                score = pairs[0][2] if pairs else 0.0
                hard.append((score, c))
            hard.sort(reverse=True)

            _, pick = hard[0]
            p, _ = best_truck_insert_pos_by_distance(sol.truck_route, pick, data)
            sol.truck_route.insert(p, pick)
            unassigned.remove(pick)

    sol.normalize(route_positions(sol.truck_route))
    return sol


def create_initial_solution(data: ProblemData, rng: random.Random) -> Solution:
    drone_customers = list(data.Nku)
    depot_x, depot_y = data.node_xy[data.depot_start_id]
    random_order = list(drone_customers)
    rng.shuffle(random_order)
    portfolio = [
        ("historical", sorted(drone_customers)),
        ("legacy", sorted(drone_customers)),
        (
            "nearest",
            sorted(drone_customers, key=lambda c: (data.customers[c].due, data.customers[c].ready, c)),
        ),
        (
            "due",
            sorted(
                drone_customers,
                key=lambda c: (
                    math.atan2(data.customers[c].y - depot_y, data.customers[c].x - depot_x),
                    data.drone_dist[(data.depot_start_id, c)],
                    c,
                ),
            ),
        ),
        ("sweep", random_order),
    ]

    candidates: List[Tuple[float, Solution]] = []
    limit = min(len(portfolio), max(1, int(data.cfg.initial_portfolio_size)))
    for truck_strategy, order in portfolio[:limit]:
        candidate = _create_initial_solution_by_order(data, order, truck_strategy=truck_strategy)
        candidate_eval = evaluate_solution(data, candidate)
        if candidate_eval.feasible:
            candidates.append((candidate_eval.objective, candidate))
    if not candidates:
        return _create_initial_solution_by_order(data, sorted(drone_customers), truck_strategy="legacy")
    return min(candidates, key=lambda item: item[0])[1]


def remove_sortie(sol: Solution, drone: int, index: int) -> List[int]:
    removed: List[int] = []
    keep: List[Sortie] = []
    for s in sol.sorties_by_drone.get(drone, []):
        if s.index == index:
            removed.extend(s.customers)
        else:
            keep.append(s)
    for idx, s in enumerate(keep, start=1):
        s.index = idx
    sol.sorties_by_drone[drone] = keep
    return removed


def remove_customers_from_solution(sol: Solution, customers: Iterable[int], cascade_if_launch_recovery: bool) -> Set[
    int]:
    to_remove = set(customers)
    removed: Set[int] = set()
    if not to_remove:
        return removed

    keep_route: List[int] = []
    for node in sol.truck_route:
        if node in to_remove and node not in {sol.truck_route[0], sol.truck_route[-1]}:
            removed.add(node)
            continue
        keep_route.append(node)
    sol.truck_route = keep_route

    for u in list(sol.sorties_by_drone.keys()):
        keep_sorties: List[Sortie] = []
        for s in sol.sorties_by_drone[u]:
            if cascade_if_launch_recovery and (s.launch in to_remove or s.recovery in to_remove):
                removed.update(s.customers)
                continue
            new_customers = [c for c in s.customers if c not in to_remove]
            removed.update(set(s.customers) - set(new_customers))
            s.customers = new_customers
            if s.customers:
                keep_sorties.append(s)
        for idx, s in enumerate(keep_sorties, start=1):
            s.index = idx
        sol.sorties_by_drone[u] = keep_sorties

    removed.update(to_remove)
    return removed


def try_best_total_insertion(
        sol: Solution,
        customer: int,
        data: ProblemData,
        allowed_missing: Optional[Set[int]] = None,
) -> Optional[Solution]:
    best_sol = None
    best_objective = float("inf")
    allowed_after = set(allowed_missing or set())
    allowed_after.discard(customer)

    for p in range(1, len(sol.truck_route)):
        sol.truck_route.insert(p, customer)
        ev = evaluate_solution(data, sol, compact_missing_violations=True)
        if evaluation_feasible_with_allowed_missing(ev, allowed_after):
            if ev.objective < best_objective:
                best_objective = ev.objective
                best_sol = sol.clone()
        sol.truck_route.pop(p)  # Undo

    for u in data.U:
        for s in sol.sorties_by_drone.get(u, []):
            for k in range(len(s.customers) + 1):
                temp_customers = list(s.customers)
                temp_customers.insert(k, customer)
                temp_sortie = Sortie(u, s.index, s.launch, s.recovery, temp_customers)
                if not is_sortie_feasible_fast(temp_sortie, data):
                    continue

                s.customers.insert(k, customer)
                ev = evaluate_solution(data, sol, compact_missing_violations=True)
                if evaluation_feasible_with_allowed_missing(ev, allowed_after):
                    if ev.objective < best_objective:
                        best_objective = ev.objective
                        best_sol = sol.clone()
                s.customers.pop(k)  # Undo

    pairs = endpoint_candidate_pairs(sol.truck_route, customer, data)
    candidate_drones = candidate_drones_for_new_sortie(sol, data)

    for launch, recovery, _ in pairs:
        for u in candidate_drones:
            new_s = Sortie(u, 999, launch, recovery, [customer])
            if not is_sortie_feasible_fast(new_s, data):
                continue

            sol.sorties_by_drone[u].append(new_s)
            ev = evaluate_solution(data, sol, compact_missing_violations=True)
            if evaluation_feasible_with_allowed_missing(ev, allowed_after):
                if ev.objective < best_objective:
                    best_objective = ev.objective
                    best_sol = sol.clone()
            sol.sorties_by_drone[u].remove(new_s)  # Undo

    return best_sol


def try_best_truck_distance_insertion(sol: Solution, customer: int, data: ProblemData) -> Solution:
    cand = sol.clone()
    p, _ = best_truck_insert_pos_by_distance(cand.truck_route, customer, data)
    cand.truck_route.insert(p, customer)
    cand.normalize(route_positions(cand.truck_route))
    return cand


def try_random_truck_insertion(sol: Solution, customer: int, rng: random.Random) -> Solution:
    cand = sol.clone()
    p = rng.randint(1, len(cand.truck_route) - 1)
    cand.truck_route.insert(p, customer)
    cand.normalize(route_positions(cand.truck_route))
    return cand


def relatedness_score(
        i: int,
        j: int,
        sol: Solution,
        data: ProblemData,
        route_pos: Optional[Dict[int, int]] = None,
        sortie_group: Optional[Dict[int, Tuple[int, int]]] = None,
) -> float:
    max_d = getattr(data, "max_customer_truck_dist", 1.0)
    d_norm = data.truck_dist[(i, j)] / max(max_d, 1e-9)
    pos = route_pos if route_pos is not None else route_positions(sol.truck_route)
    ci = data.customers[i]
    cj = data.customers[j]
    time_scale = max(data.ready_span, data.due_span, 1.0)
    time_norm = (
        abs(ci.ready - cj.ready) + abs(ci.due - cj.due)
    ) / (2.0 * time_scale)
    demand_scale = max(
        data.cfg.drone_capacity,
        data.max_customer_demand,
        1.0,
    )
    demand_norm = abs(ci.demand - cj.demand) / demand_scale

    group = sortie_group or {}
    gi = group.get(i)
    gj = group.get(j)
    both_truck = i in pos and j in pos
    both_drone = gi is not None and gj is not None
    same_sortie = gi is not None and gi == gj
    mode_difference = 0.0 if both_truck or both_drone else 1.0
    route_separation = (
        abs(pos[i] - pos[j]) / max(1, len(sol.truck_route) - 1)
        if both_truck
        else 0.0
    )

    cfg = data.cfg
    dissimilarity = (
        cfg.shaw_distance_weight * d_norm
        + cfg.shaw_time_weight * time_norm
        + cfg.shaw_demand_weight * demand_norm
        + cfg.shaw_route_weight * route_separation
        + cfg.shaw_mode_weight * mode_difference
        - cfg.shaw_same_sortie_bonus * float(same_sortie)
    )
    return -dissimilarity


def shaw_random_pick_probability(data: ProblemData) -> float:
    return min(1.0, max(0.0, data.cfg.shaw_random_pick_probability))


def choose_regret_customer(sol: Solution, remaining: List[int], data: ProblemData, regret_k: int) -> Optional[
    Tuple[int, Solution]]:
    best_customer = None
    best_next = None
    best_regret = -float("inf")
    best_priority = None
    best_move = None
    candidate_drones = candidate_drones_for_new_sortie(sol, data)
    cfg = data.cfg
    bound_enabled = cfg.experiment_regret_exact_bound and not cfg.experiment_standard_regret and all(
        value >= 0 for value in (cfg.truck_fixed_cost, cfg.drone_fixed_cost, cfg.truck_unit_cost_per_hour,
                                 cfg.drone_unit_cost_per_hour, cfg.wait_cost_per_hour,
                                 cfg.tardiness_cost_per_hour, cfg.carbon_price_per_kg,
                                 cfg.truck_emission_kg_per_km, cfg.drone_emission_kg_per_km))

    def provably_dominated(option_costs: List[float]) -> bool:
        if not bound_enabled or len(option_costs) < regret_k:
            return False
        cutoff = heapq.nsmallest(regret_k, option_costs)[-1]
        data._regret_bound_checked += 1
        bound = regret_insertion_lower_bound(sol, data)
        # Keep near ties: summation order and floating-point rounding are not a pruning criterion.
        dominated = bound > cutoff + 1e-7 * max(1.0, abs(cutoff), abs(bound))
        data._regret_bound_pruned += int(dominated)
        return dominated

    for c in remaining:
        option_costs: List[float] = []
        best_option_obj = float("inf")
        best_option_sol: Optional[Solution] = None
        best_option_move = None
        allowed_after = set(remaining) - {c}

        for p in range(1, len(sol.truck_route)):
            sol.truck_route.insert(p, c)
            if provably_dominated(option_costs):
                sol.truck_route.pop(p)
                continue
            ev = evaluate_solution(data, sol, compact_missing_violations=True)
            if evaluation_feasible_with_allowed_missing(ev, allowed_after):
                option_costs.append(ev.objective)
                if ev.objective < best_option_obj:
                    best_option_obj = ev.objective
                    best_option_move = ('truck', p, c)
                    if not data.cfg.experiment_deferred_regret_copy:
                        best_option_sol = sol.clone()
            sol.truck_route.pop(p)  # Undo

        for u in data.U:
            for s in sol.sorties_by_drone.get(u, []):
                for k in range(len(s.customers) + 1):
                    temp_customers = list(s.customers)
                    temp_customers.insert(k, c)
                    temp_s = Sortie(u, s.index, s.launch, s.recovery, temp_customers)
                    if not is_sortie_feasible_fast(temp_s, data):
                        continue

                    s.customers.insert(k, c)
                    if provably_dominated(option_costs):
                        s.customers.pop(k)
                        continue
                    ev = evaluate_solution(data, sol, compact_missing_violations=True)
                    if evaluation_feasible_with_allowed_missing(ev, allowed_after):
                        option_costs.append(ev.objective)
                        if ev.objective < best_option_obj:
                            best_option_obj = ev.objective
                            best_option_move = ('sortie', u, id(s), k, c)
                            if not data.cfg.experiment_deferred_regret_copy:
                                best_option_sol = sol.clone()
                    s.customers.pop(k)  # Undo

        pairs = endpoint_candidate_pairs(sol.truck_route, c, data)
        regret_limit = max(regret_k, int(data.cfg.regret_endpoint_candidate_limit))
        pairs = pairs[:regret_limit]

        for launch, recovery, _ in pairs:
            for u in candidate_drones:
                new_s = Sortie(u, 999, launch, recovery, [c])
                if not is_sortie_feasible_fast(new_s, data):
                    continue

                sol.sorties_by_drone[u].append(new_s)
                if provably_dominated(option_costs):
                    sol.sorties_by_drone[u].remove(new_s)
                    continue
                ev = evaluate_solution(data, sol, compact_missing_violations=True)
                if evaluation_feasible_with_allowed_missing(ev, allowed_after):
                    option_costs.append(ev.objective)
                    if ev.objective < best_option_obj:
                        best_option_obj = ev.objective
                        best_option_move = ('new', u, launch, recovery, c)
                        if not data.cfg.experiment_deferred_regret_copy:
                            best_option_sol = sol.clone()
                sol.sorties_by_drone[u].remove(new_s)  # Undo

        if not option_costs or best_option_move is None:
            continue
        smallest = heapq.nsmallest(min(regret_k, len(option_costs)), option_costs)
        regret = smallest[-1] - smallest[0]
        if data.cfg.experiment_standard_regret:
            regret = sum(value - smallest[0] for value in smallest[1:])
            priority = (int(len(option_costs) < regret_k), -len(option_costs) if len(option_costs) < regret_k else 0, regret, -smallest[0])
        else:
            priority = (regret,)
        if best_priority is None or priority > best_priority:
            best_regret = regret
            best_priority = priority
            best_customer = c
            best_next = best_option_sol
            best_move = best_option_move

    if best_customer is None or best_move is None:
        return None
    if data.cfg.experiment_deferred_regret_copy:
        best_next = sol.clone()
        if best_move[0] == 'truck':
            best_next.truck_route.insert(best_move[1], best_move[2])
        elif best_move[0] == 'sortie':
            _, u, identity, position, customer = best_move
            index = next(i for i, s in enumerate(sol.sorties_by_drone[u]) if id(s) == identity)
            best_next.sorties_by_drone[u][index].customers.insert(position, customer)
        else:
            _, u, launch, recovery, customer = best_move
            best_next.sorties_by_drone[u].append(Sortie(u, 999, launch, recovery, [customer]))
    return best_customer, best_next


def regret_insertion_lower_bound(sol: Solution, data: ProblemData) -> float:
    """Travel/fixed/carbon costs plus unavoidable truck lateness, ignoring all drone delays."""
    cfg = data.cfg
    time_matrix, distance_matrix = data.truck_tt_dense, data.truck_dist_dense
    truck_time = truck_distance = earliest = unavoidable_tardy = 0.0
    for i, j in zip(sol.truck_route[:-1], sol.truck_route[1:]):
        leg = time_matrix[i][j] if time_matrix is not None else data.truck_tt[(i, j)]
        truck_time += leg
        truck_distance += distance_matrix[i][j] if distance_matrix is not None else data.truck_dist[(i, j)]
        earliest += leg
        if j in data.customers:
            customer = data.customers[j]
            earliest = max(earliest, customer.ready)
            unavoidable_tardy += max(0.0, earliest - customer.due)
    active = [u for u in data.U if sol.sorties_by_drone.get(u)]
    sorties = [s for u in data.U for s in sol.sorties_by_drone.get(u, [])]
    drone_time = sum(sortie_flight_time(s, data) for s in sorties)
    drone_distance = sum(sortie_flight_distance(s, data) for s in sorties)
    return (cfg.truck_fixed_cost + cfg.drone_fixed_cost * len(active)
            + cfg.truck_unit_cost_per_hour * truck_time + cfg.drone_unit_cost_per_hour * drone_time
            + cfg.carbon_price_per_kg * (cfg.truck_emission_kg_per_km * truck_distance
                                       + cfg.drone_emission_kg_per_km * drone_distance)
            + cfg.tardiness_cost_per_hour * unavoidable_tardy)


def _large_destroy_set(
        sol: Solution,
        data: ProblemData,
        rng: random.Random,
        k: int,
        strategy: int,
) -> List[int]:
    """Choose a mixed truck/drone removal set for one ALNS restart."""
    all_customers = list(data.C)
    k = min(max(1, k), len(all_customers))
    if strategy == 0:
        return rng.sample(all_customers, k)

    if strategy == 1:
        seed = rng.choice(all_customers)
        pos = route_positions(sol.truck_route)
        sortie_group: Dict[int, Tuple[int, int]] = {}
        for u in data.U:
            for sortie in sol.sorties_by_drone.get(u, []):
                for customer in sortie.customers:
                    sortie_group[customer] = (u, sortie.index)
        related = sorted(
            (
                (relatedness_score(seed, c, sol, data, pos, sortie_group), c)
                for c in all_customers if c != seed
            ),
            reverse=True,
        )
        selected = [seed]
        while related and len(selected) < k:
            if rng.random() < shaw_random_pick_probability(data):
                rank = rng.randrange(len(related))
            else:
                rank = min(
                    int((rng.random() ** data.cfg.m) * len(related)),
                    len(related) - 1,
                )
            _, customer = related.pop(rank)
            selected.append(customer)
        return selected

    truck_order = [c for c in sol.truck_route if c in data.C]
    selected: List[int] = []
    if truck_order:
        block_k = min(len(truck_order), max(1, int(round(k * 0.65))))
        start = rng.randint(0, len(truck_order) - block_k)
        selected.extend(truck_order[start:start + block_k])
    remaining = [c for c in all_customers if c not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[:max(0, k - len(selected))])
    return selected[:k]


def _repair_large_destroy(
        partial: Solution,
        missing: List[int],
        data: ProblemData,
        rng: random.Random,
        repair_strategy: int,
) -> Optional[Solution]:
    cand = partial
    remaining = list(missing)
    if repair_strategy == 0:
        rng.shuffle(remaining)
        while remaining:
            customer = remaining.pop(0)
            nxt = try_best_total_insertion(cand, customer, data, allowed_missing=set(remaining))
            if nxt is None:
                return None
            cand = nxt
    elif repair_strategy == 1:
        regret_k = 2 if len(remaining) < 10 else 3
        while remaining:
            picked = choose_regret_customer(cand, remaining, data, regret_k)
            if picked is None:
                return None
            customer, cand = picked
            remaining.remove(customer)
    else:
        remaining.sort(key=lambda c: (data.customers[c].due, rng.random()))
        while remaining:
            customer = remaining.pop(0)
            nxt = try_best_total_insertion(cand, customer, data, allowed_missing=set(remaining))
            if nxt is None:
                return None
            cand = nxt

    cand.normalize(route_positions(cand.truck_route))
    return cand


def large_destroy_reconstruct(
        sol: Solution,
        data: ProblemData,
        rng: random.Random,
        intensity: float = 0.40,
        attempts: int = 2,
        strategy_offset: int = 0,
        prefer_best: bool = False,
) -> Optional[Solution]:
    """Adaptive ALNS destroy-repair restart using a small strategy portfolio."""
    if not data.C:
        return None

    feasible_candidates: List[Tuple[float, Solution]] = []
    for attempt in range(max(1, attempts)):
        pct = min(0.80, max(0.05, intensity * rng.uniform(0.88, 1.12)))
        k = min(len(data.C), max(2, int(round(len(data.C) * pct))))
        remove_list = _large_destroy_set(sol, data, rng, k, (strategy_offset + attempt) % 3)
        partial = sol.clone()
        removed = remove_customers_from_solution(partial, remove_list, cascade_if_launch_recovery=True)
        missing = [c for c in removed if c in data.C]
        rebuilt = _repair_large_destroy(partial, missing, data, rng, (strategy_offset + attempt) % 3)
        if rebuilt is None:
            continue
        rebuilt_eval = evaluate_solution(data, rebuilt)
        if rebuilt_eval.feasible:
            feasible_candidates.append((rebuilt_eval.objective, rebuilt))

    if not feasible_candidates:
        return None
    feasible_candidates.sort(key=lambda item: item[0])
    if prefer_best:
        return feasible_candidates[0][1]
    rcl_size = min(2, len(feasible_candidates))
    return feasible_candidates[rng.randrange(rcl_size)][1]


def reheat_pulse_active(progress: float, remaining: int, cfg: Config, horizon: int) -> bool:
    if not cfg.experiment_honor_reheat_pulse or remaining <= 0:
        return False
    pulse_end = min(1.0, cfg.adaptive_reheat_stop_ratio + cfg.adaptive_destroy_pulse / max(1, horizon))
    return progress < pulse_end


class ALNSOptimizer:
    def __init__(self, data: ProblemData, seed: Optional[int] = None) -> None:
        self.data = data
        self.cfg = data.cfg
        self.seed = seed if seed is not None else (time.time_ns() & 0xFFFFFFFF)
        self.rng = random.Random(self.seed)
        self.tabu: Deque[str] = deque(maxlen=self.cfg.tabu_len)
        self.history_obj: List[float] = []
        self.best_update_log: List[Tuple[int, float]] = []
        self.reconstruct_intensification_wins: Dict[str, int] = {
            "tardiness_reinsert": 0,
            "time_pressure_2opt": 0,
            "drone_wait_rebuild": 0,
            "truck_to_drone": 0,
            "endpoint_relocation": 0,
        }
        self.reconstruct_intensification_calls: Dict[str, int] = {
            name: 0 for name in self.reconstruct_intensification_wins
        }
        self.post_trigger_checks = 0
        self.post_trigger_skips = 0

        # OP21 is a focused soft-time-window destroy-repair neighborhood.
        self.operator_stats = [OperatorStat(name=f"OP{i}") for i in range(1, 22)]
        self.operator_funcs = [
            self.op1, self.op2, self.op3, self.op4, self.op5,
            self.op6, self.op7, self.op8, self.op9, self.op10,
            self.op11, self.op12, self.op13, self.op14, self.op15,
            self.op16, self.op17, self.op18, self.op19, self.op20,
            self.op21,
        ]
        if self.cfg.experiment_remove_op20:
            del self.operator_funcs[19]
            del self.operator_stats[19]
        if self.cfg.experiment_disable_structural_directed:
            retained = [(stat, func) for stat, func in zip(self.operator_stats, self.operator_funcs)
                        if int(stat.name[2:]) < 16]
            self.operator_stats = [stat for stat, _ in retained]
            self.operator_funcs = [func for _, func in retained]
            for operator_id in range(16, 22):
                def disabled(*args, **kwargs):
                    return None
                disabled.__name__ = f'op{operator_id}'
                setattr(self, disabled.__name__, disabled)

        self.strict_blocked_call_counts: Dict[str, int] = {}
        self.tabu_rule_stats: Dict[str, int] = {
            "historical_hits": 0, "current_hits": 0,
            "historical_bypasses": 0, "aspiration_bypasses": 0,
        }
        if self.cfg.experiment_disable_full_structural_group:
            retained = [(stat, func) for stat, func in zip(self.operator_stats, self.operator_funcs)
                        if int(stat.name[2:]) < 8]
            self.operator_stats = [stat for stat, _ in retained]
            self.operator_funcs = [func for _, func in retained]

            def block(name: str):
                def disabled(*args, **kwargs):
                    self.strict_blocked_call_counts[name] = self.strict_blocked_call_counts.get(name, 0) + 1
                    return None
                disabled.__name__ = name
                return disabled

            # Remove the same extended group as the historical pool-only test,
            # including equivalent neighborhoods entered directly by polishing.
            helpers = (
                "_best_endpoint_relocation", "_best_endpoint_pair_global",
                "_best_sortie_chain_merge", "_best_sortie_rebuild_global",
                "_best_drone_to_truck_global", "_best_mode_exchange_global",
            )
            for name in [*(f"op{i}" for i in range(8, 22)), *helpers]:
                setattr(self, name, block(name))

            def preserve_assignment(sol: Solution) -> Solution:
                name = "_compact_drone_assignments"
                self.strict_blocked_call_counts[name] = self.strict_blocked_call_counts.get(name, 0) + 1
                return sol.clone()

            self._compact_drone_assignments = preserve_assignment

    def _tabu_rejects(self, candidate: Evaluation, current: Evaluation, best: Evaluation) -> bool:
        if candidate.fingerprint not in self.tabu:
            return False
        is_current = candidate.fingerprint == current.fingerprint
        self.tabu_rule_stats["current_hits" if is_current else "historical_hits"] += 1
        if candidate.objective + 1e-9 < best.objective:
            self.tabu_rule_stats["aspiration_bypasses"] += 1
            return False
        # Isolate historical-memory rejection without changing repair/self-loop rules.
        if self.cfg.ablation_disable_history_tabu and not is_current:
            self.tabu_rule_stats["historical_bypasses"] += 1
            return False
        return True

    def _post_reconstruct_is_promising(self, candidate_cost: float, best_cost: float, progress: float) -> bool:
        mode = self.cfg.experiment_selective_post
        if mode == "always":
            return True
        self.post_trigger_checks += 1
        if mode == "elite":
            selected = candidate_cost <= best_cost + 1e-9
        elif mode == "adaptive":
            radius = self.cfg.sa_end_ratio + (self.cfg.sa_start_ratio - self.cfg.sa_end_ratio) * (1.0 - progress)
            radius *= max(0.0, self.cfg.experiment_post_radius_scale)
            selected = candidate_cost <= best_cost + max(1.0, best_cost) * radius
        else:
            raise ValueError(f"Unknown post-reconstruction mode: {mode}")
        self.post_trigger_skips += int(not selected)
        return selected

    def _record_operator_score(self, idx: int, score: float) -> None:
        self.operator_stats[idx].score_sum += score
        self.operator_stats[idx].use_count += 1

    def _update_weights(self) -> None:
        for s in self.operator_stats:
            if s.use_count == 0:
                continue
            avg = s.score_sum / s.use_count
            s.weight = (1.0 - self.cfg.alpha) * s.weight + self.cfg.alpha * avg
            s.score_sum = 0.0
            s.use_count = 0

    def _repair_if_needed(self, candidate: Solution, current: Solution) -> Solution:
        cand = candidate.clone()
        pos = route_positions(cand.truck_route)
        cand.normalize(pos)
        for u in self.data.U:
            for s in cand.sorties_by_drone.get(u, []):
                if s.launch in pos and s.recovery in pos and pos[s.launch] > pos[s.recovery]:
                    s.launch, s.recovery = s.recovery, s.launch
        pos = route_positions(cand.truck_route)
        for u in self.data.U:
            sorties = cand.sorties_by_drone.get(u, [])
            sorties.sort(key=lambda x: x.index)
            for a, b in zip(sorties[:-1], sorties[1:]):
                if a.recovery in pos and b.launch in pos and pos[a.recovery] > pos[b.launch]:
                    a.recovery, b.launch = b.launch, a.recovery
        cand.normalize(route_positions(cand.truck_route))
        if evaluate_solution(self.data, cand, compact_missing_violations=True).feasible:
            return cand
        if self.cfg.experiment_noop_failure:
            return candidate
        return current.clone()

    def _accept_by_sa(self, new_obj: float, cur_obj: float, temp: float) -> bool:
        if new_obj < cur_obj:
            return True
        if temp <= 1e-12:
            return False
        return self.rng.random() < math.exp(-(new_obj - cur_obj) / temp)

    def _remove_and_reinsert_total(self, base: Solution, remove_customers: List[int], random_order: bool) -> Optional[
        Solution]:
        cand = base.clone()
        removed = remove_customers_from_solution(cand, remove_customers, cascade_if_launch_recovery=True)
        missing = [c for c in removed if c in self.data.C]
        if random_order:
            self.rng.shuffle(missing)
        remaining = set(missing)
        for c in missing:
            remaining.discard(c)
            nxt = try_best_total_insertion(cand, c, self.data, allowed_missing=remaining)
            if nxt is None:
                return None
            cand = nxt
        cand.normalize(route_positions(cand.truck_route))
        return cand

    def _truck_served(self, ev: Evaluation) -> List[int]:
        return [c for c, m in ev.customer_mode.items() if m == "truck"]

    def _drone_served(self, ev: Evaluation) -> List[int]:
        return [c for c, m in ev.customer_mode.items() if m == "drone"]

    def _best_single_customer_reinsert(self, sol: Solution, ev: Evaluation) -> Optional[Solution]:
        best_sol = None
        best_obj = ev.objective
        blocked = launch_recovery_nodes(sol)

        for c in self.data.C:
            if c in blocked:
                continue

            cand = sol.clone()
            removed = remove_customers_from_solution(cand, [c], cascade_if_launch_recovery=False)
            if c not in removed:
                continue

            nxt = try_best_total_insertion(cand, c, self.data)
            if nxt is None:
                continue
            nxt.normalize(route_positions(nxt.truck_route))
            nxt_eval = evaluate_solution(self.data, nxt, compact_missing_violations=True)
            if nxt_eval.feasible and nxt_eval.objective + 1e-9 < best_obj:
                best_obj = nxt_eval.objective
                best_sol = nxt

        return best_sol

    def _best_endpoint_relocation(self, sol: Solution, ev: Evaluation) -> Optional[Solution]:
        best_sol = None
        best_obj = ev.objective
        route = sol.truck_route
        pos = route_positions(route)

        for u in self.data.U:
            for s in sol.sorties_by_drone.get(u, []):
                for side in ("launch", "recovery"):
                    other_endpoint = s.recovery if side == "launch" else s.launch
                    for node in route:
                        if node in {self.data.depot_start_id, self.data.depot_end_id, other_endpoint}:
                            continue
                        l_node = node if side == "launch" else s.launch
                        r_node = node if side == "recovery" else s.recovery
                        if pos.get(l_node, 10 ** 9) >= pos.get(r_node, -1):
                            continue
                        test_sortie = Sortie(u, s.index, l_node, r_node, list(s.customers))
                        if not is_sortie_feasible_fast(test_sortie, self.data):
                            continue

                        cand = sol.clone()
                        target = next((x for x in cand.sorties_by_drone.get(u, []) if x.index == s.index), None)
                        if target is None:
                            continue
                        if side == "launch":
                            target.launch = node
                        else:
                            target.recovery = node
                        cand.normalize(route_positions(cand.truck_route))
                        cand_eval = evaluate_solution(self.data, cand, compact_missing_violations=True)
                        if cand_eval.feasible and cand_eval.objective + 1e-9 < best_obj:
                            best_obj = cand_eval.objective
                            best_sol = cand

        return best_sol

    def _compact_drone_assignments(self, sol: Solution) -> Solution:
        """Pack non-overlapping sorties onto identical drones."""
        cand = sol.clone()
        pos = route_positions(cand.truck_route)
        sorties = [s.clone() for u in self.data.U for s in cand.sorties_by_drone.get(u, [])]
        sorties.sort(key=lambda s: (pos.get(s.launch, 10 ** 9), pos.get(s.recovery, 10 ** 9)))
        packed = {u: [] for u in self.data.U}
        last_recovery = {u: -1 for u in self.data.U}

        for sortie in sorties:
            start = pos.get(sortie.launch, 10 ** 9)
            available = [u for u in self.data.U if last_recovery[u] <= start]
            if not available:
                return cand
            drone = max(available, key=lambda u: last_recovery[u])
            sortie.drone = drone
            packed[drone].append(sortie)
            last_recovery[drone] = pos.get(sortie.recovery, 10 ** 9)

        cand.sorties_by_drone = packed
        cand.normalize(pos)
        return cand

    def _best_sortie_chain_merge(self, sol: Solution, ev: Evaluation) -> Optional[Solution]:
        """Merge adjacent sorties when recovery and relaunch use the same truck node."""
        best_sol = None
        best_obj = ev.objective
        pos = route_positions(sol.truck_route)

        for u in self.data.U:
            sorties = sorted(sol.sorties_by_drone.get(u, []), key=lambda s: pos.get(s.launch, 10 ** 9))
            for first, second in zip(sorties[:-1], sorties[1:]):
                if first.recovery != second.launch:
                    continue
                merged = Sortie(
                    drone=u,
                    index=first.index,
                    launch=first.launch,
                    recovery=second.recovery,
                    customers=list(first.customers) + list(second.customers),
                )
                if not is_sortie_feasible_fast(merged, self.data):
                    continue
                cand = sol.clone()
                cand.sorties_by_drone[u] = [
                    s for s in cand.sorties_by_drone.get(u, [])
                    if s.index not in {first.index, second.index}
                ]
                cand.sorties_by_drone[u].append(merged)
                cand.normalize(route_positions(cand.truck_route))
                cand_eval = evaluate_solution(self.data, cand, compact_missing_violations=True)
                if cand_eval.feasible and cand_eval.objective + 1e-9 < best_obj:
                    best_sol = cand
                    best_obj = cand_eval.objective
        return best_sol

    @staticmethod
    def _two_opt_routes(route: List[int]) -> Iterable[List[int]]:
        seen: Set[Tuple[int, ...]] = set()
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                candidate = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                key = tuple(candidate)
                if key not in seen:
                    seen.add(key)
                    yield candidate

    @staticmethod
    def _route_neighborhood(route: List[int]) -> Iterable[List[int]]:
        """Relocate, swap, 2-opt, and Or-opt(2) truck-route neighborhoods."""
        seen: Set[Tuple[int, ...]] = set()

        for i in range(1, len(route) - 1):
            for j in range(1, len(route) - 1):
                if i == j:
                    continue
                candidate = list(route)
                node = candidate.pop(i)
                candidate.insert(j, node)
                key = tuple(candidate)
                if key not in seen:
                    seen.add(key)
                    yield candidate

        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                swapped = list(route)
                swapped[i], swapped[j] = swapped[j], swapped[i]
                key = tuple(swapped)
                if key not in seen:
                    seen.add(key)
                    yield swapped

                reversed_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                key = tuple(reversed_route)
                if key not in seen:
                    seen.add(key)
                    yield reversed_route

        block_len = 2
        for i in range(1, len(route) - block_len):
            block = route[i:i + block_len]
            remainder = route[:i] + route[i + block_len:]
            for j in range(1, len(remainder)):
                candidate = remainder[:j] + block + remainder[j:]
                key = tuple(candidate)
                if key not in seen:
                    seen.add(key)
                    yield candidate

    def _best_route_move_global(self, sol: Solution, ev: Evaluation) -> Optional[Solution]:
        best_sol = None
        best_obj = ev.objective
        for route in self._route_neighborhood(sol.truck_route):
            cand = sol.clone()
            cand.truck_route = route
            cand = self._compact_drone_assignments(cand)
            cand_eval = evaluate_solution(self.data, cand, compact_missing_violations=True)
            if cand_eval.feasible and cand_eval.objective + 1e-9 < best_obj:
                best_sol = cand
                best_obj = cand_eval.objective
        return best_sol

    def _best_endpoint_pair_global(self, sol: Solution, ev: Evaluation) -> Optional[Solution]:
        """Relocate both launch and recovery endpoints in one move."""
        best_sol = None
        best_obj = ev.objective
        route = sol.truck_route
        candidate_limit = max(1, self.cfg.global_polish_candidate_limit)
        sortie_candidates: List[Tuple[float, int, int, Sortie]] = []
        for u in self.data.U:
            for sortie_pos, sortie in enumerate(sol.sorties_by_drone.get(u, [])):
                pressure = sum(ev.drone_tardiness.get(c, 0.0) for c in sortie.customers)
                pressure += ev.truck_tardiness.get(sortie.launch, 0.0)
                pressure += ev.truck_tardiness.get(sortie.recovery, 0.0)
                pressure += ev.truck_wait.get(sortie.recovery, 0.0)
                sortie_candidates.append((pressure, u, sortie_pos, sortie))

        sortie_candidates.sort(key=lambda item: item[0], reverse=True)
        for _, u, sortie_pos, sortie in sortie_candidates[:candidate_limit]:
            representative = max(
                sortie.customers,
                key=lambda c: ev.drone_tardiness.get(c, 0.0),
            )
            pairs = endpoint_candidate_pairs(route, representative, self.data)[:candidate_limit]
            for launch, recovery, _ in pairs:
                if launch == sortie.launch and recovery == sortie.recovery:
                    continue
                replacement = Sortie(
                    drone=u,
                    index=sortie.index,
                    launch=launch,
                    recovery=recovery,
                    customers=list(sortie.customers),
                )
                if not is_sortie_feasible_fast(replacement, self.data):
                    continue
                cand = sol.clone()
                cand.sorties_by_drone[u][sortie_pos] = replacement
                cand = self._compact_drone_assignments(cand)
                cand_eval = evaluate_solution(self.data, cand, compact_missing_violations=True)
                if cand_eval.feasible and cand_eval.objective + 1e-9 < best_obj:
                    best_sol = cand
                    best_obj = cand_eval.objective
        return best_sol

    def _best_sortie_rebuild_global(self, sol: Solution, ev: Evaluation) -> Optional[Solution]:
        """Jointly reorder a sortie's customers and rebuild both endpoints."""
        best_sol = None
        best_obj = ev.objective
        route = sol.truck_route
        max_customers = max(2, self.cfg.global_sortie_permutation_max_customers)
        candidate_limit = max(1, self.cfg.global_polish_candidate_limit)
        sortie_candidates: List[Tuple[float, int, int, Sortie]] = []
        for u in self.data.U:
            for sortie_pos, sortie in enumerate(sol.sorties_by_drone.get(u, [])):
                if 2 <= len(sortie.customers) <= max_customers:
                    pressure = sum(ev.drone_tardiness.get(c, 0.0) for c in sortie.customers)
                    pressure += ev.truck_wait.get(sortie.recovery, 0.0)
                    sortie_candidates.append((pressure, u, sortie_pos, sortie))

        sortie_candidates.sort(key=lambda item: item[0], reverse=True)
        for _, u, sortie_pos, sortie in sortie_candidates[:candidate_limit]:
            representative = max(
                sortie.customers,
                key=lambda c: ev.drone_tardiness.get(c, 0.0),
            )
            pairs = endpoint_candidate_pairs(route, representative, self.data)[:candidate_limit]
            for order_tuple in itertools.permutations(sortie.customers):
                order = list(order_tuple)
                for launch, recovery, _ in pairs:
                    if (
                            order == sortie.customers
                            and launch == sortie.launch
                            and recovery == sortie.recovery
                    ):
                        continue
                    replacement = Sortie(
                        drone=u,
                        index=sortie.index,
                        launch=launch,
                        recovery=recovery,
                        customers=order,
                    )
                    if not is_sortie_feasible_fast(replacement, self.data):
                        continue
                    cand = sol.clone()
                    cand.sorties_by_drone[u][sortie_pos] = replacement
                    cand = self._compact_drone_assignments(cand)
                    cand_eval = evaluate_solution(
                        self.data,
                        cand,
                        compact_missing_violations=True,
                    )
                    if cand_eval.feasible and cand_eval.objective + 1e-9 < best_obj:
                        best_sol = cand
                        best_obj = cand_eval.objective
        return best_sol

    def _global_vnd(self, sol: Solution, max_passes: Optional[int] = None) -> Tuple[Solution, Evaluation]:
        cur = self._compact_drone_assignments(sol)
        cur_eval = evaluate_solution(self.data, cur, compact_missing_violations=True)
        pass_budget = (
            max(1, self.cfg.global_polish_vnd_passes)
            if max_passes is None
            else max(1, max_passes)
        )
        for _ in range(pass_budget):
            candidates: List[Tuple[float, Solution, Evaluation]] = []
            for cand in (
                    self._best_route_move_global(cur, cur_eval),
                    self._best_endpoint_pair_global(cur, cur_eval),
                    self._best_sortie_rebuild_global(cur, cur_eval),
                    self._best_sortie_chain_merge(cur, cur_eval),
            ):
                if cand is None:
                    continue
                cand_eval = evaluate_solution(self.data, cand, compact_missing_violations=True)
                if cand_eval.feasible and cand_eval.objective + 1e-9 < cur_eval.objective:
                    candidates.append((cand_eval.objective, cand, cand_eval))
            if not candidates:
                break
            _, cur, cur_eval = min(candidates, key=lambda item: item[0])
        return cur, cur_eval

    def _remove_drone_customer(self, sol: Solution, customer: int) -> Optional[Solution]:
        cand = sol.clone()
        found = False
        for u in self.data.U:
            kept: List[Sortie] = []
            for sortie in cand.sorties_by_drone.get(u, []):
                if customer in sortie.customers:
                    sortie.customers = [c for c in sortie.customers if c != customer]
                    found = True
                if sortie.customers:
                    kept.append(sortie)
            cand.sorties_by_drone[u] = kept
        cand.normalize(route_positions(cand.truck_route))
        return cand if found else None

    def _best_drone_to_truck_global(self, sol: Solution, ev: Evaluation) -> Optional[Solution]:
        raw: List[Tuple[float, Solution]] = []
        candidate_limit = max(1, self.cfg.global_polish_candidate_limit)
        drone_customers = [
            customer for customer, mode in ev.customer_mode.items() if mode == "drone"
        ]
        drone_customers.sort(
            key=lambda c: ev.drone_tardiness.get(c, 0.0),
            reverse=True,
        )
        for customer in drone_customers[:candidate_limit]:
            partial = self._remove_drone_customer(sol, customer)
            if partial is None:
                continue
            for insert_pos in range(1, len(partial.truck_route)):
                cand = partial.clone()
                cand.truck_route.insert(insert_pos, customer)
                cand = self._compact_drone_assignments(cand)
                cand_eval = evaluate_solution(self.data, cand, compact_missing_violations=True)
                if cand_eval.feasible:
                    raw.append((cand_eval.objective, cand))

        best_sol = None
        best_obj = ev.objective
        top_k = max(1, self.cfg.global_polish_top_insertions)
        for _, candidate in sorted(raw, key=lambda item: item[0])[:top_k]:
            improved, improved_eval = self._global_vnd(candidate)
            if improved_eval.feasible and improved_eval.objective + 1e-9 < best_obj:
                best_sol = improved
                best_obj = improved_eval.objective
        return best_sol

    def _best_mode_exchange_global(self, sol: Solution, ev: Evaluation) -> Optional[Solution]:
        """Exchange drone/truck service modes and cross a two-move 2-opt barrier."""
        if not self.data.U:
            return None
        drone_customers = [c for c, mode in ev.customer_mode.items() if mode == "drone"]
        truck_customers = [
            c for c, mode in ev.customer_mode.items()
            if mode == "truck" and c in self.data.Nku
        ]
        best_sol = None
        best_obj = ev.objective
        beam_width = max(1, self.cfg.global_polish_beam_width)
        candidate_limit = max(1, self.cfg.global_polish_candidate_limit)
        drone_customers.sort(key=lambda c: ev.drone_tardiness.get(c, 0.0), reverse=True)
        truck_customers.sort(key=lambda c: ev.truck_tardiness.get(c, 0.0), reverse=True)
        drone_customers = drone_customers[:candidate_limit]
        truck_customers = truck_customers[:candidate_limit]

        for drone_customer in drone_customers:
            after_drone = self._remove_drone_customer(sol, drone_customer)
            if after_drone is None:
                continue
            protected_endpoints = launch_recovery_nodes(after_drone)
            for truck_customer in truck_customers:
                if truck_customer in protected_endpoints or truck_customer not in after_drone.truck_route:
                    continue
                partial = after_drone.clone()
                partial.truck_route.remove(truck_customer)

                raw: List[Tuple[float, Solution]] = []
                insertion_positions: List[Tuple[float, int]] = []
                for insert_pos in range(1, len(partial.truck_route)):
                    prev_node = partial.truck_route[insert_pos - 1]
                    next_node = partial.truck_route[insert_pos]
                    delta = (
                        self.data.truck_dist[(prev_node, drone_customer)]
                        + self.data.truck_dist[(drone_customer, next_node)]
                        - self.data.truck_dist[(prev_node, next_node)]
                    )
                    insertion_positions.append((delta, insert_pos))
                insertion_positions.sort(key=lambda item: item[0])

                for _, insert_pos in insertion_positions[:candidate_limit]:
                    with_truck = partial.clone()
                    with_truck.truck_route.insert(insert_pos, drone_customer)
                    route = with_truck.truck_route
                    pairs = endpoint_candidate_pairs(route, truck_customer, self.data)[:candidate_limit]
                    for launch, recovery, _ in pairs:
                        sortie = Sortie(
                            drone=self.data.U[-1],
                            index=999,
                            launch=launch,
                            recovery=recovery,
                            customers=[truck_customer],
                        )
                        if not is_sortie_feasible_fast(sortie, self.data):
                            continue
                        cand = with_truck.clone()
                        cand.sorties_by_drone[self.data.U[-1]].append(sortie)
                        cand = self._compact_drone_assignments(cand)
                        cand_eval = evaluate_solution(self.data, cand, compact_missing_violations=True)
                        if cand_eval.feasible:
                            raw.append((cand_eval.objective, cand))

                for _, candidate in sorted(raw, key=lambda item: item[0])[:beam_width]:
                    first_layer: List[Tuple[float, Solution]] = []
                    first_routes = [candidate.truck_route, *list(self._two_opt_routes(candidate.truck_route))]
                    for first_route in first_routes:
                        first = candidate.clone()
                        first.truck_route = first_route
                        first = self._compact_drone_assignments(first)
                        first_eval = evaluate_solution(self.data, first, compact_missing_violations=True)
                        if not first_eval.feasible:
                            continue
                        first_layer.append((first_eval.objective, first))
                        if first_eval.objective + 1e-9 < best_obj:
                            best_sol = first
                            best_obj = first_eval.objective

                    for _, first in sorted(first_layer, key=lambda item: item[0])[:beam_width]:
                        for second_route in self._two_opt_routes(first.truck_route):
                            second = first.clone()
                            second.truck_route = second_route
                            second = self._compact_drone_assignments(second)
                            second_eval = evaluate_solution(self.data, second, compact_missing_violations=True)
                            if second_eval.feasible and second_eval.objective + 1e-9 < best_obj:
                                best_sol = second
                                best_obj = second_eval.objective
        return best_sol

    def _global_polish_best(self, sol: Solution, ev: Evaluation) -> Tuple[Solution, Evaluation]:
        customer_count = max(1, len(self.data.C))
        strength = continuous_size_strength(
            customer_count,
            self.cfg.global_polish_reference_customers,
            self.cfg.global_polish_decay_exponent,
        )

        base_passes = max(1, int(self.cfg.global_polish_passes))
        base_beam = max(1, int(self.cfg.global_polish_beam_width))
        base_top = max(1, int(self.cfg.global_polish_top_insertions))
        base_permutation = max(2, int(self.cfg.global_sortie_permutation_max_customers))
        base_candidate_limit = max(1, int(self.cfg.global_polish_candidate_limit))
        base_vnd_passes = max(1, int(self.cfg.global_polish_vnd_passes))

        pass_budget = max(1, math.ceil(base_passes * strength))
        beam_budget = max(1, math.ceil(base_beam * strength))
        top_budget = max(1, math.ceil(base_top * strength))
        permutation_budget = max(2, round(2 + (base_permutation - 2) * strength))
        candidate_budget = max(1, math.ceil(base_candidate_limit * strength))
        vnd_budget = max(1, math.ceil(base_vnd_passes * strength))

        print(
            f"[Polish] customers={customer_count}, strength={strength:.4f}, "
            f"passes={pass_budget}, candidates={candidate_budget}, "
            f"beam={beam_budget}, top={top_budget}, "
            f"sortie_perm={permutation_budget}, vnd={vnd_budget}"
        )

        saved = (
            self.cfg.global_polish_beam_width,
            self.cfg.global_polish_top_insertions,
            self.cfg.global_sortie_permutation_max_customers,
            self.cfg.global_polish_candidate_limit,
            self.cfg.global_polish_vnd_passes,
        )
        self.cfg.global_polish_beam_width = beam_budget
        self.cfg.global_polish_top_insertions = top_budget
        self.cfg.global_sortie_permutation_max_customers = permutation_budget
        self.cfg.global_polish_candidate_limit = candidate_budget
        self.cfg.global_polish_vnd_passes = vnd_budget

        cur = sol.clone()
        cur_eval = ev
        try:
            for _ in range(pass_budget):
                candidates: List[Tuple[float, Solution, Evaluation]] = []
                for build_candidate in (
                        lambda: self._best_sortie_chain_merge(cur, cur_eval),
                        lambda: self._best_sortie_rebuild_global(cur, cur_eval),
                        lambda: self._best_drone_to_truck_global(cur, cur_eval),
                        lambda: self._best_mode_exchange_global(cur, cur_eval),
                ):
                    cand = build_candidate()
                    if cand is None:
                        continue
                    cand_eval = evaluate_solution(self.data, cand)
                    if cand_eval.feasible and cand_eval.objective + 1e-9 < cur_eval.objective:
                        candidates.append((cand_eval.objective, cand, cand_eval))
                if not candidates:
                    break
                _, cur, cur_eval = min(candidates, key=lambda item: item[0])
        finally:
            (
                self.cfg.global_polish_beam_width,
                self.cfg.global_polish_top_insertions,
                self.cfg.global_sortie_permutation_max_customers,
                self.cfg.global_polish_candidate_limit,
                self.cfg.global_polish_vnd_passes,
            ) = saved
        return cur, cur_eval

    def _polish_best(self, best: Solution, best_eval: Evaluation, passes: Optional[int] = None) -> Tuple[Solution, Evaluation]:
        if self.cfg.experiment_polish_budget:
            base_passes = self.cfg.polish_passes if passes is None else passes
            passes = max(1, math.ceil(base_passes / (1.0 + len(self.data.C) / 25.0)))
        polish_passes = self.cfg.polish_passes if passes is None else passes
        if polish_passes <= 0:
            return best, best_eval

        cur = best.clone()
        cur_eval = best_eval
        for _ in range(polish_passes):
            improved = False
            single_reinsert = self._best_single_customer_reinsert(cur, cur_eval)
            endpoint_relocation = self._best_endpoint_relocation(cur, cur_eval)
            polish_candidates = [single_reinsert, endpoint_relocation, self.op16(cur, cur_eval, self.cfg.max_iter),
                                 self.op17(cur, cur_eval, self.cfg.max_iter)]
            for cand in polish_candidates:
                if cand is None:
                    continue
                cand.normalize(route_positions(cand.truck_route))
                cand_eval = evaluate_solution(self.data, cand)
                if cand_eval.feasible and cand_eval.objective + 1e-9 < cur_eval.objective:
                    cur = cand
                    cur_eval = cand_eval
                    improved = True
            if not improved:
                break
        if self.cfg.enable_global_polish:
            cur, cur_eval = self._global_polish_best(cur, cur_eval)
        return cur, cur_eval

    def _intensify_after_reconstruct(
            self,
            sol: Solution,
            ev: Evaluation,
            iteration: int,
    ) -> Tuple[Solution, Evaluation, bool]:
        """Apply sequential strict-improvement neighborhoods after reconstruction."""
        cur = sol.clone()
        cur_eval = ev
        improved_any = False
        passes = max(0, int(self.cfg.reconstruct_intensification_passes))

        for _ in range(passes):
            improved_this_pass = False
            neighborhoods = (
                ("tardiness_reinsert", lambda: self.op21(cur, cur_eval, iteration)),
                ("time_pressure_2opt", lambda: self.op17(cur, cur_eval, iteration)),
                ("drone_wait_rebuild", lambda: self.op19(cur, cur_eval, iteration)),
                ("truck_to_drone", lambda: self.op16(cur, cur_eval, iteration)),
                ("endpoint_relocation", lambda: self._best_endpoint_relocation(cur, cur_eval)),
            )
            for name, build_candidate in neighborhoods:
                self.reconstruct_intensification_calls[name] += 1
                cand = build_candidate()
                if cand is None:
                    continue
                cand.normalize(route_positions(cand.truck_route))
                cand_eval = evaluate_solution(self.data, cand)
                if cand_eval.feasible and cand_eval.objective + 1e-9 < cur_eval.objective:
                    cur = cand
                    cur_eval = cand_eval
                    improved_any = True
                    improved_this_pass = True
                    self.reconstruct_intensification_wins[name] += 1
            if not improved_this_pass:
                break

        return cur, cur_eval, improved_any

    def run(self, init_solution: Optional[Solution] = None) -> Tuple[Solution, Evaluation]:
        cur = init_solution.clone() if init_solution is not None else create_initial_solution(self.data, self.rng)
        cur_eval = evaluate_solution(self.data, cur)
        if not cur_eval.feasible:
            cur = self._repair_if_needed(cur, cur)
            cur_eval = evaluate_solution(self.data, cur)
        if not cur_eval.feasible:
            raise RuntimeError("Initial solution is infeasible and cannot be repaired.")

        best = cur.clone()
        best_eval = cur_eval
        self.tabu.append(best_eval.fingerprint)
        self.best_update_log = [(0, best_eval.objective)]

        T_start = -(self.cfg.sa_start_ratio * max(1.0, cur_eval.objective)) / math.log(0.5)
        T_end = -(self.cfg.sa_end_ratio * max(1.0, cur_eval.objective)) / math.log(0.5)
        if T_end >= T_start: T_end = T_start * 0.001

        # Geometric cooling with plateau-triggered reheating in the same chain.
        schedule_horizon = (
            min(self.cfg.max_iter, self.cfg.runtime_schedule_horizon)
            if self.cfg.runtime_schedule_horizon > 0
            else self.cfg.max_iter
        )
        freeze_iter = max(1, int(schedule_horizon * self.cfg.sa_freeze_ratio))
        cooling_rate = (T_end / T_start) ** (1.0 / freeze_iter)

        T = T_start
        Tmin = T_end

        reconstruct_no_improve = 0
        finish_no_improve = 0
        reconstruct_count = 0
        reconstruct_success_count = 0
        reconstruct_intensification_improvements = 0
        plateau_polish_attempts = 0
        plateau_polish_improvements = 0
        last_reconstruct_attempt_iter = 0
        destroy_pulse_remaining = 0
        start = time.time()
        cpu_start = time.process_time()
        self._experiment_cpu_start = cpu_start
        self._experiment_last_iteration = 0
        def search_progress(iteration: int) -> float:
            if self.cfg.experiment_cpu_budget > 0:
                search_budget = self.cfg.experiment_cpu_budget * (1.0 - self.cfg.experiment_final_reserve)
                return min(1.0, (time.process_time() - cpu_start) / max(1e-9, search_budget))
            return iteration / max(1, schedule_horizon)
        last_it = 0

        def on_no_best_iteration(iteration: int) -> bool:
            nonlocal cur, cur_eval, best, best_eval, reconstruct_no_improve, finish_no_improve
            nonlocal reconstruct_count, reconstruct_success_count, last_reconstruct_attempt_iter
            nonlocal reconstruct_intensification_improvements
            nonlocal plateau_polish_attempts, plateau_polish_improvements
            nonlocal destroy_pulse_remaining, T
            reconstruct_no_improve += 1
            finish_no_improve += 1

            if (
                    self.cfg.enable_plateau_polish
                    and self.cfg.cycle_no_improve_polish > 0
                    and finish_no_improve >= self.cfg.cycle_no_improve_polish
                    and finish_no_improve % self.cfg.cycle_no_improve_polish == 0
            ):
                plateau_polish_attempts += 1
                polished, pol_eval = self._polish_best(best, best_eval, passes=self.cfg.plateau_polish_passes)
                if pol_eval.feasible and pol_eval.objective + 1e-9 < best_eval.objective:
                    best = polished.clone()
                    best_eval = pol_eval
                    self.best_update_log.append((iteration, best_eval.objective))
                    plateau_polish_improvements += 1
                    reconstruct_no_improve = 0
                    finish_no_improve = 0
                    return False

            if (
                    self.cfg.enable_large_reconstruct
                    and self.cfg.cycle_no_improve_reconstruct > 0
                    and reconstruct_no_improve >= self.cfg.cycle_no_improve_reconstruct
                    and iteration - last_reconstruct_attempt_iter >= self.cfg.cycle_no_improve_reconstruct
                    and search_progress(iteration) < self.cfg.adaptive_reheat_stop_ratio
            ):
                reconstruct_count += 1
                last_reconstruct_attempt_iter = iteration
                patience = max(1.0, float(self.cfg.cycle_no_improve_reconstruct))
                growth_scale = max(1e-6, float(self.cfg.reconstruct_growth_scale))
                excess_stagnation = max(0.0, reconstruct_no_improve - patience)
                severity = 1.0 - math.exp(-excess_stagnation / (patience * growth_scale))
                span = self.cfg.large_reconstruct_max_fraction - self.cfg.large_reconstruct_min_fraction
                intensity = self.cfg.large_reconstruct_min_fraction + span * severity
                best_base_probability = self.cfg.reconstruct_best_base_min_probability + (
                    self.cfg.reconstruct_best_base_max_probability
                    - self.cfg.reconstruct_best_base_min_probability
                ) * severity
                restart_base = best if self.rng.random() < best_base_probability else cur
                reconstructed = large_destroy_reconstruct(
                    restart_base,
                    self.data,
                    self.rng,
                    intensity=intensity,
                    attempts=self.cfg.large_reconstruct_attempts,
                    strategy_offset=self.rng.randrange(3),
                    prefer_best=(
                        search_progress(iteration)
                        >= self.cfg.reconstruct_intensify_ratio
                    ),
                )
                rec_eval = evaluate_solution(self.data, reconstructed) if reconstructed is not None else None
                if (
                        rec_eval is not None
                        and rec_eval.feasible
                        and rec_eval.fingerprint != cur_eval.fingerprint
                ):
                    reconstruct_success_count += 1

                    if (self.cfg.enable_reconstruct_immediate_intensification
                            and self._post_reconstruct_is_promising(rec_eval.objective, best_eval.objective,
                                                                   search_progress(iteration))):
                        reconstructed, rec_eval, intensified = self._intensify_after_reconstruct(
                            reconstructed,
                            rec_eval,
                            iteration,
                        )
                        if intensified:
                            reconstruct_intensification_improvements += 1

                    if self.cfg.experiment_reconstruct_guard:
                        progress = search_progress(iteration)
                        allowed_relative_loss = (0.01 + 0.09 * severity) * max(0.0, 1.0 - progress)
                        reheat_guard = max(T, T_start * self.cfg.adaptive_reheat_ratio * (1.0 - 0.60 * progress))
                        if rec_eval.objective > cur_eval.objective * (1.0 + allowed_relative_loss):
                            return False
                        if not self._accept_by_sa(rec_eval.objective, cur_eval.objective, reheat_guard):
                            return False
                    cur = reconstructed
                    cur_eval = rec_eval
                    self.tabu.clear()
                    self.tabu.append(rec_eval.fingerprint)
                    pulse_ratio = self.cfg.reconstruct_pulse_min_strength + (
                        1.0 - self.cfg.reconstruct_pulse_min_strength
                    ) * severity
                    self.cfg.runtime_destroy_scale = 1.0 + (
                        self.cfg.adaptive_destroy_scale - 1.0
                    ) * pulse_ratio
                    destroy_pulse_remaining = self.cfg.adaptive_destroy_pulse

                    progress = search_progress(iteration)
                    reheat_decay = max(0.35, 1.0 - 0.60 * progress)
                    reheat_strength = 0.65 + 0.35 * severity
                    reheat_target = T_start * self.cfg.adaptive_reheat_ratio * reheat_decay * reheat_strength
                    T = max(T, reheat_target)

                    # Smooth operator weights continuously as stagnation deepens.
                    neutral_weight = sum(s.weight for s in self.operator_stats) / len(self.operator_stats)
                    smoothing = self.cfg.reconstruct_weight_smoothing_min + (
                        self.cfg.reconstruct_weight_smoothing_max
                        - self.cfg.reconstruct_weight_smoothing_min
                    ) * severity
                    for stat in self.operator_stats:
                        stat.weight = (1.0 - smoothing) * stat.weight + smoothing * neutral_weight
                        stat.score_sum = 0.0
                        stat.use_count = 0

                    if rec_eval.objective + 1e-9 < best_eval.objective:
                        best = reconstructed.clone()
                        best_eval = rec_eval
                        self.best_update_log.append((iteration, best_eval.objective))
                        reconstruct_no_improve = 0
                        finish_no_improve = 0

            return (
                    self.cfg.enable_early_stop
                    and self.cfg.cycle_no_improve_finish > 0
                    and finish_no_improve >= self.cfg.cycle_no_improve_finish
            )

        for it in range(1, self.cfg.max_iter + 1):
            last_it = it
            self._experiment_last_iteration = it
            progress = search_progress(it)
            if self.cfg.experiment_cpu_budget > 0 and progress >= 1.0:
                break
            frozen = progress >= self.cfg.sa_freeze_ratio if self.cfg.experiment_cpu_budget > 0 else it >= freeze_iter
            if reheat_pulse_active(progress, destroy_pulse_remaining, self.cfg, schedule_horizon):
                frozen = False
            if self.cfg.force_final_sa_freeze and frozen:
                T = Tmin
            elif self.cfg.experiment_cpu_budget > 0:
                target = T_start * (T_end / T_start) ** min(1.0, progress / self.cfg.sa_freeze_ratio)
                T = max(Tmin, target, T * cooling_rate)
            elif it > 1:
                T = max(Tmin, cooling_rate * T)
            if destroy_pulse_remaining > 0:
                destroy_pulse_remaining -= 1
            else:
                self.cfg.runtime_destroy_scale = 1.0

            op_idx = roulette_select(self.operator_stats, self.rng)
            op = self.operator_funcs[op_idx]

            operator_it = max(1, min(self.cfg.max_iter, int(progress * self.cfg.max_iter))) if self.cfg.experiment_cpu_budget > 0 else it
            cand = op(cur, cur_eval, operator_it)
            if cand is None:
                self._record_operator_score(op_idx, self.cfg.score5)
                if it % self.cfg.cycle_update == 0:
                    self._update_weights()
                should_stop = on_no_best_iteration(it)
                self.history_obj.append(best_eval.objective)
                if should_stop:
                    break
                continue

            cand_eval = evaluate_solution(self.data, cand)
            if not cand_eval.feasible:
                cand = self._repair_if_needed(cand, cur)
                cand_eval = evaluate_solution(self.data, cand)
            if not cand_eval.feasible:
                self._record_operator_score(op_idx, self.cfg.score5)
                if it % self.cfg.cycle_update == 0:
                    self._update_weights()
                should_stop = on_no_best_iteration(it)
                self.history_obj.append(best_eval.objective)
                if should_stop:
                    break
                continue
            if self.cfg.experiment_noop_failure and cand_eval.fingerprint == cur_eval.fingerprint:
                self._record_operator_score(op_idx, self.cfg.score5)
                if it % self.cfg.cycle_update == 0:
                    self._update_weights()
                should_stop = on_no_best_iteration(it)
                self.history_obj.append(best_eval.objective)
                if should_stop:
                    break
                continue
            if self._tabu_rejects(cand_eval, cur_eval, best_eval):
                self._record_operator_score(op_idx, self.cfg.score5)
                if it % self.cfg.cycle_update == 0:
                    self._update_weights()
                should_stop = on_no_best_iteration(it)
                self.history_obj.append(best_eval.objective)
                if should_stop:
                    break
                continue

            accepted = self._accept_by_sa(cand_eval.objective, cur_eval.objective, T)
            improved_best = cand_eval.objective < best_eval.objective
            no_best_this_iter = False
            if improved_best:
                best = cand.clone()
                best_eval = cand_eval
                self.best_update_log.append((it, best_eval.objective))
                reconstruct_no_improve = 0
                finish_no_improve = 0
                destroy_pulse_remaining = 0
                self.cfg.runtime_destroy_scale = 1.0
            else:
                no_best_this_iter = True

            if accepted:
                if improved_best:
                    score = self.cfg.score1
                elif cand_eval.objective < cur_eval.objective:
                    score = self.cfg.score2
                else:
                    score = self.cfg.score3
                cur = cand
                cur_eval = cand_eval
            else:
                score = self.cfg.score4

            if accepted:
                self.tabu.append(cand_eval.fingerprint)
            self._record_operator_score(op_idx, score)

            if it % self.cfg.cycle_update == 0:
                self._update_weights()

            should_stop = on_no_best_iteration(it) if no_best_this_iter else False

            if (
                    self.cfg.enable_schedule_horizon_polish
                    and schedule_horizon < self.cfg.max_iter
                    and it == schedule_horizon
            ):
                polished, pol_eval = self._polish_best(best, best_eval)
                if pol_eval.feasible and pol_eval.objective + 1e-9 < best_eval.objective:
                    best = polished.clone()
                    best_eval = pol_eval
                    cur = polished.clone()
                    cur_eval = pol_eval
                    self.best_update_log.append((it, best_eval.objective))
                    reconstruct_no_improve = 0
                    finish_no_improve = 0
                    destroy_pulse_remaining = 0
                    self.cfg.runtime_destroy_scale = 1.0

            self.history_obj.append(best_eval.objective)
            if should_stop:
                break

        before_polish_obj = best_eval.objective
        if self.cfg.enable_final_polish:
            best, best_eval = self._polish_best(best, best_eval)
        if best_eval.objective + 1e-9 < before_polish_obj:
            self.best_update_log.append((last_it, best_eval.objective))
            if self.history_obj:
                self.history_obj[-1] = best_eval.objective
        if self.best_update_log:
            late_from = max(0, last_it - 500)
            late_updates = [(i, obj) for i, obj in self.best_update_log if i >= late_from and i > 0]
            print(
                f"[ALNS] best updates={len(self.best_update_log) - 1}, "
                f"last_update_iter={self.best_update_log[-1][0]}, "
                f"updates_in_last_500={len(late_updates)}"
            )
        print(
            f"[ALNS] reconstruct_attempts={reconstruct_count}, "
            f"reconstruct_successes={reconstruct_success_count}, "
            f"reconstruct_intensified={reconstruct_intensification_improvements}, "
            f"final_temperature={T:.6f}, iterations={last_it}"
        )
        print(
            f"[ALNS] plateau_polish_attempts={plateau_polish_attempts}, "
            f"plateau_polish_improvements={plateau_polish_improvements}"
        )
        print(
            "[ALNS] reconstruct_intensification_calls="
            + ",".join(
                f"{name}:{count}"
                for name, count in self.reconstruct_intensification_calls.items()
            )
        )
        print(
            "[ALNS] reconstruct_intensification_wins="
            + ",".join(
                f"{name}:{count}"
                for name, count in self.reconstruct_intensification_wins.items()
            )
        )
        print(f"[ALNS] finished in {time.time() - start:.2f}s, best objective={best_eval.objective:.6f}")
        return best, best_eval

    def op1(self, cur: Solution, ev: Evaluation, it: int) -> Optional[Solution]:
        cand = cur.clone()
        lr = launch_recovery_nodes(cand)
        truck_customers = [c for c in cand.truck_route if c in self.data.C and c not in lr]
        if not truck_customers:
            return None
        k = operator_remove_counts(len(truck_customers), it, self.cfg.max_iter,
                                   [(0.15, 0.20), (0.10, 0.20), (0.05, 0.10), (0.01, 0.10)], self.rng, self.cfg)
        remove_list = self.rng.sample(truck_customers, min(k, len(truck_customers)))
        remove_customers_from_solution(cand, remove_list, cascade_if_launch_recovery=False)
        self.rng.shuffle(remove_list)
        for c in remove_list:
            cand = try_random_truck_insertion(cand, c, self.rng)
        cand.normalize(route_positions(cand.truck_route))
        return cand

    def op2(self, cur: Solution, ev: Evaluation, it: int) -> Optional[Solution]:
        cand = cur.clone()
        lr = launch_recovery_nodes(cand)
        truck_customers = [c for c in cand.truck_route if c in self.data.C and c not in lr]
        if not truck_customers:
            return None
        k = operator_remove_counts(len(truck_customers), it, self.cfg.max_iter,
                                   [(0.25, 0.40), (0.15, 0.30), (0.10, 0.20), (0.05, 0.15)], self.rng, self.cfg)
        remove_list = self.rng.sample(truck_customers, min(k, len(truck_customers)))
        remove_customers_from_solution(cand, remove_list, cascade_if_launch_recovery=False)
        self.rng.shuffle(remove_list)
        for c in remove_list:
            cand = try_best_truck_distance_insertion(cand, c, self.data)
        cand.normalize(route_positions(cand.truck_route))
        return cand

    def op3(self, cur: Solution, ev: Evaluation, it: int) -> Optional[Solution]:
        all_customers = list(self.data.C)
        if not all_customers:
            return None
        k = operator_remove_counts(len(all_customers), it, self.cfg.max_iter,
                                   [(0.15, 0.20), (0.10, 0.20), (0.05, 0.15), (0.01, 0.10)], self.rng, self.cfg)
        remove_list = self.rng.sample(all_customers, min(k, len(all_customers)))
        return self._remove_and_reinsert_total(cur, remove_list, random_order=True)

    def op4(self, cur: Solution, ev: Evaluation, it: int) -> Optional[Solution]:
        cand = cur.clone()
        lr = launch_recovery_nodes(cand)
        route = cand.truck_route
        truck_customers = [c for c in route if c in self.data.C and c not in lr]
        if not truck_customers:
            return None
        k = operator_remove_counts(len(truck_customers), it, self.cfg.max_iter,
                                   [(0.25, 0.40), (0.15, 0.30), (0.10, 0.20), (0.05, 0.15)], self.rng, self.cfg)
        gains: List[Tuple[float, int]] = []
        for idx in range(1, len(route) - 1):
            c = route[idx]
            if c not in truck_customers:
                continue
            prev_n = route[idx - 1]
            next_n = route[idx + 1]
            gain = self.data.truck_dist[(prev_n, c)] + self.data.truck_dist[(c, next_n)] - self.data.truck_dist[
                (prev_n, next_n)]
            gains.append((gain, c))
        if not gains:
            return None
        gains.sort(reverse=True)
        remove_list: List[int] = []
        for _ in range(min(k, len(gains))):
            ridx = int(round((self.rng.random() ** self.cfg.m) * (len(gains) - 1)))
            remove_list.append(gains[ridx][1])
            gains.pop(ridx)
            if not gains:
                break
        remove_customers_from_solution(cand, remove_list, cascade_if_launch_recovery=False)
        self.rng.shuffle(remove_list)
        for c in remove_list:
            cand = try_best_truck_distance_insertion(cand, c, self.data)
        cand.normalize(route_positions(cand.truck_route))
        return cand

    def op5(self, cur: Solution, ev: Evaluation, it: int) -> Optional[Solution]:
        all_customers = list(self.data.C)
        if len(all_customers) < 2:
            return None
        k = operator_remove_counts(len(all_customers), it, self.cfg.max_iter,
                                   [(0.15, 0.20), (0.10, 0.15), (0.05, 0.15), (0.01, 0.10)], self.rng, self.cfg)
        seed = self.rng.choice(all_customers)
        pos = route_positions(cur.truck_route)
        sortie_group: Dict[int, Tuple[int, int]] = {}
        for u in self.data.U:
            for s in cur.sorties_by_drone.get(u, []):
                for c in s.customers:
                    sortie_group[c] = (u, s.index)
        rel = [(relatedness_score(seed, c, cur, self.data, pos, sortie_group), c) for c in all_customers if c != seed]
        rel.sort(reverse=True)
        remove_list = [seed]
        while rel and len(remove_list) < k:
            if self.rng.random() < shaw_random_pick_probability(self.data):
                rank = self.rng.randrange(len(rel))
            else:
                rank = int((self.rng.random() ** self.cfg.m) * len(rel))
                rank = min(rank, len(rel) - 1)
            _, customer = rel.pop(rank)
            remove_list.append(customer)
        cand = cur.clone()
        removed = remove_customers_from_solution(cand, remove_list, cascade_if_launch_recovery=True)
        remaining = [c for c in removed if c in self.data.C]
        if not remaining:
            return cand
        regret_k = 2 if self.rng.random() < 0.7 else 3
        while remaining:
            picked = choose_regret_customer(cand, remaining, self.data, regret_k)
            if picked is None:
                return None
            c, nxt = picked
            cand = nxt
            remaining.remove(c)
        cand.normalize(route_positions(cand.truck_route))
        return cand

    def op6(self, cur: Solution, ev: Evaluation, it: int) -> Optional[Solution]:
        cand = cur.clone()
        all_sorties = [(u, s.index) for u in self.data.U for s in cand.sorties_by_drone.get(u, [])]
        if not all_sorties:
            return None
        u, p = self.rng.choice(all_sorties)
        removed = remove_sortie(cand, u, p)
        if not removed:
            return None
        self.rng.shuffle(removed)
        remaining = set(c for c in removed if c in self.data.C)
        for c in removed:
            remaining.discard(c)
            nxt = try_best_total_insertion(cand, c, self.data, allowed_missing=remaining)
            if nxt is None:
                return None
            cand = nxt
        cand.normalize(route_positions(cand.truck_route))
        return cand

    def op7(self, cur: Solution, ev: Evaluation, it: int) -> Optional[Solution]:
        cand = cur.clone()
        all_sorties = [(u, s.index) for u in self.data.U for s in cand.sorties_by_drone.get(u, [])]
        if not all_sorties:
            return None
        u, p = self.rng.choice(all_sorties)
        removed = remove_sortie(cand, u, p)
        if not removed:
            return None
        self.rng.shuffle(removed)
        for c in removed:
            cand = try_best_truck_distance_insertion(cand, c, self.data)
        cand.normalize(route_positions(cand.truck_route))
        return cand

    def op8(self, cur: Solution, ev: Evaluation, it: int) -> Optional[Solution]:
        dcs = self._drone_served(ev)
        tcs = self._truck_served(ev)
        if not dcs or not tcs:
            return None
        c_drone = self.rng.choice(dcs)
        c_truck = self.rng.choice(tcs)
        if c_drone == c_truck:
            return None
        cand = cur.clone()
        removed = remove_customers_from_solution(cand, [c_drone, c_truck], cascade_if_launch_recovery=True)
        to_insert = [c for c in removed if c in self.data.C]

        if c_drone in to_insert:
            cand = try_best_truck_distance_insertion(cand, c_drone, self.data)
            to_insert.remove(c_drone)

        if c_truck in to_insert:
            allowed_after = set(to_insert) - {c_truck}
            nxt = try_best_total_insertion(cand, c_truck, self.data, allowed_missing=allowed_after)
            if nxt is None:
                return None
            cand = nxt
            to_insert.remove(c_truck)

        self.rng.shuffle(to_insert)
        remaining = set(to_insert)
        for c in to_insert:
            remaining.discard(c)
            nxt = try_best_total_insertion(cand, c, self.data, allowed_missing=remaining)
            if nxt is None:
                return None
            cand = nxt
        cand.normalize(route_positions(cand.truck_route))
        return cand

    def op9(self, cur: Solution, ev: Evaluation, it: int) -> Optional[Solution]:
        cand = cur.clone()
        options: List[Tuple[int, int, str]] = []
        for u in self.data.U:
            for s in cand.sorties_by_drone.get(u, []):
                options.append((u, s.index, "launch"))
                options.append((u, s.index, "recovery"))
        if not options:
            return None
        u, p, side = self.rng.choice(options)
        s = next((x for x in cand.sorties_by_drone[u] if x.index == p), None)
        if s is None:
            return None
        endpoint = s.launch if side == "launch" else s.recovery
        if endpoint in {self.data.depot_start_id, self.data.depot_end_id} or endpoint not in self.data.C:
            return None

        for uu in self.data.U:
            for ss in cand.sorties_by_drone.get(uu, []):
                if uu == u and ss.index == p:
                    continue
                if endpoint == ss.launch or endpoint == ss.recovery:
                    return None

        if endpoint in cand.truck_route:
            cand.truck_route = [n for n in cand.truck_route if n != endpoint]
        pos = route_positions(cand.truck_route)
        if side == "launch":
            rec_idx = pos.get(s.recovery)
            if rec_idx is None or rec_idx <= 0:
                return None
            s.launch = cand.truck_route[rec_idx - 1]
        else:
            l_idx = pos.get(s.launch)
            if l_idx is None or l_idx >= len(cand.truck_route) - 1:
                return None
            s.recovery = cand.truck_route[l_idx + 1]
        best_k = None
        best_delta = float("inf")
        for k in range(len(s.customers) + 1):
            prev_n = s.launch if k == 0 else s.customers[k - 1]
            next_n = s.recovery if k == len(s.customers) else s.customers[k]
            delta = self.data.drone_dist[(prev_n, endpoint)] + self.data.drone_dist[(endpoint, next_n)] - \
                    self.data.drone_dist[(prev_n, next_n)]
            if delta < best_delta:
                best_delta = delta
                best_k = k
        if best_k is None:
            return None
        s.customers.insert(best_k, endpoint)
        cand.normalize(route_positions(cand.truck_route))
        return cand

    def op10(self, cur: Solution, ev: Evaluation, it: int) -> Optional[Solution]:
        cand = cur.clone()
        all_sorties = [(u, s.index) for u in self.data.U for s in cand.sorties_by_drone.get(u, [])]
        if not all_sorties:
            return None
        u, p = self.rng.choice(all_sorties)
        side = "launch" if self.rng.random() < 0.5 else "recovery"
        base = evaluate_solution(self.data, cand)
        best_sol = None
        best_delta = float("inf")
        pos = route_positions(cand.truck_route)

        target = next((s for s in cand.sorties_by_drone[u] if s.index == p), None)
        if target is None:
            return None

        original_endpoint = target.launch if side == "launch" else target.recovery

        for node in cand.truck_route:
            if node in {self.data.depot_start_id, self.data.depot_end_id}: continue
            if node == original_endpoint: continue
            if side == "launch" and node == target.recovery: continue
            if side == "recovery" and node == target.launch: continue

            l_node = node if side == "launch" else target.launch
            r_node = node if side == "recovery" else target.recovery
            if l_node in pos and r_node in pos and pos[l_node] >= pos[r_node]:
                continue

            temp_s = Sortie(u, target.index, l_node, r_node, target.customers)
            if not is_sortie_feasible_fast(temp_s, self.data):
                continue

            if side == "launch":
                target.launch = node
            else:
                target.recovery = node

            ev2 = evaluate_solution(self.data, cand)
            if ev2.feasible:
                delta = ev2.objective - base.objective
                if delta < best_delta:
                    best_delta = delta
                    best_sol = cand.clone()

            if side == "launch":
                target.launch = original_endpoint
            else:
                target.recovery = original_endpoint

        return best_sol

    def op11(self, cur: Solution, ev: Evaluation, it: int) -> Optional[Solution]:
        cand = cur.clone()
        lr = launch_recovery_nodes(cand)
        candidates = [c for c in cand.truck_route if
                      c in self.data.Nku and c not in lr and c not in {self.data.depot_start_id,
                                                                       self.data.depot_end_id}]
        if not candidates:
            return None
        c = self.rng.choice(candidates)
        idx = cand.truck_route.index(c)
        if idx <= 0 or idx >= len(cand.truck_route) - 1:
            return None
        launch = cand.truck_route[idx - 1]
        recovery = cand.truck_route[idx + 1]
        if launch == recovery:
            return None
        cand.truck_route.pop(idx)
        insert_sortie_by_order(cand,
                               Sortie(drone=self.data.U[0], index=1, launch=launch, recovery=recovery, customers=[c]))
        cand.normalize(route_positions(cand.truck_route))
        return cand

    def op12(self, cur: Solution, ev: Evaluation, it: int) -> Optional[Solution]:
        cand = cur.clone()
        multi = [(u, s.index) for u in self.data.U for s in cand.sorties_by_drone.get(u, []) if len(s.customers) >= 2]
        if not multi:
            return None
        u, p = self.rng.choice(multi)
        s = next((x for x in cand.sorties_by_drone[u] if x.index == p), None)
        if s is None:
            return None
        c = self.rng.choice(s.customers)
        s.customers.remove(c)
        if not s.customers:
            remove_sortie(cand, u, p)
        best = try_insert_customer_into_existing_sortie(
            cand,
            c,
            self.data,
            metric="total",
            excluded_sortie=(u, p),
        )
        if best is None:
            best = try_create_new_sortie_for_customer(cand, c, self.data, metric="total")
        if best is None:
            return None
        best.normalize(route_positions(best.truck_route))
        return best

    def op13(self, cur: Solution, ev: Evaluation, it: int) -> Optional[Solution]:
        cand = cur.clone()
        lr = launch_recovery_nodes(cand)
        truck_customers = [c for c in cand.truck_route if c in self.data.C and c not in lr]
        if not truck_customers:
            return None
        k = operator_remove_counts(len(truck_customers), it, self.cfg.max_iter,
                                   [(0.25, 0.40), (0.15, 0.30), (0.10, 0.20), (0.05, 0.15)], self.rng, self.cfg)
        max_wait = max(ev.truck_wait.get(c, 0.0) for c in truck_customers)
        if max_wait > 1e-9:
            ranked = sorted(truck_customers, key=lambda c: ev.truck_wait.get(c, 0.0), reverse=True)
        else:
            ranked = sorted(
                truck_customers,
                key=lambda c: (
                    self.data.customers[c].due - self.data.customers[c].ready,
                    self.data.customers[c].due,
                    c,
                ),
            )
        remove_list = ranked[:min(k, len(ranked))]
        remove_customers_from_solution(cand, remove_list, cascade_if_launch_recovery=False)
        self.rng.shuffle(remove_list)
        remaining = set(remove_list)
        for c in remove_list:
            remaining.discard(c)
            nxt = try_best_total_insertion(cand, c, self.data, allowed_missing=remaining)
            if nxt is None:
                return None
            cand = nxt
        cand.normalize(route_positions(cand.truck_route))
        return cand

    def op14(self, cur: Solution, ev: Evaluation, it: int) -> Optional[Solution]:
        all_customers = list(self.data.C)
        if not all_customers:
            return None
        k = operator_remove_counts(len(all_customers), it, self.cfg.max_iter,
                                   [(0.10, 0.20), (0.10, 0.15), (0.05, 0.10), (0.01, 0.10)], self.rng, self.cfg)
        ranked: List[Tuple[float, int]] = []
        for c in all_customers:
            t = ev.truck_tardiness.get(c, 0.0) + ev.drone_tardiness.get(c, 0.0)
            ranked.append((t, c))
        ranked.sort(reverse=True)
        if ranked and ranked[0][0] <= 1e-9:
            drone_service = {
                c: service_time
                for (_, _, c), service_time in ev.drone_customer_service_time.items()
            }
            ranked = []
            for c in all_customers:
                if ev.customer_mode.get(c) == "drone":
                    service_time = drone_service.get(c, 0.0)
                else:
                    service_time = max(
                        ev.truck_arrival.get(c, 0.0),
                        self.data.customers[c].ready,
                    )
                slack = self.data.customers[c].due - service_time
                ranked.append((-slack, c))
            ranked.sort(reverse=True)
        remove_list = [c for _, c in ranked[:min(k, len(ranked))]]
        return self._remove_and_reinsert_total(cur, remove_list, random_order=True)

    def op15(self, cur: Solution, ev: Evaluation, it: int) -> Optional[Solution]:
        cand = cur.clone()
        gains: List[Tuple[float, int]] = []
        for u in self.data.U:
            for s in cand.sorties_by_drone.get(u, []):
                nodes = [s.launch] + s.customers + [s.recovery]
                for idx in range(1, len(nodes) - 1):
                    c = nodes[idx]
                    if c not in self.data.C:
                        continue
                    prev_n = nodes[idx - 1]
                    next_n = nodes[idx + 1]
                    gain = self.data.drone_dist[(prev_n, c)] + self.data.drone_dist[(c, next_n)] - self.data.drone_dist[
                        (prev_n, next_n)]
                    gains.append((gain, c))
        if not gains:
            return None
        gains.sort(reverse=True)
        unique: List[int] = []
        seen: Set[int] = set()
        for _, c in gains:
            if c not in seen:
                unique.append(c)
                seen.add(c)
        k = operator_remove_counts(len(unique), it, self.cfg.max_iter,
                                   [(0.25, 0.40), (0.15, 0.30), (0.10, 0.20), (0.05, 0.15)], self.rng, self.cfg)
        remove_list = unique[:min(k, len(unique))]
        return self._remove_and_reinsert_total(cand, remove_list, random_order=True)

    def op16(self, cur: Solution, ev: Evaluation, it: int) -> Optional[Solution]:
        cand = cur.clone()
        route = cand.truck_route
        n = len(route)
        if n < 4:
            return None

        best_sol = None
        best_obj = ev.objective
        candidate_drones = candidate_drones_for_new_sortie(cand, self.data)

        for i in range(1, n - 2):
            for length in range(1, 4):
                if i + length >= n:
                    continue
                segment = route[i: i + length]

                if any(c not in self.data.Nku for c in segment):
                    continue

                launch = route[i - 1]
                recovery = route[i + length]

                for u in candidate_drones:
                    new_s = Sortie(u, 999, launch, recovery, list(segment))
                    if not is_sortie_feasible_fast(new_s, self.data):
                        continue

                    test = cand.clone()
                    test.truck_route = test.truck_route[:i] + test.truck_route[i + length:]
                    test.sorties_by_drone[u].append(new_s)
                    ev2 = evaluate_solution(self.data, test)

                    if ev2.feasible and ev2.objective < best_obj:
                        best_obj = ev2.objective
                        best_sol = test

        return best_sol

    def op17(self, cur: Solution, ev: Evaluation, it: int) -> Optional[Solution]:
        cand = cur.clone()
        route = cand.truck_route
        n = len(route)
        if n < 4:
            return None

        best_sol = None
        best_obj = ev.objective
        base_moves: List[Tuple[int, int]] = []
        extra_moves: List[Tuple[float, int, int]] = []
        pressure_prefix = [0.0]
        for node in route:
            pressure = 0.0
            if node in self.data.C:
                pressure += ev.truck_tardiness.get(node, 0.0) * self.cfg.tardiness_cost_per_hour
                pressure += ev.truck_wait.get(node, 0.0) * self.cfg.wait_cost_per_hour
            pressure_prefix.append(pressure_prefix[-1] + pressure)

        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                a, b = route[i - 1], route[i]
                c, d = route[j], route[j + 1]

                delta_dist = self.data.truck_dist[(a, c)] + self.data.truck_dist[(b, d)] - self.data.truck_dist[
                    (a, b)] - self.data.truck_dist[(c, d)]
                if delta_dist < -1e-6 or (it >= self.cfg.max_iter and delta_dist <= 1e-6):
                    base_moves.append((i, j))
                    continue

                transport_delta = delta_dist * (
                    self.cfg.truck_unit_cost_per_hour / self.cfg.truck_speed
                    + self.cfg.carbon_price_per_kg * self.cfg.truck_emission_kg_per_km
                )
                segment_pressure = pressure_prefix[j + 1] - pressure_prefix[i]
                surrogate = transport_delta - 0.50 * segment_pressure
                extra_moves.append((surrogate, i, j))

        extra_moves.sort(key=lambda item: item[0])
        candidate_moves: List[Tuple[int, int, bool]] = [
            (i, j, False) for i, j in base_moves
        ]
        candidate_moves.extend((i, j, True) for _, i, j in extra_moves)

        extra_evaluated = 0
        for i, j, is_extra in candidate_moves:
            if is_extra and extra_evaluated >= self.cfg.two_opt_extra_candidates:
                break
            new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
            pos_check = route_positions(new_route)

            valid_launches = True
            for u in self.data.U:
                for s in cand.sorties_by_drone.get(u, []):
                    if s.launch in pos_check and s.recovery in pos_check:
                        if pos_check[s.launch] >= pos_check[s.recovery]:
                            valid_launches = False
                            break
                if not valid_launches:
                    break

            if not valid_launches:
                continue

            if is_extra:
                extra_evaluated += 1

            test = cand.clone()
            test.truck_route = new_route
            ev2 = evaluate_solution(self.data, test)

            if ev2.feasible and ev2.objective < best_obj:
                best_obj = ev2.objective
                best_sol = test

        return best_sol

    # ==================== Soft-time-window neighborhoods ====================

    def op18(self, cur: Solution, ev: Evaluation, it: int) -> Optional[Solution]:
        """Remove and repair a contiguous block selected by penalty pressure."""
        cand = cur.clone()
        route = cand.truck_route
        if len(route) <= 3:
            return None

        penalties: List[Tuple[float, int]] = []
        for idx, node in enumerate(route):
            if node not in self.data.C:
                continue

            wait_cost = ev.truck_wait.get(node, 0.0) * self.cfg.wait_cost_per_hour
            tard_cost = ev.truck_tardiness.get(node, 0.0) * self.cfg.tardiness_cost_per_hour

            drone_tard_cost = 0.0
            for u in self.data.U:
                for s in cand.sorties_by_drone.get(u, []):
                    if s.launch == node or s.recovery == node:
                        drone_tard_cost += sum(
                            ev.drone_tardiness.get(c, 0.0) for c in s.customers) * self.cfg.tardiness_cost_per_hour

            total_penalty = wait_cost + tard_cost + drone_tard_cost
            if total_penalty > 1e-3:
                penalties.append((total_penalty, idx))

        if not penalties:
            return None

        penalties.sort(reverse=True)
        _, target_idx = self.rng.choice(penalties[:min(3, len(penalties))])

        k = operator_remove_counts(len(self.data.C), it, self.cfg.max_iter,
                                   [(0.25, 0.40), (0.15, 0.30), (0.10, 0.20), (0.05, 0.15)], self.rng, self.cfg)

        start_idx = max(1, int(target_idx - k * 0.5))
        end_idx = min(len(route) - 1, start_idx + k)

        remove_list = [route[i] for i in range(start_idx, end_idx) if route[i] in self.data.C]
        if not remove_list:
            return None

        return self._remove_and_reinsert_total(cand, remove_list, random_order=True)

    def op19(self, cur: Solution, ev: Evaluation, it: int) -> Optional[Solution]:
        """Reconstruct a sortie selected by recovery-wait cost."""
        cand = cur.clone()
        bottlenecks: List[Tuple[float, int, int]] = []

        for u in self.data.U:
            for s in cand.sorties_by_drone.get(u, []):
                wait_time = ev.truck_wait.get(s.recovery, 0.0)
                if self.cfg.experiment_causal_wait:
                    node = s.recovery
                    other_events = [ev.truck_arrival.get(node, 0.0)]
                    if node in self.data.customers:
                        other_events.append(self.data.customers[node].ready)
                    for v in self.data.U:
                        for other in cand.sorties_by_drone.get(v, []):
                            key = (v, other.index)
                            if other.launch == node:
                                other_events.append(ev.drone_launch_time.get(key, 0.0))
                            if other.recovery == node and (v, other.index) != (u, s.index):
                                other_events.append(ev.drone_recovery_time.get(key, 0.0))
                    wait_time = max(0.0, ev.truck_departure.get(node, 0.0) - max(other_events))
                if wait_time > 1e-3:
                    cost = wait_time * self.cfg.wait_cost_per_hour
                    bottlenecks.append((cost, u, s.index))

        if not bottlenecks:
            return None

        bottlenecks.sort(reverse=True)
        _, target_u, target_p = self.rng.choice(bottlenecks[:min(3, len(bottlenecks))])

        removed = remove_sortie(cand, target_u, target_p)
        if not removed:
            return None

        self.rng.shuffle(removed)
        remaining = set(c for c in removed if c in self.data.C)
        for c in removed:
            remaining.discard(c)
            nxt = try_best_total_insertion(cand, c, self.data, allowed_missing=remaining)
            if nxt is None:
                return None
            cand = nxt

        cand.normalize(route_positions(cand.truck_route))
        return cand

    def op20(self, cur: Solution, ev: Evaluation, it: int) -> Optional[Solution]:
        """Relocate a short truck-route block under full feasibility evaluation."""
        cand = cur.clone()
        route = cand.truck_route
        n = len(route)
        if n < 6:
            return None

        best_sol = None
        best_obj = ev.objective

        max_block_len = min(4, n - 3)
        if max_block_len < 2:
            return None
        if self.cfg.experiment_legal_blocks:
            protected = launch_recovery_nodes(cand)
            blocks = [(start, length) for length in range(2, max_block_len + 1)
                      for start in range(1, n - length)
                      if all(node not in protected for node in route[start:start + length])]
            if not blocks:
                return None
            start_idx, block_len = self.rng.choice(blocks)
        else:
            block_len = self.rng.randint(2, max_block_len)
        max_start_idx = n - 1 - block_len
        if max_start_idx < 1:
            return None
        if not self.cfg.experiment_legal_blocks:
            start_idx = self.rng.randint(1, max_start_idx)
        block = route[start_idx: start_idx + block_len]
        if any(node in {self.data.depot_start_id, self.data.depot_end_id} for node in block):
            return None

        lr_nodes = launch_recovery_nodes(cand)
        if any(node in lr_nodes for node in block):
            return None

        rem_route = route[:start_idx] + route[start_idx + block_len:]

        for i in range(1, len(rem_route)):
            if self.cfg.experiment_legal_blocks and i == start_idx:
                continue
            test = cand.clone()
            test.truck_route = rem_route[:i] + block + rem_route[i:]
            test.normalize(route_positions(test.truck_route))

            test_ev = evaluate_solution(self.data, test)
            if test_ev.feasible and test_ev.objective < best_obj:
                best_obj = test_ev.objective
                best_sol = test

        return best_sol

    def op21(self, cur: Solution, ev: Evaluation, it: int) -> Optional[Solution]:
        """Destroy and reinsert a compact neighborhood around the most tardy customers."""
        tardiness = {
            c: ev.truck_tardiness.get(c, 0.0) + ev.drone_tardiness.get(c, 0.0)
            for c in self.data.C
        }
        ranked = [c for c in sorted(self.data.C, key=lambda node: tardiness[node], reverse=True)
                  if tardiness[c] > 1e-9]
        if not ranked:
            return None

        pool_size = max(1, min(int(self.cfg.tardiness_directed_top_pool), len(ranked)))
        target = self.rng.choice(ranked[:pool_size])
        stage = stage_of_iteration(it, self.cfg.max_iter, self.cfg)
        stage_cap = [
            self.cfg.tardiness_directed_max_remove,
            self.cfg.tardiness_directed_max_remove,
            max(self.cfg.tardiness_directed_min_remove, self.cfg.tardiness_directed_max_remove - 1),
            max(self.cfg.tardiness_directed_min_remove, self.cfg.tardiness_directed_max_remove - 2),
        ][stage]
        max_remove = max(self.cfg.tardiness_directed_min_remove, int(stage_cap))

        selected: List[int] = [target]
        selected_set = {target}

        # Keep customers from the same late sortie together so that reinsertion can
        # jointly change their launch/recovery pair instead of moving one in isolation.
        for u in self.data.U:
            for sortie in cur.sorties_by_drone.get(u, []):
                if target not in sortie.customers:
                    continue
                peers = sorted(sortie.customers, key=lambda c: tardiness.get(c, 0.0), reverse=True)
                for customer in peers:
                    if customer not in selected_set and len(selected) < max_remove:
                        selected.append(customer)
                        selected_set.add(customer)

        # For a truck-served late customer, include immediate route neighbors to
        # alter downstream timing and launch/recovery synchronization.
        if target in cur.truck_route:
            target_pos = cur.truck_route.index(target)
            for offset in (-1, 1, -2, 2):
                pos = target_pos + offset
                if not (1 <= pos < len(cur.truck_route) - 1):
                    continue
                customer = cur.truck_route[pos]
                if customer in self.data.C and customer not in selected_set and len(selected) < max_remove:
                    selected.append(customer)
                    selected_set.add(customer)

        # Complete the neighborhood with other high-tardiness customers. Insertion
        # order is then sorted by tardiness so the most critical customers claim
        # their best positions first.
        for customer in ranked:
            if customer not in selected_set and len(selected) < max_remove:
                selected.append(customer)
                selected_set.add(customer)

        if len(selected) < self.cfg.tardiness_directed_min_remove:
            return None
        selected.sort(key=lambda c: tardiness.get(c, 0.0), reverse=True)
        return self._remove_and_reinsert_total(cur, selected, random_order=False)


def serialize_solution(
    sol: Solution,
    ev: Evaluation,
    data: ProblemData,
    *,
    seed: Optional[int] = None,
    runtime_sec: Optional[float] = None,
    initial_evaluation: Optional[Evaluation] = None,
) -> Dict[str, Any]:
    used_drones = sum(1 for u in data.U if sol.sorties_by_drone.get(u))
    number_of_sorties = sum(len(sol.sorties_by_drone.get(u, [])) for u in data.U)
    arrival_based_launch = data.cfg.allow_launch_during_truck_ready_wait
    return {
        "algorithm": "IALNS-arrival-launch" if arrival_based_launch else "IALNS-service-first",
        "timing_policy": {
            "allow_launch_during_truck_ready_wait": arrival_based_launch,
            "launch_lower_bound": (
                "max(truck_arrival, drone_available)"
                if arrival_based_launch
                else "max(truck_service_ready, node_recovery_completion, drone_available)"
            ),
        },
        "seed": seed,
        "runtime_sec": runtime_sec,
        "initial_objective": initial_evaluation.objective if initial_evaluation is not None else None,
        "objective": ev.objective,
        "components": {"C1": ev.c1, "C2": ev.c2, "C3": ev.c3, "C4": ev.c4, "C5": ev.c5},
        "metrics": {
            "truck_distance": ev.truck_distance,
            "drone_distance": ev.drone_distance,
            "carbon_emissions_kg": ev.carbon_emissions,
            "number_of_drones_used": used_drones,
            "number_of_sorties": number_of_sorties,
            "late_customers": ev.late_customers,
            "total_tardiness": ev.total_tardiness,
        },
        "feasible": ev.feasible,
        "violations": ev.violations,
        "truck_route": sol.truck_route,
        "schedule": {
            "truck_arrival": {str(node): value for node, value in ev.truck_arrival.items()},
            "truck_departure": {str(node): value for node, value in ev.truck_departure.items()},
            "truck_wait": {str(node): value for node, value in ev.truck_wait.items()},
            "drone_launch_time": {
                f"U{u}-P{p}": value for (u, p), value in ev.drone_launch_time.items()
            },
            "drone_recovery_time": {
                f"U{u}-P{p}": value for (u, p), value in ev.drone_recovery_time.items()
            },
            "drone_customer_service_time": {
                f"U{u}-P{p}-C{customer}": value
                for (u, p, customer), value in ev.drone_customer_service_time.items()
            },
        },
        "sorties": {
            str(u): [
                {
                    "index": s.index,
                    "launch": s.launch,
                    "customers": s.customers,
                    "recovery": s.recovery,
                }
                for s in sol.sorties_by_drone.get(u, [])
            ]
            for u in data.U
        },
    }


def try_plot_solution(
        sol: Solution,
        history: List[float],
        data: ProblemData,
        out_dir: Path,
        plot_route_map: bool,
        plot_objective_history: bool,
) -> List[Path]:
    if not plot_route_map and not plot_objective_history:
        return []

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[Plot] matplotlib is not available. Skip plotting.")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    if plot_route_map:
        fig, ax = plt.subplots(figsize=(9, 7))

        x_nk = []
        y_nk = []
        x_nku = []
        y_nku = []
        for c in data.C:
            if c in data.Nk:
                x_nk.append(data.customers[c].x)
                y_nk.append(data.customers[c].y)
            else:
                x_nku.append(data.customers[c].x)
                y_nku.append(data.customers[c].y)
        ax.scatter(x_nk, y_nk, c="tab:blue", label="Nk (truck-only)")
        ax.scatter(x_nku, y_nku, c="tab:orange", label="Nku")

        sx, sy = data.node_xy[data.depot_start_id]
        ex, ey = data.node_xy[data.depot_end_id]
        ax.scatter([sx], [sy], c="green", s=100, marker="s", label="Depot start")
        ax.scatter([ex], [ey], c="red", s=100, marker="s", label="Depot end")

        tx = [data.node_xy[n][0] for n in sol.truck_route]
        ty = [data.node_xy[n][1] for n in sol.truck_route]
        ax.plot(tx, ty, c="black", linewidth=2, label="Truck route")

        colors = ["tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]
        for u in data.U:
            color = colors[(u - 1) % len(colors)]
            for s in sol.sorties_by_drone.get(u, []):
                nodes = s.route_nodes()
                dx = [data.node_xy[n][0] for n in nodes]
                dy = [data.node_xy[n][1] for n in nodes]
                ax.plot(dx, dy, c=color, linestyle="--", alpha=0.85, label=f"Drone {u}" if s.index == 1 else None)

        ax.set_title("Truck-Drone Collaborative Routing")
        ax.legend(loc="best")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        route_path = out_dir / "best_routes.png"
        fig.savefig(route_path, dpi=160)
        saved.append(route_path)
        plt.close(fig)

    if plot_objective_history and history:
        fig2, ax2 = plt.subplots(figsize=(9, 4))
        ax2.plot(history, c="tab:green", linewidth=1.8)
        ax2.set_title("Best Objective over Iterations")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Objective")
        ax2.grid(alpha=0.2)
        fig2.tight_layout()
        objective_path = out_dir / "objective_history.png"
        fig2.savefig(objective_path, dpi=160)
        saved.append(objective_path)
        plt.close(fig2)

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IALNS for truck-drone routing with arrival-based drone launch timing"
    )
    parser.add_argument("--customers", type=str, default="", help="Path to customers.csv")
    parser.add_argument(
        "--excel",
        type=str,
        default="",
        help="Path to customers.xlsx (id=0 depot; columns: id,x,y,demand,can_drone,ready,due,max_outside)",
    )
    parser.add_argument("--config", type=str, default="", help="Path to config.json")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (omit for time-based seed)")
    parser.add_argument("--max-iter", type=int, default=None, help="Override ALNS max iterations")
    parser.add_argument("--stage-cuts", type=str, default=None,
                        help="Comma-separated stage cut ratios, e.g. 0.08,0.18,0.34")
    parser.add_argument("--sa-start", type=float, default=None, help="SA start temperature ratio")
    parser.add_argument("--sa-end", type=float, default=None, help="SA end temperature ratio")
    parser.add_argument("--sa-freeze", type=float, default=None, help="SA cooling completion ratio")
    parser.add_argument("--reconstruct-patience", type=int, default=None,
                        help="No-best-improvement iterations before large reconstruct")
    parser.add_argument("--finish-patience", type=int, default=None,
                        help="No-best-improvement iterations before early stop")
    parser.add_argument("--plateau-polish-patience", type=int, default=None,
                        help="No-best-improvement iterations before one plateau polish pass")
    parser.add_argument("--polish-passes", type=int, default=None, help="Override final polish pass count")
    parser.add_argument("--plateau-polish-passes", type=int, default=None, help="Override plateau polish pass count")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--synthetic-n", type=int, default=25, help="Synthetic customer count")
    parser.add_argument(
        "--large-reconstruct",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable large destroy-reconstruct after no improvement cycles",
    )
    parser.add_argument(
        "--early-stop",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable early stop after no historical-best improvement cycles",
    )
    parser.add_argument(
        "--polish",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable final deterministic polishing",
    )
    parser.add_argument(
        "--plateau-polish",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable deterministic polishing during no-improvement plateaus",
    )
    parser.add_argument(
        "--launch-during-ready-wait",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow drone launch after truck arrival while waiting for customer readiness",
    )
    parser.add_argument("--output-dir", type=str, default="output", help="Output folder")
    parser.add_argument("--plot", action="store_true", help="Force generating both plots if matplotlib is available")
    return parser.parse_args()


def apply_cli_config_overrides(cfg: Config, args: argparse.Namespace) -> None:
    if args.max_iter is not None:
        cfg.max_iter = args.max_iter
    if args.stage_cuts is not None:
        parts = [float(x.strip()) for x in args.stage_cuts.split(",") if x.strip()]
        if len(parts) != 3 or not (0.0 < parts[0] < parts[1] < parts[2] < 1.0):
            raise ValueError("--stage-cuts must be three increasing ratios between 0 and 1")
        cfg.stage_cut1, cfg.stage_cut2, cfg.stage_cut3 = parts
    if args.sa_start is not None:
        cfg.sa_start_ratio = args.sa_start
    if args.sa_end is not None:
        cfg.sa_end_ratio = args.sa_end
    if args.sa_freeze is not None:
        cfg.sa_freeze_ratio = args.sa_freeze
    if args.reconstruct_patience is not None:
        cfg.cycle_no_improve_reconstruct = args.reconstruct_patience
    if args.finish_patience is not None:
        cfg.cycle_no_improve_finish = args.finish_patience
    if args.plateau_polish_patience is not None:
        cfg.cycle_no_improve_polish = args.plateau_polish_patience
    if args.polish_passes is not None:
        cfg.polish_passes = args.polish_passes
    if args.plateau_polish_passes is not None:
        cfg.plateau_polish_passes = args.plateau_polish_passes
    if args.large_reconstruct is not None:
        cfg.enable_large_reconstruct = args.large_reconstruct
    if args.early_stop is not None:
        cfg.enable_early_stop = args.early_stop
    if args.polish is not None:
        cfg.enable_final_polish = args.polish
    if args.plateau_polish is not None:
        cfg.enable_plateau_polish = args.plateau_polish
    if args.launch_during_ready_wait is not None:
        cfg.allow_launch_during_truck_ready_wait = args.launch_during_ready_wait
    if args.plot:
        cfg.plot_route_map = True
        cfg.plot_objective_history = True


def resolve_seed(seed: Optional[int]) -> int:
    return seed if seed is not None else (time.time_ns() & 0xFFFFFFFF)


def build_problem_from_args(args: argparse.Namespace, seed: int) -> ProblemData:
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"config file not found: {cfg_path}")
        cfg = load_config_json(cfg_path)
    else:
        cfg = Config()
    apply_cli_config_overrides(cfg, args)

    if args.synthetic:
        return create_synthetic_problem(seed=seed, n_customers=args.synthetic_n, cfg=cfg)

    customers: Dict[int, Customer]
    if args.excel:
        excel_path = Path(args.excel)
        if not excel_path.exists():
            raise FileNotFoundError(f"excel file not found: {excel_path}")
        customers = load_customers_excel(excel_path, cfg)
    elif args.customers:
        customers_path = Path(args.customers)
        if not customers_path.exists():
            raise FileNotFoundError(f"customers file not found: {customers_path}")
        customers = load_customers_csv(customers_path, cfg)
    else:
        return create_synthetic_problem(seed=seed, n_customers=args.synthetic_n, cfg=cfg)

    return ProblemData(
        customers=customers,
        cfg=cfg,
        depot_start=(cfg.depot_start_x, cfg.depot_start_y),
        depot_end=(cfg.depot_end_x, cfg.depot_end_y),
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = resolve_seed(args.seed)
    print("[Seed] using:", seed)
    data = build_problem_from_args(args, seed)
    print(
        "[Timing] launch during customer-ready waiting:",
        data.cfg.allow_launch_during_truck_ready_wait,
    )
    rng = random.Random(seed)
    init_sol = create_initial_solution(data, rng)
    init_eval = evaluate_solution(data, init_sol)
    print("[Init] feasible:", init_eval.feasible, "objective:", round(init_eval.objective, 6))
    if not init_eval.feasible:
        print("[Init] violations (first 20):")
        for v in init_eval.violations[:20]:
            print(" -", v)

    optimizer = ALNSOptimizer(data=data, seed=seed)
    solve_start = time.perf_counter()
    best_sol, best_eval = optimizer.run(init_solution=init_sol)
    runtime_sec = time.perf_counter() - solve_start
    best_sol.normalize(route_positions(best_sol.truck_route))
    best_eval = evaluate_solution(data, best_sol)

    print("\n=== Best Solution ===")
    print("Feasible:", best_eval.feasible)
    print("Objective:", round(best_eval.objective, 6))
    print(
        "C1:", round(best_eval.c1, 6),
        "C2:", round(best_eval.c2, 6),
        "C3:", round(best_eval.c3, 6),
        "C4:", round(best_eval.c4, 6),
        "C5:", round(best_eval.c5, 6),
    )
    print("Truck route:", best_sol.truck_route)
    for u in data.U:
        for s in best_sol.sorties_by_drone.get(u, []):
            print(f"Drone {u}, Sortie {s.index}: {s.launch} -> {s.customers} -> {s.recovery}")

    payload = serialize_solution(
        best_sol,
        best_eval,
        data,
        seed=seed,
        runtime_sec=runtime_sec,
        initial_evaluation=init_eval,
    )
    out_json = out_dir / "best_solution.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", out_json)

    plot_paths = try_plot_solution(
        best_sol,
        optimizer.history_obj,
        data,
        out_dir,
        plot_route_map=data.cfg.plot_route_map,
        plot_objective_history=data.cfg.plot_objective_history,
    )
    for path in plot_paths:
        print("Plot saved:", path)


if __name__ == "__main__":
    main()
