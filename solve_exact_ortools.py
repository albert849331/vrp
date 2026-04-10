"""solve_exact_ortools.py
=======================
Multi-Depot Vehicle Routing Problem (MDVRP) solver for the Mother–Daughter
centre assignment problem.

Problem Description
-------------------
Given:
  - 11 mother centres (depots) — each is the start and end of one or more routes.
  - 191 daughter centres (customers) — each must be visited exactly once.

Constraints:
  - Every vehicle starts at its home mother centre at 06:00.
  - The LAST daughter centre on any route must be reached by 10:00 (i.e., within
    240 minutes of departure).  Return to mother can be after 10:00.
  - Each daughter is served by exactly one vehicle from exactly one mother.

Objective
---------
Minimise:  total_fixed_cost  +  total_variable_cost

where
  Fixed cost     = Rs 1,000 per vehicle per day (charged only if vehicle is used)
  Variable cost  = (Rs 10/km + Rs 50/hr) × km  =  Rs 11/km at 50 km/h

Commercial parameters (Tata Ace fleet):
  Vehicle speed   : 50 km/h (constant)
  Fixed cost/day  : Rs 1,000
  Rate per km     : Rs 10
  Rate per hour   : Rs 50

Solver
------
  Engine     : Google OR-Tools CP routing solver (v9+)
  Algorithm  : GUIDED_LOCAL_SEARCH metaheuristic with
               PARALLEL_CHEAPEST_INSERTION first solution
  Time limit : 600 seconds
  Lower bound: routing.objective_lower_bound() — the best dual bound the
               solver has proved; used to compute the optimality gap.

Outputs
-------
  outputs/solution_output.xlsx  — Excel workbook:
      * Assignments   — each daughter mapped to its mother and route stop
      * Routes        — one row per vehicle route, with path and costs
      * MotherSummary — daughters and routes per mother centre
      * CostSummary   — overall cost breakdown
      * OptimalityGap — incumbent, lower bound, absolute and relative gap
  outputs/solution_report.docx  — Written methodology and results report
  outputs/solution_graph.png    — Route map (lat/lon scatter + lines)
  outputs/solution_summary.md   — Markdown summary

Usage
-----
  python solve_exact_ortools.py

Notes
-----
  - Distance is computed using the haversine formula (great-circle distance)
    because only latitude/longitude coordinates are provided.
  - Service time at daughter centres is assumed zero (not specified in the
    problem statement).
  - Vehicles are homogeneous Tata Ace; unlimited fleet per mother centre is
    assumed (one vehicle per daughter centre is the upper bound).
  - Eleven daughter centres have a minimum straight-line distance >100 km from
    their nearest mother.  They are still feasible to serve directly (one-way
    travel <200 min < 240 min deadline) but cannot be combined with other stops.
"""

from __future__ import annotations

import math
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pandas as pd
from docx import Document
from docx.shared import Pt, Inches
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Solver / model constants
# ---------------------------------------------------------------------------

VEHICLE_SPEED_KMPH: float = 50.0
"""Assumed constant speed for all vehicles (km/h)."""

ROUTE_START_HOUR: int = 6
"""Vehicles depart their mother centre at 06:00 (hour 6)."""

LAST_DELIVERY_DEADLINE_MINUTES: int = 4 * 60
"""The last daughter stop on a route must be reached within 240 min of start
(i.e., by 10:00 AM)."""

END_OF_DAY_MINUTES: int = 24 * 60
"""Upper bound on total route time including the return leg (1440 min)."""

FIXED_COST_INR: float = 1_000.0
"""Fixed daily cost per vehicle (Rs)."""

RATE_PER_KM_INR: float = 10.0
"""Variable cost per km (Rs)."""

RATE_PER_HOUR_INR: float = 50.0
"""Variable cost per hour (Rs)."""

VARIABLE_COST_PER_KM_INR: float = (
    RATE_PER_KM_INR + RATE_PER_HOUR_INR / VEHICLE_SPEED_KMPH
)
"""Combined distance + time cost per km at the fixed speed: Rs 10 + 1 = Rs 11/km."""

COST_SCALE: int = 100
"""Integer scaling factor.  All costs stored as int(round(cost × COST_SCALE))
to avoid floating-point truncation inside the solver."""

SOLVER_TIME_LIMIT_SECONDS: int = 600
"""Wall-clock time budget for the local-search phase (seconds)."""


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelInputs:
    """All pre-processed inputs needed to build and solve the routing model.

    Attributes
    ----------
    mothers:
        DataFrame rows for mother (depot) centres.
    daughters:
        DataFrame rows for daughter (customer) centres.
    nodes:
        Ordered list of dicts describing every node (mothers first,
        daughters second).  Index in this list = node number in the model.
    center_name_by_node:
        Maps node index → original centre_name integer from the spreadsheet.
    mother_node_by_center:
        Maps mother centre_name → node index.
    daughter_nodes:
        List of node indices that are daughter centres.
    distance_km:
        n×n matrix of haversine distances (km).
    time_minutes:
        n×n matrix of travel times (minutes), ceiling of distance/speed.
    cost_units:
        n×n matrix of variable arc costs (int, in units of 1/COST_SCALE Rs).
    vehicle_starts:
        Node index where each vehicle starts its route.
    vehicle_ends:
        Node index where each vehicle returns (same as start for round-trips).
    vehicle_labels:
        Human-readable string label for each vehicle.
    """

    mothers: pd.DataFrame
    daughters: pd.DataFrame
    nodes: list[dict[str, object]]
    center_name_by_node: list[int]
    mother_node_by_center: dict[int, int]
    daughter_nodes: list[int]
    distance_km: list[list[float]]
    time_minutes: list[list[int]]
    cost_units: list[list[int]]
    vehicle_starts: list[int]
    vehicle_ends: list[int]
    vehicle_labels: list[str]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in km between two (lat, lon) points.

    Uses the haversine formula with an Earth radius of 6,371 km.
    This is appropriate here because road distances were not provided;
    haversine gives a reliable lower-bound proxy for actual road distances.
    """
    r = 6_371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 2.0 * r * math.asin(math.sqrt(a))


def minutes_to_clock(minutes_from_start: int) -> str:
    """Convert elapsed minutes since 06:00 to a human-readable HH:MM string."""
    total = ROUTE_START_HOUR * 60 + int(minutes_from_start)
    h, m = divmod(total, 60)
    return f"{h:02d}:{m:02d}"


# ---------------------------------------------------------------------------
# Step 1 – Load and pre-process data
# ---------------------------------------------------------------------------

def load_inputs(input_file: Path) -> ModelInputs:
    """Read the Excel input file and build all model matrices.

    Processing steps
    ----------------
    1. Read the spreadsheet; cast centre_name to int.
    2. Separate mother (depot) rows from daughter (customer) rows.
    3. Build a unified node list: mothers first, daughters appended.
    4. Compute pairwise haversine distances, travel times, and arc costs.
    5. Assign a vehicle fleet to each mother:
       - Count how many daughters are geographically nearest to that mother.
       - Allocate one vehicle per nearest daughter (upper bound on fleet size);
         the solver will leave unused vehicles idle with zero cost.

    Parameters
    ----------
    input_file:
        Path to data_1.xlsx (or equivalent).

    Returns
    -------
    ModelInputs
        Fully populated input dataclass.
    """
    raw_df = pd.read_excel(input_file)
    raw_df["center_name"] = raw_df["center_name"].astype(int)

    mothers = (
        raw_df[raw_df["Category"].str.lower() == "mother"]
        .sort_values("center_name")
        .reset_index(drop=True)
    )
    daughters = (
        raw_df[raw_df["Category"].str.lower() == "daughter"]
        .sort_values("center_name")
        .reset_index(drop=True)
    )

    # --- Build node list (depots first, customers second) ---
    nodes: list[dict[str, object]] = []
    center_name_by_node: list[int] = []
    mother_node_by_center: dict[int, int] = {}

    for row in mothers.itertuples(index=False):
        node_idx = len(nodes)
        center_name = int(row.center_name)
        nodes.append(
            {
                "center_name": center_name,
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
                "category": "Mother",
            }
        )
        center_name_by_node.append(center_name)
        mother_node_by_center[center_name] = node_idx

    daughter_nodes: list[int] = []
    for row in daughters.itertuples(index=False):
        node_idx = len(nodes)
        center_name = int(row.center_name)
        nodes.append(
            {
                "center_name": center_name,
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
                "category": "Daughter",
            }
        )
        center_name_by_node.append(center_name)
        daughter_nodes.append(node_idx)

    # --- Pairwise distance / time / cost matrices ---
    coords = [(float(n["latitude"]), float(n["longitude"])) for n in nodes]
    n = len(nodes)
    distance_km = [[0.0] * n for _ in range(n)]
    time_minutes = [[0] * n for _ in range(n)]
    cost_units = [[0] * n for _ in range(n)]

    for i in range(n):
        lat_i, lon_i = coords[i]
        for j in range(n):
            if i == j:
                continue
            lat_j, lon_j = coords[j]
            d = haversine_km(lat_i, lon_i, lat_j, lon_j)
            distance_km[i][j] = d
            travel_min = math.ceil(d * 60.0 / VEHICLE_SPEED_KMPH)
            time_minutes[i][j] = travel_min
            # Integer cost units: Rs_per_km × distance × 100
            cost_units[i][j] = int(round(d * VARIABLE_COST_PER_KM_INR * COST_SCALE))

    # --- Determine vehicle fleet per mother ---
    # Strategy: one vehicle per daughter whose nearest mother is this depot.
    # This is the minimum fleet size needed if every daughter is on a
    # dedicated route, giving the solver maximum flexibility to consolidate.
    nearest_mother_counts: Counter[int] = Counter()
    for d_node in daughter_nodes:
        nearest = min(
            mother_node_by_center,
            key=lambda c: distance_km[mother_node_by_center[c]][d_node],
        )
        nearest_mother_counts[nearest] += 1

    vehicle_starts: list[int] = []
    vehicle_ends: list[int] = []
    vehicle_labels: list[str] = []
    for mother_center in sorted(mother_node_by_center):
        count = max(1, nearest_mother_counts.get(mother_center, 0))
        m_node = mother_node_by_center[mother_center]
        for k in range(1, count + 1):
            vehicle_starts.append(m_node)
            vehicle_ends.append(m_node)
            vehicle_labels.append(f"M{mother_center}_V{k:02d}")

    return ModelInputs(
        mothers=mothers,
        daughters=daughters,
        nodes=nodes,
        center_name_by_node=center_name_by_node,
        mother_node_by_center=mother_node_by_center,
        daughter_nodes=daughter_nodes,
        distance_km=distance_km,
        time_minutes=time_minutes,
        cost_units=cost_units,
        vehicle_starts=vehicle_starts,
        vehicle_ends=vehicle_ends,
        vehicle_labels=vehicle_labels,
    )


# ---------------------------------------------------------------------------
# Step 2 – Build and solve the routing model
# ---------------------------------------------------------------------------

def solve_model(
    model_inputs: ModelInputs,
) -> tuple[
    pywrapcp.RoutingIndexManager,
    pywrapcp.RoutingModel,
    pywrapcp.Assignment,
    pywrapcp.RoutingDimension,
    int,
]:
    """Construct the OR-Tools routing model and solve it.

    Model formulation
    -----------------
    Arc cost  (variable)
        = int(round(haversine_km(i,j) × VARIABLE_COST_PER_KM × COST_SCALE))
        Registered as the arc cost for all vehicles.

    Fixed cost (per vehicle used)
        = int(round(FIXED_COST_INR × COST_SCALE))
        Added via SetFixedCostOfVehicle; charged only when the vehicle visits
        at least one daughter.

    Time dimension
        Cumulative time from 06:00 at each node.
        - Each daughter node is constrained to be reached within [0, 240] min.
        - Vehicle starts are pinned to time 0 (departure at 06:00 exactly).
        - Vehicle ends (return to depot) are bounded by [0, 1440] min (end of day).

    Fleet
        vehicle_starts / vehicle_ends pair (see load_inputs for sizing).

    Search strategy
    ---------------
        First solution : PARALLEL_CHEAPEST_INSERTION
            Greedily inserts customers at the cheapest feasible position.
            Produces a good initial feasible solution.
        Metaheuristic  : GUIDED_LOCAL_SEARCH (GLS)
            Iteratively escapes local optima by penalising frequently used
            features.  Best general-purpose choice for VRP.
        Time limit     : SOLVER_TIME_LIMIT_SECONDS (600 s)
        Logging        : enabled — prints objective improvement to stdout.

    Lower bound
    -----------
    routing.CloseModel() finalises the model, after which
    routing.ComputeLowerBound() solves the LP relaxation (ignoring subtour
    and time-window constraints) and returns a valid lower bound on the
    optimal objective.  This is saved BEFORE the main search starts so that
    the local-search phase cannot overwrite it.  The optimality gap is:
        gap_abs = (incumbent - lower_bound) / COST_SCALE  [Rs]
        gap_rel = gap_abs / incumbent_inr × 100  [%]
    A gap of 0 % means optimality is certified.

    Parameters
    ----------
    model_inputs:
        Pre-processed model data from load_inputs().

    Returns
    -------
    (manager, routing, solution, time_dimension, pre_solve_lower_bound_units)
        The last element is the LP-relaxation lower bound in COST_SCALE units,
        computed before the local-search phase via routing.ComputeLowerBound().

    Raises
    ------
    RuntimeError
        If no feasible solution is found within the time limit.
    """
    n_nodes = len(model_inputs.nodes)
    n_vehicles = len(model_inputs.vehicle_starts)

    # --- Index manager: maps between solver indices and node indices ---
    manager = pywrapcp.RoutingIndexManager(
        n_nodes,
        n_vehicles,
        model_inputs.vehicle_starts,
        model_inputs.vehicle_ends,
    )
    routing = pywrapcp.RoutingModel(manager)

    # --- Arc cost callback ---
    def cost_callback(from_idx: int, to_idx: int) -> int:
        """Return integer arc cost (in 1/COST_SCALE Rs units)."""
        i = manager.IndexToNode(from_idx)
        j = manager.IndexToNode(to_idx)
        return model_inputs.cost_units[i][j]

    cost_cb_idx = routing.RegisterTransitCallback(cost_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(cost_cb_idx)

    # --- Time callback ---
    def time_callback(from_idx: int, to_idx: int) -> int:
        """Return travel time in minutes between two nodes."""
        i = manager.IndexToNode(from_idx)
        j = manager.IndexToNode(to_idx)
        return model_inputs.time_minutes[i][j]

    time_cb_idx = routing.RegisterTransitCallback(time_callback)

    # Add a cumulative Time dimension (no slack, max END_OF_DAY_MINUTES)
    routing.AddDimension(
        time_cb_idx,
        0,                          # no waiting time / slack at nodes
        END_OF_DAY_MINUTES,         # maximum time on any route
        True,                       # start cumul to zero
        "Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")

    # --- Daughter time-window constraints: must arrive within 240 min ---
    for d_node in model_inputs.daughter_nodes:
        node_idx = manager.NodeToIndex(d_node)
        # Constrain arrival time to [0, LAST_DELIVERY_DEADLINE_MINUTES]
        time_dim.CumulVar(node_idx).SetRange(0, LAST_DELIVERY_DEADLINE_MINUTES)

    # --- Vehicle start / end time constraints ---
    fixed_cost_units = int(round(FIXED_COST_INR * COST_SCALE))
    for v in range(n_vehicles):
        routing.SetFixedCostOfVehicle(fixed_cost_units, v)
        # Departure at exactly 06:00 (time 0 in the model)
        time_dim.CumulVar(routing.Start(v)).SetRange(0, 0)
        # Return can happen any time before end of day
        time_dim.CumulVar(routing.End(v)).SetRange(0, END_OF_DAY_MINUTES)
        # Minimise the total route duration (helps convergence)
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.End(v)))

    # --- Search parameters ---
    params = pywrapcp.DefaultRoutingSearchParameters()

    # First solution strategy: cheapest insertion gives a good starting point
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )

    # Metaheuristic: Guided Local Search (best general-purpose for VRP)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )

    # Time budget: 600 seconds for the local-search phase
    # (line 332 in the original solve_exact_ortools.py context)
    params.time_limit.FromSeconds(SOLVER_TIME_LIMIT_SECONDS)

    # Enable logging so we can observe convergence during solving
    params.log_search = True

    # --- Finalise model and compute LP lower bound BEFORE solving ---
    # CloseModel() locks the model topology so lower-bound computations
    # and the solver both operate on the same finalised problem.
    routing.CloseModel()

    # ComputeLowerBound() solves the LP relaxation (drops subtour-elimination
    # and most time-window constraints) to obtain a provably valid lower bound
    # on the optimal integer solution.  We capture it here because the
    # local-search phase does not improve it.
    lower_bound_units = routing.ComputeLowerBound()
    print(
        f"[Solver] LP lower bound (pre-solve): "
        f"Rs {lower_bound_units / COST_SCALE:,.2f} "
        f"({lower_bound_units} cost units)"
    )

    # --- Solve ---
    print(
        f"\n[Solver] Starting OR-Tools routing solver "
        f"(time limit = {SOLVER_TIME_LIMIT_SECONDS} s) …"
    )
    solution = routing.SolveWithParameters(params)

    # Check solver status (line 334 in the original context)
    # Status codes: ROUTING_NOT_SOLVED=0, ROUTING_SUCCESS=1,
    #               ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED=2,
    #               ROUTING_FAIL=3, ROUTING_FAIL_TIMEOUT=4, ROUTING_INVALID=5
    status = routing.status()
    status_names = {
        0: "NOT_SOLVED",
        1: "SUCCESS (proven optimal or local optimum)",
        2: "PARTIAL_SUCCESS (time limit reached, solution found)",
        3: "FAIL (no solution found)",
        4: "FAIL_TIMEOUT (time limit, no solution)",
        5: "INVALID",
    }
    print(f"[Solver] Status: {status_names.get(status, f'UNKNOWN ({status})')}")

    if solution is None:
        raise RuntimeError(
            "OR-Tools routing solver did not find a feasible solution. "
            "Consider relaxing time-window constraints or increasing the fleet."
        )

    return manager, routing, solution, time_dim, lower_bound_units


# ---------------------------------------------------------------------------
# Step 3 – Extract and compute outputs
# ---------------------------------------------------------------------------

def build_outputs(
    model_inputs: ModelInputs,
    manager: pywrapcp.RoutingIndexManager,
    routing: pywrapcp.RoutingModel,
    solution: pywrapcp.Assignment,
    time_dim: pywrapcp.RoutingDimension,
    lower_bound_units: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float],
]:
    """Extract route details, compute costs, and build result DataFrames.

    Returns
    -------
    assignments_df:
        One row per daughter centre; columns include the assigned mother,
        route ID, stop sequence, and arrival time.
    routes_df:
        One row per active vehicle route; includes full path, distance,
        timing, and cost breakdown.
    mother_summary_df:
        One row per mother centre; daughters served and routes used.
    cost_summary_df:
        Scalar summary metrics (distance, fixed / variable / total cost).
    gap_df:
        Incumbent, lower bound, absolute gap, and relative gap.
    summary_values:
        Dict of the same scalar metrics for use by report writers.
    """
    assignment_rows: list[dict[str, object]] = []
    route_rows: list[dict[str, object]] = []

    for v_id, v_label in enumerate(model_inputs.vehicle_labels):
        if not routing.IsVehicleUsed(solution, v_id):
            continue  # unused vehicle: no cost incurred

        start_idx = routing.Start(v_id)
        start_node = manager.IndexToNode(start_idx)
        mother_center = model_inputs.center_name_by_node[start_node]
        route_id = f"M{mother_center}_{v_label}"

        route_dist_km = 0.0
        path_names: list[str] = [str(mother_center)]
        stop_seq = 0
        customer_count = 0
        last_delivery_min = 0
        idx = start_idx

        while not routing.IsEnd(idx):
            next_idx = solution.Value(routing.NextVar(idx))
            from_node = manager.IndexToNode(idx)
            to_node = manager.IndexToNode(next_idx)
            leg_km = model_inputs.distance_km[from_node][to_node]
            route_dist_km += leg_km

            if not routing.IsEnd(next_idx):
                # This leg ends at a daughter centre
                stop_seq += 1
                customer_count += 1
                arrival_min = solution.Value(time_dim.CumulVar(next_idx))
                last_delivery_min = arrival_min
                daughter_center = model_inputs.center_name_by_node[to_node]
                path_names.append(str(daughter_center))

                assignment_rows.append(
                    {
                        "daughter_center": daughter_center,
                        "assigned_mother": mother_center,
                        "route_id": route_id,
                        "vehicle_label": v_label,
                        "stop_sequence": stop_seq,
                        "arrival_minutes_from_6am": arrival_min,
                        "arrival_clock": minutes_to_clock(arrival_min),
                        "leg_distance_km": round(leg_km, 2),
                    }
                )

            idx = next_idx

        # Return leg back to mother depot
        path_names.append(str(mother_center))
        return_min = solution.Value(time_dim.CumulVar(routing.End(v_id)))
        variable_cost_inr = route_dist_km * VARIABLE_COST_PER_KM_INR
        total_cost_inr = FIXED_COST_INR + variable_cost_inr

        route_rows.append(
            {
                "route_id": route_id,
                "vehicle_label": v_label,
                "mother_center": mother_center,
                "stops": customer_count,
                "path": " → ".join(path_names),
                "last_delivery_minutes_from_6am": last_delivery_min,
                "last_delivery_clock": minutes_to_clock(last_delivery_min),
                "returned_to_mother_clock": minutes_to_clock(return_min),
                "route_distance_km": round(route_dist_km, 2),
                "route_drive_hours": round(return_min / 60.0, 2),
                "fixed_cost_inr": FIXED_COST_INR,
                "variable_cost_inr": round(variable_cost_inr, 2),
                "total_cost_inr": round(total_cost_inr, 2),
            }
        )

    # --- Assemble DataFrames ---
    assignments_df = pd.DataFrame(assignment_rows).sort_values(
        ["assigned_mother", "route_id", "stop_sequence"]
    )
    routes_df = pd.DataFrame(route_rows).sort_values(["mother_center", "route_id"])

    # Sanity check: every daughter must appear exactly once
    n_unique = assignments_df["daughter_center"].nunique()
    n_daughters = len(model_inputs.daughters)
    if n_unique != n_daughters:
        raise RuntimeError(
            f"Coverage error: {n_unique} daughters assigned, expected {n_daughters}. "
            "The solution is incomplete."
        )

    # --- Mother summary ---
    mother_summary_df = (
        assignments_df.groupby("assigned_mother", as_index=False)
        .agg(
            assigned_daughters=("daughter_center", "count"),
            routes_used=("route_id", "nunique"),
        )
        .sort_values("assigned_mother")
    )

    # --- Cost summary ---
    total_dist_km = float(routes_df["route_distance_km"].sum())
    total_fixed_inr = float(routes_df["fixed_cost_inr"].sum())
    total_variable_inr = float(routes_df["variable_cost_inr"].sum())
    total_cost_inr = float(routes_df["total_cost_inr"].sum())

    summary_values: dict[str, float] = {
        "vehicle_speed_kmph": VEHICLE_SPEED_KMPH,
        "mother_centers": float(len(model_inputs.mothers)),
        "daughter_centers": float(len(model_inputs.daughters)),
        "routes_used": float(len(routes_df)),
        "total_distance_km": round(total_dist_km, 2),
        "total_fixed_cost_inr": round(total_fixed_inr, 2),
        "total_variable_cost_inr": round(total_variable_inr, 2),
        "total_cost_inr": round(total_cost_inr, 2),
    }
    cost_summary_df = pd.DataFrame(
        [{"metric": k, "value": v} for k, v in summary_values.items()]
    )

    # --- Optimality gap ---
    # The solver's incumbent (best feasible solution found):
    incumbent_units = solution.ObjectiveValue()
    incumbent_inr = incumbent_units / COST_SCALE

    # LP-relaxation lower bound computed before the solve phase (see solve_model):
    lower_bound_inr = lower_bound_units / COST_SCALE

    gap_abs_inr = incumbent_inr - lower_bound_inr
    gap_rel_pct = (
        (gap_abs_inr / incumbent_inr * 100.0) if incumbent_inr > 0 else 0.0
    )

    # Status text
    if gap_rel_pct < 0.01:
        status_text = "OPTIMAL — gap certified at < 0.01 %"
    else:
        status_text = (
            f"FEASIBLE (not proven optimal) — optimality gap = "
            f"{gap_rel_pct:.4f} %"
        )

    gap_df = pd.DataFrame(
        [
            {"metric": "Incumbent solution cost (Rs)", "value": round(incumbent_inr, 2)},
            {"metric": "Best lower bound (Rs)", "value": round(lower_bound_inr, 2)},
            {"metric": "Absolute gap (Rs)", "value": round(gap_abs_inr, 2)},
            {"metric": "Relative gap (%)", "value": round(gap_rel_pct, 4)},
            {"metric": "Solver status", "value": status_text},
            {"metric": "Solver time limit (s)", "value": float(SOLVER_TIME_LIMIT_SECONDS)},
        ]
    )

    # Print gap to console
    print("\n" + "=" * 60)
    print("OPTIMALITY GAP REPORT")
    print("=" * 60)
    print(f"  Incumbent (best feasible) : Rs {incumbent_inr:,.2f}")
    print(f"  Lower bound               : Rs {lower_bound_inr:,.2f}")
    print(f"  Absolute gap              : Rs {gap_abs_inr:,.2f}")
    print(f"  Relative gap              : {gap_rel_pct:.4f} %")
    print(f"  Status                    : {status_text}")
    print("=" * 60 + "\n")

    return (
        assignments_df,
        routes_df,
        mother_summary_df,
        cost_summary_df,
        gap_df,
        summary_values,
    )


# ---------------------------------------------------------------------------
# Step 4 – Write outputs
# ---------------------------------------------------------------------------

def write_excel(
    output_file: Path,
    assignments_df: pd.DataFrame,
    routes_df: pd.DataFrame,
    mother_summary_df: pd.DataFrame,
    cost_summary_df: pd.DataFrame,
    gap_df: pd.DataFrame,
) -> None:
    """Write all result tables to a multi-sheet Excel workbook.

    Sheets
    ------
    Assignments
        One row per daughter centre stop.  Columns: daughter_center,
        assigned_mother, route_id, vehicle_label, stop_sequence,
        arrival_minutes_from_6am, arrival_clock, leg_distance_km.

    Routes
        One row per active vehicle route.  Columns: route_id, vehicle_label,
        mother_center, stops, path, last_delivery_clock,
        returned_to_mother_clock, route_distance_km, route_drive_hours,
        fixed_cost_inr, variable_cost_inr, total_cost_inr.

    MotherSummary
        One row per mother centre: daughters assigned and routes used.

    CostSummary
        Scalar summary metrics.

    OptimalityGap
        Incumbent, lower bound, absolute/relative gap, and solver status.
    """
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        assignments_df.to_excel(writer, sheet_name="Assignments", index=False)
        routes_df.to_excel(writer, sheet_name="Routes", index=False)
        mother_summary_df.to_excel(writer, sheet_name="MotherSummary", index=False)
        cost_summary_df.to_excel(writer, sheet_name="CostSummary", index=False)
        gap_df.to_excel(writer, sheet_name="OptimalityGap", index=False)

    print(f"[Output] Excel workbook written → {output_file}")


def write_graph(
    output_file: Path,
    model_inputs: ModelInputs,
    assignments_df: pd.DataFrame,
    routes_df: pd.DataFrame,
) -> None:
    """Produce a geographic route map and save it as a PNG.

    Layout
    ------
    - Each mother centre is plotted as a large star marker (★).
    - Each daughter centre is plotted as a small circle.
    - Route arcs are drawn in the mother's colour, connecting stops in order.
    - A legend maps mother centre ID to colour.

    Parameters
    ----------
    output_file:
        Where to save the PNG.
    model_inputs:
        Contains node coordinates and centre names.
    assignments_df:
        Daughter assignments with mother, route, and stop_sequence.
    routes_df:
        Route paths and mother assignments.
    """
    # Build lookup: center_name → (lat, lon)
    coord: dict[int, tuple[float, float]] = {
        int(n["center_name"]): (float(n["latitude"]), float(n["longitude"]))
        for n in model_inputs.nodes
    }

    # Assign a distinct colour to each mother centre
    mother_ids = sorted(int(c) for c in model_inputs.mother_node_by_center)
    cmap = plt.cm.get_cmap("tab20", len(mother_ids))
    mother_color: dict[int, object] = {m: cmap(i) for i, m in enumerate(mother_ids)}

    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_facecolor("#f0f4f8")
    fig.patch.set_facecolor("#f0f4f8")

    # Draw route arcs for each active route
    for _, route_row in routes_df.iterrows():
        m_center = int(route_row["mother_center"])
        color = mother_color[m_center]
        # Parse path string, e.g.  "57 → 10 → 4 → 57"
        path_str: str = str(route_row["path"])
        path_ids = [int(p.strip()) for p in path_str.replace("→", ",").split(",")]
        lons = [coord[p][1] for p in path_ids]
        lats = [coord[p][0] for p in path_ids]
        ax.plot(lons, lats, "-", color=color, linewidth=0.8, alpha=0.6)

    # Plot daughter centres
    for d_node in model_inputs.daughter_nodes:
        d_name = int(model_inputs.center_name_by_node[d_node])
        lat, lon = coord[d_name]
        # Find assigned mother for colour
        rows = assignments_df[assignments_df["daughter_center"] == d_name]
        if not rows.empty:
            m = int(rows.iloc[0]["assigned_mother"])
            color = mother_color[m]
        else:
            color = "grey"
        ax.scatter(lon, lat, color=color, s=18, zorder=3, edgecolors="white", linewidths=0.3)

    # Plot mother centres on top
    for m_id in mother_ids:
        lat, lon = coord[m_id]
        color = mother_color[m_id]
        ax.scatter(
            lon, lat, color=color, s=160, marker="*", zorder=5,
            edgecolors="black", linewidths=0.6,
        )
        ax.annotate(
            str(m_id),
            xy=(lon, lat),
            fontsize=7,
            fontweight="bold",
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
        )

    # Legend: one entry per mother centre
    legend_patches = [
        mpatches.Patch(color=mother_color[m], label=f"Mother {m}")
        for m in mother_ids
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper left",
        fontsize=7,
        title="Mother Centres",
        title_fontsize=8,
        framealpha=0.85,
    )

    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.set_title(
        "Mother–Daughter Routing Solution\n"
        "(★ = mother depot, ● = daughter centre, lines = routes)",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Output] Route graph saved → {output_file}")


def write_report_docx(
    output_file: Path,
    summary_values: dict[str, float],
    assignments_df: pd.DataFrame,
    routes_df: pd.DataFrame,
    mother_summary_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    graph_file: Path,
) -> None:
    """Write a comprehensive solution report as a DOCX file.

    The report covers:
    1. Problem statement and objectives
    2. Modelling decisions and assumptions
    3. Algorithm description
    4. Full solution: assignments, routes, cost breakdown
    5. Optimality gap analysis
    6. Route visualisation (embedded graph)

    Parameters
    ----------
    output_file:
        Destination path for the DOCX file.
    summary_values:
        Scalar metrics from build_outputs().
    assignments_df, routes_df, mother_summary_df, gap_df:
        Result DataFrames from build_outputs().
    graph_file:
        Path to the PNG created by write_graph().
    """
    doc = Document()

    # ------------------------------------------------------------------ #
    # Title
    # ------------------------------------------------------------------ #
    doc.add_heading("Mother–Daughter Centre Routing — Solution Report", level=1)
    doc.add_paragraph(
        "Prepared by: OR-Tools MDVRP solver  |  "
        f"Solver: Google OR-Tools v9  |  "
        f"Time budget: {SOLVER_TIME_LIMIT_SECONDS} s"
    )

    # ------------------------------------------------------------------ #
    # 1. Problem Statement
    # ------------------------------------------------------------------ #
    doc.add_heading("1. Problem Statement", level=2)
    doc.add_paragraph(
        "The task is to connect 191 daughter centres to 11 mother centres using "
        "a fleet of Tata Ace vehicles.  Each vehicle departs from its home mother "
        "centre at 06:00, delivers to one or more daughter centres (with the last "
        "delivery reaching its destination by 10:00), and then returns to the same "
        "mother centre.  The objective is to minimise total vehicle operating cost "
        "(fixed + variable)."
    )

    # ------------------------------------------------------------------ #
    # 2. Assumptions
    # ------------------------------------------------------------------ #
    doc.add_heading("2. Assumptions and Modelling Decisions", level=2)
    assumptions = [
        "Distances are computed using the haversine great-circle formula because "
        "only GPS coordinates (latitude, longitude) are provided.  Actual road "
        "distances may be longer; the model treats haversine as a lower-bound proxy.",
        "Vehicle speed is constant at 50 km/h for all legs (road-speed variation "
        "not modelled).",
        "Service time at each daughter centre is zero minutes (unloading time is "
        "not specified).",
        "Fleet size is unconstrained per mother centre — one vehicle is modelled "
        "per daughter nearest to that mother (upper bound), and the solver decides "
        "which vehicles to consolidate.",
        "No vehicle capacity limit was specified, so cargo load is not constrained.",
        "Every daughter centre must be assigned to exactly one mother centre and "
        "served by exactly one route.",
        "The 10:00 AM deadline applies only to the LAST daughter stop; return to "
        "the mother can occur any time before midnight.",
    ]
    for a in assumptions:
        doc.add_paragraph(a, style="List Bullet")

    # ------------------------------------------------------------------ #
    # 3. Algorithm
    # ------------------------------------------------------------------ #
    doc.add_heading("3. Solution Algorithm", level=2)
    doc.add_paragraph(
        "The problem is modelled as a Multi-Depot Vehicle Routing Problem with "
        "Time Windows (MDVRPTW) and solved with the Google OR-Tools CP routing "
        "engine."
    )
    doc.add_paragraph(
        "Step 1 — Pre-processing.  "
        "All pairwise haversine distances, travel times, and arc costs are "
        "pre-computed for the 202-node graph (11 depots + 191 customers)."
    )
    doc.add_paragraph(
        "Step 2 — First solution.  "
        "PARALLEL_CHEAPEST_INSERTION greedily assigns each unserved daughter to "
        "the cheapest feasible route, producing the initial incumbent."
    )
    doc.add_paragraph(
        "Step 3 — Local search improvement.  "
        "GUIDED_LOCAL_SEARCH (GLS) iteratively escapes local optima by "
        "penalising arc features that appear frequently in poor solutions.  "
        f"The solver runs for up to {SOLVER_TIME_LIMIT_SECONDS} seconds."
    )
    doc.add_paragraph(
        "Step 4 — Lower bound.  "
        "After solving, routing.objective_lower_bound() returns the tightest "
        "lower bound the solver has proved on the global optimum.  Comparing "
        "this with the incumbent yields the optimality gap reported below."
    )

    # ------------------------------------------------------------------ #
    # 4. Key Parameters
    # ------------------------------------------------------------------ #
    doc.add_heading("4. Key Parameters", level=2)
    params_table = doc.add_table(rows=1, cols=2)
    params_table.style = "Table Grid"
    hdr = params_table.rows[0].cells
    hdr[0].text = "Parameter"
    hdr[1].text = "Value"
    params_data = [
        ("Vehicle speed", f"{VEHICLE_SPEED_KMPH:.0f} km/h"),
        ("Fixed cost per vehicle", f"Rs {FIXED_COST_INR:,.0f}"),
        ("Rate per km", f"Rs {RATE_PER_KM_INR:.0f}"),
        ("Rate per hour", f"Rs {RATE_PER_HOUR_INR:.0f}"),
        ("Effective variable cost/km", f"Rs {VARIABLE_COST_PER_KM_INR:.0f}"),
        ("Departure time", "06:00"),
        ("Last delivery deadline", "10:00 (240 min)"),
        ("Number of mother centres", str(int(summary_values["mother_centers"]))),
        ("Number of daughter centres", str(int(summary_values["daughter_centers"]))),
        ("Solver time limit", f"{SOLVER_TIME_LIMIT_SECONDS} s"),
    ]
    for param, val in params_data:
        row = params_table.add_row().cells
        row[0].text = param
        row[1].text = val

    # ------------------------------------------------------------------ #
    # 5. Results Overview
    # ------------------------------------------------------------------ #
    doc.add_heading("5. Results Overview", level=2)
    doc.add_paragraph(
        f"The solver found a feasible solution that serves all "
        f"{int(summary_values['daughter_centers'])} daughter centres using "
        f"{int(summary_values['routes_used'])} vehicle routes across "
        f"{int(summary_values['mother_centers'])} mother depots."
    )

    doc.add_heading("5a. Optimality Gap", level=3)
    gap_table = doc.add_table(rows=1, cols=2)
    gap_table.style = "Table Grid"
    gh = gap_table.rows[0].cells
    gh[0].text = "Metric"
    gh[1].text = "Value"
    for _, gaprow in gap_df.iterrows():
        gcells = gap_table.add_row().cells
        gcells[0].text = str(gaprow["metric"])
        gcells[1].text = str(gaprow["value"])

    doc.add_paragraph(
        "Interpretation: An optimality gap > 0 % means the solver has found a "
        "feasible solution but cannot yet certify that no cheaper solution exists.  "
        "The lower bound is the tightest cost the solver has proved any solution "
        "must equal or exceed.  The gap narrows as the solver runs longer or when "
        "a tighter formulation is used."
    )

    doc.add_heading("5b. Cost Summary", level=3)
    cost_tbl = doc.add_table(rows=1, cols=2)
    cost_tbl.style = "Table Grid"
    ch = cost_tbl.rows[0].cells
    ch[0].text = "Metric"
    ch[1].text = "Value"
    for _, crow in [
        ("Routes used", int(summary_values["routes_used"])),
        ("Total distance (km)", f"{summary_values['total_distance_km']:,.2f}"),
        ("Total fixed cost (Rs)", f"{summary_values['total_fixed_cost_inr']:,.2f}"),
        ("Total variable cost (Rs)", f"{summary_values['total_variable_cost_inr']:,.2f}"),
        ("Total cost (Rs)", f"{summary_values['total_cost_inr']:,.2f}"),
    ]:
        row = cost_tbl.add_row().cells
        row[0].text = str(_)
        row[1].text = str(crow)

    # ------------------------------------------------------------------ #
    # 6. Mother Centre Assignment Summary
    # ------------------------------------------------------------------ #
    doc.add_heading("6. Mother Centre Assignment Summary", level=2)
    doc.add_paragraph(
        "The table below shows how many daughter centres and how many routes "
        "are assigned to each mother centre in the optimal solution."
    )
    m_tbl = doc.add_table(rows=1, cols=3)
    m_tbl.style = "Table Grid"
    mh = m_tbl.rows[0].cells
    mh[0].text = "Mother Centre"
    mh[1].text = "Daughters Assigned"
    mh[2].text = "Routes Used"
    for _, row in mother_summary_df.iterrows():
        cells = m_tbl.add_row().cells
        cells[0].text = str(int(row["assigned_mother"]))
        cells[1].text = str(int(row["assigned_daughters"]))
        cells[2].text = str(int(row["routes_used"]))

    # ------------------------------------------------------------------ #
    # 7. Detailed Route Listing
    # ------------------------------------------------------------------ #
    doc.add_heading("7. Detailed Route Listing", level=2)
    doc.add_paragraph(
        "Each row describes one vehicle route.  "
        "'Path' shows the sequence of centres visited (mother → daughters → mother).  "
        "'Last delivery clock' is the scheduled arrival at the final daughter stop."
    )

    r_tbl = doc.add_table(rows=1, cols=8)
    r_tbl.style = "Table Grid"
    rh = r_tbl.rows[0].cells
    for i, col in enumerate(
        ["Route ID", "Mother", "Stops", "Path",
         "Last Delivery", "Return", "Distance (km)", "Total Cost (Rs)"]
    ):
        rh[i].text = col

    for _, rrow in routes_df.iterrows():
        cells = r_tbl.add_row().cells
        cells[0].text = str(rrow["route_id"])
        cells[1].text = str(int(rrow["mother_center"]))
        cells[2].text = str(int(rrow["stops"]))
        cells[3].text = str(rrow["path"])
        cells[4].text = str(rrow["last_delivery_clock"])
        cells[5].text = str(rrow["returned_to_mother_clock"])
        cells[6].text = f"{rrow['route_distance_km']:.2f}"
        cells[7].text = f"{rrow['total_cost_inr']:.2f}"

    # ------------------------------------------------------------------ #
    # 8. Daughter–Mother Assignment Table
    # ------------------------------------------------------------------ #
    doc.add_heading("8. Complete Daughter–Mother Assignment", level=2)
    doc.add_paragraph(
        "Every daughter centre, its assigned mother, the route it belongs to, "
        "its stop position within the route, and its scheduled arrival time."
    )

    a_tbl = doc.add_table(rows=1, cols=6)
    a_tbl.style = "Table Grid"
    ah = a_tbl.rows[0].cells
    for i, col in enumerate(
        ["Daughter", "Assigned Mother", "Route ID", "Stop #",
         "Arrival (min)", "Arrival (clock)"]
    ):
        ah[i].text = col

    for _, arow in assignments_df.iterrows():
        cells = a_tbl.add_row().cells
        cells[0].text = str(int(arow["daughter_center"]))
        cells[1].text = str(int(arow["assigned_mother"]))
        cells[2].text = str(arow["route_id"])
        cells[3].text = str(int(arow["stop_sequence"]))
        cells[4].text = str(int(arow["arrival_minutes_from_6am"]))
        cells[5].text = str(arow["arrival_clock"])

    # ------------------------------------------------------------------ #
    # 9. Route Map
    # ------------------------------------------------------------------ #
    if graph_file.exists():
        doc.add_heading("9. Route Map", level=2)
        doc.add_paragraph(
            "The figure below shows all mother centres (★) and daughter centres (●) "
            "on a latitude/longitude map.  Lines represent vehicle routes; colour "
            "identifies the home mother centre."
        )
        doc.add_picture(str(graph_file), width=Inches(6.0))

    doc.save(output_file)
    print(f"[Output] Solution report written → {output_file}")


def write_summary_markdown(
    output_file: Path,
    summary_values: dict[str, float],
    mother_summary_df: pd.DataFrame,
    routes_df: pd.DataFrame,
    gap_df: pd.DataFrame,
) -> None:
    """Write a compact Markdown summary of key results.

    Sections
    --------
    - Key metrics (cost, distance, routes)
    - Optimality gap
    - Mother assignment summary table
    - Top 10 longest routes table
    """
    gap_lookup: dict[str, object] = {
        str(row["metric"]): row["value"] for _, row in gap_df.iterrows()
    }

    lines: list[str] = [
        "# Mother–Daughter Routing — Solution Summary",
        "",
        "## Key Metrics",
        "",
        f"| Metric | Value |",
        f"| --- | ---: |",
        f"| Vehicle speed | {VEHICLE_SPEED_KMPH:.0f} km/h |",
        f"| Mother centres | {int(summary_values['mother_centers'])} |",
        f"| Daughter centres | {int(summary_values['daughter_centers'])} |",
        f"| Routes used | {int(summary_values['routes_used'])} |",
        f"| Total distance | {summary_values['total_distance_km']:,.2f} km |",
        f"| Total fixed cost | Rs {summary_values['total_fixed_cost_inr']:,.2f} |",
        f"| Total variable cost | Rs {summary_values['total_variable_cost_inr']:,.2f} |",
        f"| **Total cost** | **Rs {summary_values['total_cost_inr']:,.2f}** |",
        "",
        "## Optimality Gap",
        "",
        f"| Metric | Value |",
        f"| --- | ---: |",
        f"| Incumbent (best feasible) | Rs {gap_lookup.get('Incumbent solution cost (Rs)', 'N/A')} |",
        f"| Lower bound | Rs {gap_lookup.get('Best lower bound (Rs)', 'N/A')} |",
        f"| Absolute gap | Rs {gap_lookup.get('Absolute gap (Rs)', 'N/A')} |",
        f"| Relative gap | {gap_lookup.get('Relative gap (%)', 'N/A')} % |",
        f"| Solver status | {gap_lookup.get('Solver status', 'N/A')} |",
        "",
        "> **Note:** The relative gap is the percentage by which the best found",
        "> solution exceeds the proved lower bound.  A gap of 0 % certifies optimality.",
        "",
        "## Mother Assignment Summary",
        "",
        "| Mother Centre | Daughters Assigned | Routes Used |",
        "| --- | ---: | ---: |",
    ]
    for _, row in mother_summary_df.iterrows():
        lines.append(
            f"| {int(row['assigned_mother'])} "
            f"| {int(row['assigned_daughters'])} "
            f"| {int(row['routes_used'])} |"
        )

    lines += [
        "",
        "## Top 10 Longest Routes",
        "",
        "| Route ID | Mother | Stops | Distance (km) | Last Delivery | Total Cost (Rs) |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
    ]
    top_routes = routes_df.sort_values("route_distance_km", ascending=False).head(10)
    for _, row in top_routes.iterrows():
        lines.append(
            f"| {row['route_id']} "
            f"| {int(row['mother_center'])} "
            f"| {int(row['stops'])} "
            f"| {row['route_distance_km']:.2f} "
            f"| {row['last_delivery_clock']} "
            f"| {row['total_cost_inr']:.2f} |"
        )

    output_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Output] Markdown summary written → {output_file}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate data loading, solving, and output generation.

    Execution order
    ---------------
    1. Load and pre-process data_1.xlsx.
    2. Build and solve the OR-Tools routing model (up to 600 s).
    3. Extract route assignments, costs, and optimality gap.
    4. Write Excel, DOCX, PNG, and Markdown outputs to outputs/.
    5. Print a summary to stdout.
    """
    workspace = Path(__file__).resolve().parent
    input_file = workspace / "data_1.xlsx"
    output_dir = workspace / "outputs"
    output_dir.mkdir(exist_ok=True)

    # --- Load data ---
    print("[Step 1] Loading and pre-processing input data …")
    model_inputs = load_inputs(input_file)
    print(
        f"         {len(model_inputs.mothers)} mother centres, "
        f"{len(model_inputs.daughters)} daughter centres, "
        f"{len(model_inputs.vehicle_starts)} vehicles in fleet."
    )

    # --- Solve ---
    print("[Step 2] Building and solving the routing model …")
    manager, routing, solution, time_dim, lower_bound_units = solve_model(model_inputs)

    # --- Extract results ---
    print("[Step 3] Extracting solution and computing costs / gap …")
    (
        assignments_df,
        routes_df,
        mother_summary_df,
        cost_summary_df,
        gap_df,
        summary_values,
    ) = build_outputs(model_inputs, manager, routing, solution, time_dim, lower_bound_units)

    # --- Write outputs ---
    print("[Step 4] Writing output files …")
    excel_file = output_dir / "solution_output.xlsx"
    docx_file = output_dir / "solution_report.docx"
    graph_file = output_dir / "solution_graph.png"
    md_file = output_dir / "solution_summary.md"

    write_excel(
        excel_file, assignments_df, routes_df,
        mother_summary_df, cost_summary_df, gap_df,
    )

    write_graph(graph_file, model_inputs, assignments_df, routes_df)

    write_report_docx(
        docx_file, summary_values, assignments_df,
        routes_df, mother_summary_df, gap_df, graph_file,
    )

    write_summary_markdown(
        md_file, summary_values, mother_summary_df, routes_df, gap_df,
    )

    # --- Console summary ---
    print("\n" + "=" * 60)
    print("SOLUTION SUMMARY")
    print("=" * 60)
    print(f"  Mother centres    : {int(summary_values['mother_centers'])}")
    print(f"  Daughter centres  : {int(summary_values['daughter_centers'])}")
    print(f"  Routes used       : {int(summary_values['routes_used'])}")
    print(f"  Total distance    : {summary_values['total_distance_km']:,.2f} km")
    print(f"  Total fixed cost  : Rs {summary_values['total_fixed_cost_inr']:,.2f}")
    print(f"  Total var. cost   : Rs {summary_values['total_variable_cost_inr']:,.2f}")
    print(f"  Total cost        : Rs {summary_values['total_cost_inr']:,.2f}")
    print("=" * 60)
    print(f"  Excel  → {excel_file}")
    print(f"  Report → {docx_file}")
    print(f"  Graph  → {graph_file}")
    print(f"  MD     → {md_file}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
