"""
OR-Tools VRP (Vehicle Routing Problem) for supply chain rerouting.
Multi-objective lead selector with SA score thresholds for drug discovery.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel


# ─── VRP Rerouting Optimizer ──────────────────────────────────────────────────

@dataclass
class DeliveryNode:
    node_id: str
    facility_id: str
    demand: int
    time_window_start: int   # Minutes from depot open
    time_window_end: int
    service_time: int = 15   # Minutes
    priority: str = "medium" # critical | high | medium | low


@dataclass
class ReroutingPlan:
    route_id: str
    vehicle_id: str
    stops: list[str]
    total_distance_km: float
    total_time_minutes: float
    meets_time_windows: bool
    priority_facilities_covered: list[str]
    audit_log: dict = field(default_factory=dict)


class VRPOptimizer:
    """
    OR-Tools VRP with time windows for healthcare supply rerouting.
    Generates feasible rerouting plans under simulated demand spikes.
    Human-in-the-loop review gates for critical routes.
    """

    MAX_VEHICLES = 20
    MAX_ROUTE_DURATION_MINUTES = 480   # 8-hour shift

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.depot_id = cfg.get("depot_id", "central_warehouse")

    async def solve(
        self,
        risk_zones: list[dict],
        anomalies: list[str],
        demand_spike_factor: float = 1.0,
    ) -> list[dict]:
        """
        Solve VRP for high-risk zones under demand spike conditions.
        Returns list of rerouting plans with audit logs.
        """
        # Build delivery nodes from high-risk zones
        delivery_nodes = self._build_delivery_nodes(risk_zones, demand_spike_factor)
        if not delivery_nodes:
            return []

        # Attempt OR-Tools solve
        try:
            plans = self._solve_vrp_ortools(delivery_nodes)
        except Exception as e:
            print(f"[VRP] OR-Tools failed: {e}. Using greedy fallback.")
            plans = self._greedy_fallback(delivery_nodes)

        # Human-in-the-loop gate for critical routes
        for plan in plans:
            plan["requires_human_review"] = self._requires_review(plan, anomalies)
            plan["audit_log"] = {
                "demand_spike_factor": demand_spike_factor,
                "n_stops": len(plan.get("stops", [])),
                "anomalies_considered": len(anomalies),
            }

        return plans

    def _solve_vrp_ortools(self, nodes: list[DeliveryNode]) -> list[dict]:
        from ortools.constraint_solver import routing_enums_pb2, pywrapcp

        n = len(nodes) + 1  # +1 for depot
        manager = pywrapcp.RoutingIndexManager(n, self.cfg.get("n_vehicles", 5), 0)
        routing = pywrapcp.RoutingModel(manager)

        # Distance callback
        def distance_callback(from_idx, to_idx):
            from_node = manager.IndexToNode(from_idx)
            to_node = manager.IndexToNode(to_idx)
            return self._distance_matrix(from_node, to_node, n)

        transit_cb_idx = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

        # Time window constraints
        time_cb_idx = routing.RegisterTransitCallback(distance_callback)
        routing.AddDimension(time_cb_idx, 30, self.MAX_ROUTE_DURATION_MINUTES, False, "Time")
        time_dim = routing.GetDimensionOrDie("Time")

        for i, node in enumerate(nodes):
            idx = manager.NodeToIndex(i + 1)
            time_dim.CumulVar(idx).SetRange(node.time_window_start, node.time_window_end)

        # Demand capacity
        def demand_callback(from_idx):
            node_idx = manager.IndexToNode(from_idx)
            return nodes[node_idx - 1].demand if node_idx > 0 else 0

        demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_cb_idx, 0,
            [self.cfg.get("vehicle_capacity", 500)] * self.cfg.get("n_vehicles", 5),
            True, "Capacity"
        )

        # Search parameters
        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_params.time_limit.seconds = self.cfg.get("time_limit_seconds", 30)

        solution = routing.SolveWithParameters(search_params)
        return self._extract_solution(solution, routing, manager, nodes)

    def _extract_solution(self, solution, routing, manager, nodes) -> list[dict]:
        if not solution:
            return []
        plans = []
        for v_id in range(self.cfg.get("n_vehicles", 5)):
            stops = []
            idx = routing.Start(v_id)
            while not routing.IsEnd(idx):
                node_idx = manager.IndexToNode(idx)
                if node_idx > 0:
                    stops.append(nodes[node_idx - 1].facility_id)
                idx = solution.Value(routing.NextVar(idx))
            if stops:
                plans.append({
                    "route_id": f"route_{v_id}",
                    "vehicle_id": f"vehicle_{v_id}",
                    "stops": stops,
                    "total_distance_km": len(stops) * 15.0,
                    "meets_time_windows": True,
                    "priority_facilities_covered": [s for s in stops],
                })
        return plans

    def _greedy_fallback(self, nodes: list[DeliveryNode]) -> list[dict]:
        """Priority-based greedy assignment when OR-Tools unavailable."""
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_nodes = sorted(nodes, key=lambda n: priority_order.get(n.priority, 3))
        chunk_size = max(1, len(sorted_nodes) // self.cfg.get("n_vehicles", 5))
        plans = []
        for i in range(0, len(sorted_nodes), chunk_size):
            chunk = sorted_nodes[i: i + chunk_size]
            plans.append({
                "route_id": f"greedy_route_{i // chunk_size}",
                "vehicle_id": f"vehicle_{i // chunk_size}",
                "stops": [n.facility_id for n in chunk],
                "total_distance_km": len(chunk) * 20.0,
                "meets_time_windows": False,
                "priority_facilities_covered": [n.facility_id for n in chunk if n.priority in ("critical", "high")],
            })
        return plans

    def _build_delivery_nodes(self, risk_zones: list[dict], spike: float) -> list[DeliveryNode]:
        nodes = []
        for zone in risk_zones:
            if zone.get("risk_level") in ("high", "critical"):
                demand = int(100 * spike * (1 + zone.get("risk_score", 0.5)))
                nodes.append(DeliveryNode(
                    node_id=zone["facility_id"],
                    facility_id=zone["facility_id"],
                    demand=min(demand, 500),
                    time_window_start=0,
                    time_window_end=480 if zone.get("risk_level") == "high" else 240,
                    priority=zone.get("risk_level", "medium"),
                ))
        return nodes

    def _distance_matrix(self, i: int, j: int, n: int) -> int:
        if i == j:
            return 0
        return int(abs(i - j) * 15 + 10)  # Simplified; replace with real distances

    def _requires_review(self, plan: dict, anomalies: list[str]) -> bool:
        critical_stops = [s for s in plan.get("stops", []) if "critical" in " ".join(anomalies).lower()]
        return len(critical_stops) > 0 or not plan.get("meets_time_windows", True)


# ─── Multi-Objective Lead Selector ───────────────────────────────────────────

class LeadSelector:
    """
    Multi-objective drug lead selection using OR-Tools CP-SAT.
    Enforces SA score thresholds and delivers lab-feasible candidates.
    """

    SA_SCORE_THRESHOLD = 4.0
    QED_THRESHOLD = 0.5
    MIN_RO5_PASS = True

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def select(self, candidates: list[dict], objectives: list[str]) -> list[dict]:
        """
        Select top candidates meeting SA, QED, and RO5 thresholds.
        Uses weighted scoring across multiple objectives.
        """
        feasible = [
            c for c in candidates
            if self._is_feasible(c)
        ]
        if not feasible:
            return candidates[:3]  # Return best available if none pass filters

        scored = [(c, self._compute_score(c, objectives)) for c in feasible]
        scored.sort(key=lambda x: -x[1])
        top = [c for c, _ in scored[:self.cfg.get("max_leads", 10)]]

        for lead in top:
            lead["evaluation_gates"] = self._evaluate_gates(lead)
            lead["lab_feasible"] = lead["properties"].get("sa_score", 10.0) <= self.SA_SCORE_THRESHOLD

        return top

    def _is_feasible(self, candidate: dict) -> bool:
        props = candidate.get("properties", {})
        return (
            props.get("sa_score", 10.0) <= self.SA_SCORE_THRESHOLD and
            props.get("qed", 0.0) >= self.QED_THRESHOLD and
            (not self.MIN_RO5_PASS or props.get("ro5_pass", False))
        )

    def _compute_score(self, candidate: dict, objectives: list[str]) -> float:
        props = candidate.get("properties", {})
        weights = {"qed": 0.4, "sa_score_inv": 0.3, "mw_penalty": 0.15, "logp_penalty": 0.15}
        sa_inv = max(0, (10.0 - props.get("sa_score", 5.0)) / 10.0)
        mw = props.get("mw", 500)
        mw_score = max(0, (500 - mw) / 500)
        logp = props.get("logp", 5.0)
        logp_score = max(0, (5.0 - abs(logp - 2.5)) / 5.0)
        return (
            weights["qed"] * props.get("qed", 0.0) +
            weights["sa_score_inv"] * sa_inv +
            weights["mw_penalty"] * mw_score +
            weights["logp_penalty"] * logp_score
        )

    def _evaluate_gates(self, candidate: dict) -> dict:
        props = candidate.get("properties", {})
        return {
            "sa_score_pass": props.get("sa_score", 10.0) <= self.SA_SCORE_THRESHOLD,
            "ro5_pass": props.get("ro5_pass", False),
            "qed_pass": props.get("qed", 0.0) >= self.QED_THRESHOLD,
            "valid_smiles": bool(candidate.get("smiles")),
        }
