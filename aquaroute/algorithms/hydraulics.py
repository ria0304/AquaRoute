"""Hydraulic validation against the original EPANET network.

The optimisation layer works on a graph abstraction.  This module closes the
loop by feeding allocation-derived demands back into the *original* EPANET
model (real topology, pipe diameters, lengths, roughness, elevations, pumps,
valves) and running the EPANET hydraulic engine through WNTR.  Pressure and
flow results are then compared against a full-demand baseline simulation of
the same network.

Two things are therefore validated:

1. *Deliverability* -- can the requested allocations actually be carried by the
   physical network without violating pressure/velocity constraints?
2. *Fidelity* -- how far do the allocation-derived pressures and flows depart
   from the baseline (full-demand) hydraulics?

Scenario-specific disruptions are represented physically in the EPANET model
rather than merely by editing allocations: demand spikes scale junction
demands, pipeline failures close pipes, and supply reductions cap source
outflow (via flow-control valves on pipe outlets and reduced pump speed on
pump outlets).
"""

from __future__ import annotations

import copy
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np

from aquaroute.models.water_network import WaterNetwork

__all__ = ["HydraulicValidator", "validate_allocation"]

# Typical service constraints for distribution networks.
MIN_PRESSURE_M = 20.0      # ~ 15 psi residual pressure
MAX_PRESSURE_M = 100.0     # ~ 120 psi
MAX_VELOCITY_M_S = 2.0     # erosion / surge guidance
LARGE_SETTING = 1.0e6      # effectively uncapped flow-control valve


class HydraulicValidator:
    """Run EPANET hydraulics on the original network for a set of demands."""

    def __init__(self, network: WaterNetwork, metadata: Optional[dict] = None):
        import wntr  # local import so the module loads without wntr installed

        self._wntr = wntr
        self.network = network
        self.metadata = metadata or {}
        source = self.metadata.get("source_path")
        if not source or not pathlib.Path(source).is_file():
            raise FileNotFoundError(
                "Hydraulic validation requires metadata['source_path'] pointing "
                "at the original EPANET .inp file."
            )
        self.source_path = pathlib.Path(source)
        self.original_model = wntr.network.WaterNetworkModel(str(self.source_path))
        self.aro_to_wntr = {int(k): v for k, v in self.metadata.get("aro_to_wntr", {}).items()}
        # Original (full) junction base demands in m3/s.
        self.base_demands: Dict[str, float] = {}
        for j in self.original_model.junction_name_list:
            try:
                self.base_demands[j] = float(
                    self.original_model.get_node(j).demand_timeseries_list[0].base_value)
            except Exception:
                self.base_demands[j] = 0.0
        self._baseline: Optional[dict] = None

    # -- helpers -----------------------------------------------------------
    def _run(self, model):
        sim = self._wntr.sim.EpanetSimulator(model)
        results = sim.run_sim()
        return results.node, results.link

    def _summarise(self, node, link) -> dict:
        """Summarise steady-state (first period) hydraulics."""
        wn = self.original_model
        pressure: Dict[str, float] = {}
        for j in wn.junction_name_list:
            try:
                pressure[j] = float(node["pressure"][j].iloc[0])
            except Exception:
                pressure[j] = 0.0
        flow: Dict[str, float] = {}
        velocity: Dict[str, float] = {}
        for p in wn.pipe_name_list:
            try:
                flow[p] = float(link["flowrate"][p].iloc[0])
                velocity[p] = float(link["velocity"][p].iloc[0])
            except Exception:
                flow[p] = 0.0
                velocity[p] = 0.0
        pvals = np.array([v for v in pressure.values() if np.isfinite(v)])
        vvals = np.array([abs(v) for v in velocity.values() if np.isfinite(v)])
        return {
            "pressure": pressure,
            "flow": flow,
            "velocity": velocity,
            "min_pressure": float(pvals.min()) if pvals.size else 0.0,
            "max_pressure": float(pvals.max()) if pvals.size else 0.0,
            "max_velocity": float(vvals.max()) if vvals.size else 0.0,
            "nodes_below_min_pressure": int((pvals < MIN_PRESSURE_M).sum()) if pvals.size else 0,
            "links_above_max_velocity": int((vvals > MAX_VELOCITY_M_S).sum()) if vvals.size else 0,
            "n_nodes": int(pvals.size),
            "n_links": int(vvals.size),
        }

    def baseline(self) -> dict:
        """Full-demand (undisrupted) hydraulic baseline of the original network."""
        if self._baseline is None:
            node, link = self._run(self.original_model)
            self._baseline = self._summarise(node, link)
        return self._baseline

    def _replace_with_fcv(self, model, link_name: str, cap: float) -> None:
        """Replace a source-outlet pipe with a flow-control valve capping outflow."""
        link = model.get_link(link_name)
        start, end = link.start_node_name, link.end_node_name
        diameter = getattr(link, "diameter", 0.3048)
        try:
            model.remove_link(link_name)
            model.add_valve(
                link_name + "_fcv", start, end,
                diameter=float(diameter), valve_type="FCV",
                initial_setting=float(cap), initial_status="ACTIVE",
            )
        except Exception:
            pass  # leave the original link if replacement is not possible

    def _apply_scenario(self, model, scenario: dict) -> None:
        """Apply a disruption scenario *physically* to the EPANET model."""
        scenario_demands: Dict[str, float] = scenario.get("demands", {}) or {}
        failures: List[Tuple[int, int]] = scenario.get("failures", []) or []
        supply: Dict = scenario.get("supply") or {}

        # 1) Demand spike / scenario demands: set junction base demands directly.
        for j in self.original_model.junction_name_list:
            scen_demand = float(scenario_demands.get(j, self.base_demands.get(j, 0.0)))
            try:
                model.get_node(j).demand_timeseries_list[0].base_value = scen_demand
            except Exception:
                pass

        # 2) Pipeline failure: close the failed links in both orientations.
        for (u, v) in failures:
            for ln in list(model.link_name_list):
                l = model.get_link(ln)
                if {l.start_node_name, l.end_node_name} == {u, v}:
                    try:
                        l._user_status = self._wntr.network.LinkStatus.Closed
                    except Exception:
                        pass

        # 3) Supply reduction: cap source outflow physically.
        if supply:
            baseline_total = sum(self.base_demands.values())
            total_cap = sum(float(v) for v in supply.values() if isinstance(v, (int, float)))
            reduction_frac = (total_cap / baseline_total) if baseline_total > 0 else total_cap
            for src in (list(self.original_model.reservoir_name_list)
                        + list(self.original_model.tank_name_list)):
                cap = supply.get(src)
                if not isinstance(cap, (int, float)):
                    continue
                for ln in list(model.get_links_for_node(src)):
                    l = model.get_link(ln)
                    try:
                        if type(l).__name__ == "HeadPump":
                            # Pump outlet: reduce pump speed to cut source output.
                            l.speed = max(0.01, float(reduction_frac))
                        else:
                            # Pipe outlet: insert a flow-control valve capping flow.
                            self._replace_with_fcv(model, ln, float(cap))
                    except Exception:
                        pass

    # -- validation --------------------------------------------------------
    def validate(self, allocations: Dict[int, float],
                 scenario: Optional[dict] = None) -> dict:
        """Apply allocation-derived demands under a scenario and validate.

        ``allocations`` maps AquaRoute community ID -> delivered flow (m3/s).
        For each community the served fraction is ``alloc / scenario_demand``
        (capped at 1.0) and the junction base demand is set to the served
        volume before running EPANET.
        """
        if scenario is None:
            scenario = {"scenario_type": "normal", "demands": {},
                        "supply": {}, "failures": []}
        scenario_type = scenario.get("scenario_type", "normal")

        model = copy.deepcopy(self.original_model)
        self._apply_scenario(model, scenario)

        applied: Dict[int, dict] = {}
        for c in self.network.get_communities():
            jname = self.aro_to_wntr.get(c)
            if jname is None or jname not in model.junction_name_list:
                continue
            scen_demand = float(scenario.get("demands", {}).get(
                jname, self.base_demands.get(jname, 0.0)))
            alloc = max(0.0, float(allocations.get(c, 0.0)))
            if scen_demand > 0:
                fraction = min(1.0, alloc / scen_demand)
                served = scen_demand * fraction
            else:
                fraction = 1.0 if alloc == 0 else 0.0
                served = 0.0
            try:
                model.get_node(jname).demand_timeseries_list[0].base_value = served
            except Exception:
                served = 0.0
                fraction = 0.0
            applied[c] = {
                "junction": jname,
                "scenario_demand": scen_demand,
                "served_demand": served,
                "served_fraction": fraction,
            }

        node, link = self._run(model)
        summary = self._summarise(node, link)
        baseline = self.baseline()

        # Fidelity: departure from the full-demand baseline.
        p_deltas, f_deltas = [], []
        for j in self.original_model.junction_name_list:
            p_new = summary["pressure"].get(j, 0.0)
            p_base = baseline["pressure"].get(j, 0.0)
            if np.isfinite(p_new) and np.isfinite(p_base):
                p_deltas.append(abs(p_new - p_base))
        for p in self.original_model.pipe_name_list:
            f_new = summary["flow"].get(p, 0.0)
            f_base = baseline["flow"].get(p, 0.0)
            if np.isfinite(f_new) and np.isfinite(f_base):
                f_deltas.append(abs(f_new - f_base))

        # Absolute violations vs fixed thresholds.
        abs_violations: List[str] = []
        if summary["min_pressure"] < MIN_PRESSURE_M:
            abs_violations.append(
                f"min pressure {summary['min_pressure']:.2f}m < {MIN_PRESSURE_M}m")
        if summary["max_pressure"] > MAX_PRESSURE_M:
            abs_violations.append(
                f"max pressure {summary['max_pressure']:.2f}m > {MAX_PRESSURE_M}m")
        if summary["max_velocity"] > MAX_VELOCITY_M_S:
            abs_violations.append(
                f"max velocity {summary['max_velocity']:.2f}m/s > {MAX_VELOCITY_M_S}m/s")

        # New violations relative to the baseline (some benchmarks already sit
        # below the absolute threshold in their baseline state).
        base_p = baseline["pressure"]
        base_v = baseline["velocity"]
        new_low_pressure = [
            j for j, p in summary["pressure"].items()
            if np.isfinite(p) and p < MIN_PRESSURE_M
            and np.isfinite(base_p.get(j, np.nan)) and base_p.get(j) >= MIN_PRESSURE_M
        ]
        new_high_velocity = [
            p for p, v in summary["velocity"].items()
            if np.isfinite(v) and abs(v) > MAX_VELOCITY_M_S
            and abs(base_v.get(p, 0.0)) <= MAX_VELOCITY_M_S
        ]
        relative_feasible = not new_low_pressure and not new_high_velocity

        return {
            "scenario_type": scenario_type,
            "feasible": relative_feasible,
            "absolute_feasible": len(abs_violations) == 0,
            "violations": abs_violations,
            "new_pressure_violations": new_low_pressure,
            "new_velocity_violations": new_high_velocity,
            "applied_demands": applied,
            "hydraulics": summary,
            "baseline": {
                "min_pressure": baseline["min_pressure"],
                "max_pressure": baseline["max_pressure"],
                "max_velocity": baseline["max_velocity"],
                "nodes_below_min_pressure": baseline["nodes_below_min_pressure"],
            },
            "fidelity": {
                "mean_abs_pressure_delta_m": float(np.mean(p_deltas)) if p_deltas else 0.0,
                "max_abs_pressure_delta_m": float(np.max(p_deltas)) if p_deltas else 0.0,
                "mean_abs_flow_delta": float(np.mean(f_deltas)) if f_deltas else 0.0,
                "max_abs_flow_delta": float(np.max(f_deltas)) if f_deltas else 0.0,
            },
            "constraints": {
                "min_pressure_m": MIN_PRESSURE_M,
                "max_pressure_m": MAX_PRESSURE_M,
                "max_velocity_m_s": MAX_VELOCITY_M_S,
            },
        }


def validate_allocation(
    network: WaterNetwork,
    allocations: Dict[int, float],
    metadata: Optional[dict] = None,
    scenario: Optional[dict] = None,
) -> dict:
    """Convenience wrapper around :class:`HydraulicValidator`."""
    return HydraulicValidator(network, metadata).validate(allocations, scenario)
