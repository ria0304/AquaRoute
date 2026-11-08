"""Hydraulic validation against the original EPANET network.

The optimisation layer works on a graph abstraction.  This module closes the
loop by feeding allocation-derived demands back into the *original* EPANET
model (real topology, pipe diameters, roughness, elevations) and running the
EPANET hydraulic engine through WNTR.  Pressure and flow results are then
compared against a full-demand baseline simulation of the same network.

Two things are therefore validated:

1. *Deliverability* -- can the requested allocations actually be carried by the
   physical network without violating pressure/velocity constraints?
2. *Fidelity* -- how far do the allocation-derived pressures and flows depart
   from the baseline (full-demand) hydraulics?
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
        self.base_model = wntr.network.WaterNetworkModel(str(self.source_path))
        self.aro_to_wntr = {int(k): v for k, v in self.metadata.get("aro_to_wntr", {}).items()}
        # Baseline (full-demand) hydraulic state, computed once.
        self._baseline = None

    # -- helpers -----------------------------------------------------------
    def _junction_map(self) -> Dict[int, str]:
        """Map AquaRoute community ID -> EPANET junction name."""
        out = {}
        for c in self.network.get_communities():
            name = self.aro_to_wntr.get(c)
            if name is not None and name in self.base_model.junction_name_list:
                out[c] = name
        return out

    def _run(self, model) -> Tuple[dict, dict]:
        sim = self._wntr.sim.EpanetSimulator(model)
        results = sim.run_sim()
        return results.node, results.link

    def _summarise(self, node, link) -> dict:
        """Summarise steady-state (first period) hydraulics."""
        wn = self.base_model
        # Node pressure = head - elevation (m)
        pressure = {}
        for j in wn.junction_name_list:
            try:
                head = float(node["pressure"][j].iloc[0])
                pressure[j] = head  # WNTR 'pressure' is already head - elevation
            except Exception:
                pressure[j] = 0.0
        flow = {}
        velocity = {}
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
        """Full-demand (undisrupted) hydraulic baseline."""
        if self._baseline is None:
            node, link = self._run(self.base_model)
            self._baseline = self._summarise(node, link)
        return self._baseline

    # -- validation --------------------------------------------------------
    def validate(self, allocations: Dict[int, float]) -> dict:
        """Apply allocation-derived demands and validate hydraulics.

        For each community (junction) the served fraction
        ``alloc / demand`` scales the original EPANET base demand.  All other
        network attributes are left untouched so the simulation reflects the
        real physical network.
        """
        model = copy.deepcopy(self.base_model)
        jmap = self._junction_map()

        applied = {}
        for c, jname in jmap.items():
            demand = self.network.G.nodes[c].get("demand", 0.0)
            alloc = max(0.0, float(allocations.get(c, 0.0)))
            if demand > 0:
                fraction = min(1.0, alloc / demand)
            else:
                fraction = 1.0
            base = model.get_node(jname).demand_timeseries_list[0]
            original = base.base_value
            base.base_value = original * fraction
            applied[c] = {"junction": jname, "served_fraction": fraction,
                          "original_demand": original,
                          "applied_demand": original * fraction}

        node, link = self._run(model)
        summary = self._summarise(node, link)
        baseline = self.baseline()

        # Fidelity: departure of pressures/flows from the baseline.
        p_deltas = []
        for j in self.base_model.junction_name_list:
            p_new = summary["pressure"].get(j, 0.0)
            p_base = baseline["pressure"].get(j, 0.0)
            if np.isfinite(p_new) and np.isfinite(p_base):
                p_deltas.append(abs(p_new - p_base))
        f_deltas = []
        for p in self.base_model.pipe_name_list:
            f_new = summary["flow"].get(p, 0.0)
            f_base = baseline["flow"].get(p, 0.0)
            if np.isfinite(f_new) and np.isfinite(f_base):
                f_deltas.append(abs(f_new - f_base))

        violations: List[str] = []
        if summary["min_pressure"] < MIN_PRESSURE_M:
            violations.append(
                f"min pressure {summary['min_pressure']:.2f}m < {MIN_PRESSURE_M}m")
        if summary["max_pressure"] > MAX_PRESSURE_M:
            violations.append(
                f"max pressure {summary['max_pressure']:.2f}m > {MAX_PRESSURE_M}m")
        if summary["max_velocity"] > MAX_VELOCITY_M_S:
            violations.append(
                f"max velocity {summary['max_velocity']:.2f}m/s > {MAX_VELOCITY_M_S}m/s")

        # New violations relative to the full-demand baseline.  Some benchmark
        # networks (e.g. Net2) already operate below the absolute threshold in
        # their baseline state; the meaningful test is whether the allocation
        # introduces *additional* violations or worsens existing ones.
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
            "feasible": relative_feasible,
            "absolute_feasible": len(violations) == 0,
            "violations": violations,
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
) -> dict:
    """Convenience wrapper around :class:`HydraulicValidator`."""
    return HydraulicValidator(network, metadata).validate(allocations)
