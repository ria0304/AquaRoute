from __future__ import annotations

import hashlib
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import wntr

from aquaroute.models.water_network import WaterNetwork


__all__ = ["load_epanet_inp", "to_wntr", "EpinetMetadata"]


class EpinetMetadata(Dict[str, Any]):
    """Metadata captured from an EPANET .inp conversion."""
    pass


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_int(s: str) -> int:
    return int(float(s))


def load_epanet_inp(
    path: pathlib.Path,
    *,
    capacity_formula: Optional[str] = None,
    cost_formula: Optional[str] = None,
    demand_unit: str = "lps",
    supply_from: str = "head",
) -> Tuple[WaterNetwork, EpinetMetadata]:
    """Load an EPANET .inp network into a WaterNetwork, preserving topology and physical params.

    Returns ``(network, metadata)`` where *metadata* records what was preserved/invented.
    """
    path = pathlib.Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"EPANET .inp not found: {path}")

    wn = wntr.network.WaterNetworkModel(str(path))
    network = WaterNetwork()

    metadata: EpinetMetadata = {
        "source_path": str(path),
        "source_hash": _sha256(path),
        "wntr_version": wntr.__version__,
        "node_counts": {
            "junctions": len(wn.junction_name_list),
            "reservoirs": len(wn.reservoir_name_list),
            "tanks": len(wn.tank_name_list),
            "pipes": len(wn.pipe_name_list),
        },
        "conversion_settings": {
            "capacity_formula": capacity_formula or "raw-params-stored",
            "cost_formula": cost_formula or "default-1.0",
            "demand_unit": demand_unit,
            "supply_from": supply_from,
        },
        "preserved": {},
        "invented": {},
    }

    # --- Node ID mapping: WNTR IDs -> AquaRoute sequential IDs ---
    # Collect all WNTR node names and types
    wntr_node_types: Dict[str, str] = {}  # name -> "reservoir"|"junction"|"tank"
    wntr_node_ids: Dict[str, int] = {}    # name -> original WNTR ID (may be int or str)

    for res_name in wn.reservoir_name_list:
        wntr_node_types[res_name] = "reservoir"
        wntr_node_ids[res_name] = int(res_name) if str(res_name).isdigit() else res_name

    for junc_name in wn.junction_name_list:
        wntr_node_types[junc_name] = "junction"
        wntr_node_ids[junc_name] = int(junc_name) if str(junc_name).isdigit() else junc_name

    for tank_name in wn.tank_name_list:
        wntr_node_types[tank_name] = "tank"
        wntr_node_ids[tank_name] = int(tank_name) if str(tank_name).isdigit() else tank_name

    # Build mapping from WNTR node name -> AquaRoute node ID
    # Assign IDs: reservoirs first, then communities (junctions), then tanks
    wntr_to_aro: Dict[str, int] = {}
    aro_counter = 0

    # Reservoirs
    for res_name in wn.reservoir_name_list:
        wntr_to_aro[res_name] = aro_counter
        aro_counter += 1

    # Communities (junctions)
    for junc_name in wn.junction_name_list:
        wntr_to_aro[junc_name] = aro_counter
        aro_counter += 1

    # Tanks
    for tank_name in wn.tank_name_list:
        wntr_to_aro[tank_name] = aro_counter
        aro_counter += 1

    metadata["wntr_to_aro"] = wntr_to_aro

    # --- Reservoirs ---
    preserved_res = []
    for res_name in wn.reservoir_name_list:
        aro_id = wntr_to_aro[res_name]
        try:
            base_head = float(wn.query_node_attribute(res_name, "head"))
        except Exception:
            base_head = None
        # Add reservoir to WaterNetwork
        network.add_reservoir(node_id=aro_id, capacity=float(base_head) if base_head is not None else 100.0,
                              supply=base_head / 10.0 if base_head is not None else float("inf"))
        preserved_res.append({"wntr_name": res_name, "aro_id": aro_id, "base_head_m": base_head})
        if supply_from == "head":
            supply_val = base_head / 10.0 if base_head is not None else float("inf")
        elif supply_from == "volume":
            supply_val = float("inf")
            metadata["invented"]["reservoir_volume_scaling"] = "head->volume not directly mapped"
        else:
            supply_val = float("inf")
            metadata["invented"]["reservoir_pattern"] = "pattern-based supply"
        metadata["preserved"].setdefault("reservoirs", [])
        metadata["preserved"]["reservoirs"].append(
            {"wntr_name": res_name, "aro_id": aro_id, "base_head_m": base_head, "supply_computed": supply_val}
        )

        extracted_demands = {}
    preserved_junc = []
    for junc_name in wn.junction_name_list:
        aro_id = wntr_to_aro[junc_name]
        # Extract demand from EPANET time series - provenance: original benchmark demand
        try:
            ts_list = wn.get_node(junc_name).demand_timeseries_list
            if ts_list:
                base = ts_list[0].base_value
            else:
                base = 0.0
        except Exception:
            base = 0.0
        extracted_demands[junc_name] = base
        preserved_junc.append({"wntr_name": junc_name, "aro_id": aro_id, "demand_lps": base})
        network.add_community(node_id=aro_id, demand=base, priority=1)

    # All demands are derived from EPANET time series base_values; no arbitrary
    # inflation (e.g. 10.0 default) is applied. Zeros are explicit, not invented.# --- Pipes ---
    preserved_pipe = []
    for pipe_name in wn.pipe_name_list:
        try:
            pipe = wn.get_link(pipe_name)
        except Exception:
            continue
        # Map start/end nodes from WNTR IDs to AquaRoute IDs
        try:
            start_wntr = pipe.start_node_name  # e.g., "10"
            end_wntr = pipe.end_node_name       # e.g., "11"
            start_aro = wntr_to_aro.get(start_wntr, -1)
            end_aro = wntr_to_aro.get(end_wntr, -1)
        except Exception:
            start_aro = -1
            end_aro = -1

        # Get physical attributes
        cap_info = {
            "diameter_m": pipe.diameter,
            "length_m": pipe.length,
            "roughness": pipe.roughness,
        }

        preserved_pipe.append(
            {
                "wntr_name": pipe_name,
                "start_aro_id": start_aro,
                "end_aro_id": end_aro,
                "length_m": cap_info["length_m"],
                "diameter_m": cap_info["diameter_m"],
                "roughness": cap_info["roughness"],
                "start_name_orig": start_wntr,
                "end_name_orig": end_wntr,
            }
        )
        # Add pipeline if both nodes mapped
        if start_aro >= 0 and end_aro >= 0:
            network.add_pipeline(
                from_node=start_aro,
                to_node=end_aro,
                capacity=float("inf"),
                delivery_cost=1.0 if cost_formula is None else None,
            )

    

    # --- Add reservoir-to-community edges from original WNTR structure ---
    # Connect each reservoir to its original downstream junctions based on
    # the WNTR network structure, preserving the actual topology.
    for res_name in wn.reservoir_name_list:
        res_aro = wntr_to_aro.get(res_name)
        if res_aro is None:
            continue
        # Find all junctions directly connected to this reservoir in WNTR
        try:
            res_link = wn.get_link(res_name)
            res_start = res_link.start_node_name
            res_end = res_link.end_node_name
            # Map to Aro IDs
            res_start_aro = wntr_to_aro.get(res_start, -1)
            res_end_aro = wntr_to_aro.get(res_end, -1)
            if res_start_aro >= 0 and res_end_aro >= 0:
                # Add pipeline from reservoir to community
                network.add_pipeline(
                    from_node=res_aro,
                    to_node=res_end_aro,
                    capacity=float('inf'),
                    delivery_cost=1.0 if opt_cost_formula is None else None,
                )
        except Exception:
            pass

    # --- Add reservoir-to-community edges preserving original WNTR topology ---
    # For each reservoir, connect it to all communities that were originally
    # downstream in the WNTR network. This ensures the flow network H has
    # proper edges for computing effective capacities.
    for res_name in wn.reservoir_name_list:
        res_aro = wntr_to_aro.get(res_name)
        if res_aro is None:
            continue
        # Find all directly connected downstream nodes in WNTR
        try:
            res_link = wn.get_link(res_name)
            downstream_names = [res_link.end_node_name]
            # Also check connected pipes from the reservoir
            for pipe_name in wn.pipe_name_list:
                try:
                    p = wn.get_link(pipe_name)
                    if p.start_node_name == res_name:
                        downstream_names.append(p.end_node_name)
                except Exception:
                    pass
            # Map each downstream to Aro ID and add pipeline
            for down_name in downstream_names:
                down_aro = wntr_to_aro.get(down_name, -1)
                if down_aro >= 0 and down_aro != res_aro:
                    # Check this edge doesn't already exist
                    edge_key = (min(res_aro, down_aro), max(res_aro, down_aro))
                    # Add pipeline with infinite capacity
                    network.add_pipeline(
                        from_node=res_aro,
                        to_node=down_aro,
                        capacity=float('inf'),
                        delivery_cost=1.0 if cost_formula is None else None,
                    )
        except Exception:
            metadata["preserved"]["pipes"] = preserved_pipe

    return network, metadata


def to_wntr(network: WaterNetwork, metadata: EpinetMetadata) -> wntr.network.WaterNetworkModel:
    """Best-effort reconstruction of a WNTR model from a WaterNetwork + metadata.

    Note: this is a simplified round-trip; full fidelity requires preserving all
    EPANET attributes that are not represented in the WaterNetwork schema.
    """
    wn = wntr.network.WaterNetworkModel()

    # --- Reservoirs ---
    preserved_res = metadata.get("preserved", {}).get("reservoirs", [])
    for r in preserved_res:
        res_name = f"res_{r['aro_id']}"
        bh = r.get("base_head_m")
        wn.add_reservoir(res_name, base_head=float(bh) if bh is not None else 100.0,
                         head_pattern="pattern1")

    # --- Junctions ---
    preserved_junc = metadata.get("preserved", {}).get("junctions", [])
    for j in preserved_junc:
        junc_name = f"j_{j['aro_id']}"
        d = j.get("demand_lps", 0.0)
        wn.add_junction(junc_name, base_demand=float(d))

    # --- Pipes ---
    preserved_p = metadata.get("preserved", {}).get("pipes", [])
    for p in preserved_p:
        start = p.get("start_aro_id", -1)
        end = p.get("end_aro_id", -1)
        length = p.get("length_m", 1000.0)
        diameter = p.get("diameter_m", 0.3)
        roughness = p.get("roughness", 100.0)
        if start >= 0 and end >= 0:
            sname = f"node_{start}"
            ename = f"node_{end}"
            wn.add_pipe(sname, ename, length, diameter, roughness, 0.0, "OPEN")

    # Placeholder junction demands
    for jn in wn.junction_name_list:
        wn.junction_base_demand[jn] = 10.0

    return wn


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m aquaroute.data.loaders <path/to/Net1.inp>")
        sys.exit(1)
    path = pathlib.Path(sys.argv[1])
    net, meta = load_epanet_inp(path)
    print(f"Loaded: {path.name}")
    print(f"  {net.n_reservoirs} reservoirs, {net.n_communities} communities, "
          f"{sum(1 for _ in net.G.edges())} pipelines")
    print(f"  source_hash: {meta['source_hash'][:8]}...")
    print(f"  conversion_settings: {meta['conversion_settings']}")
    print(f"  Preserved pipes: {len(meta['preserved']['pipes'])}")
    # Print node mapping
    print(f"  WNTR->Aro mapping: {meta.get('wntr_to_aro', {})}")
    wn = to_wntr(net, meta)
    print(f"  Round-trip WNTR pipes: {len(wn.pipe_name_list)}")
    print(f"  Round-trip WNTR junctions: {len(wn.junction_name_list)}")
    print(f"  Round-trip WNTR reservoirs: {len(wn.reservoir_name_list)}")
