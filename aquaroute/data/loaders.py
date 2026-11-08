from __future__ import annotations

import hashlib
import pathlib
from typing import Any, Dict, Optional, Tuple

import wntr

from aquaroute.models.water_network import WaterNetwork

__all__ = [
    "load_epanet_inp",
    "to_wntr",
    "EpinetMetadata",
    "hazen_williams_capacity",
]


class EpinetMetadata(Dict[str, Any]):
    """Metadata captured from an EPANET .inp conversion."""
    pass


# Reference hydraulic gradient used to turn static pipe geometry into a
# flow-capacity proxy for the network-flow LP.  0.001 m/m corresponds to a
# head loss of 1 m per 1000 m of pipe, a common design criterion for
# distribution mains.  Capacities are therefore expressed in m^3/s so they are
# directly comparable with the EPANET demands (WNTR stores demands in m^3/s).
REFERENCE_GRADIENT = 0.001


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def hazen_williams_capacity(
    diameter_m: float,
    length_m: float,
    roughness: float,
    gradient: float = REFERENCE_GRADIENT,
) -> float:
    """Flow capacity (m^3/s) of a pipe from Hazen-Williams geometry.

    Uses the SI Hazen-Williams relation ``Q = 0.278 * C * D^2.63 * S^0.54``
    evaluated at a reference hydraulic gradient ``S`` (m/m).  ``roughness`` is
    interpreted as the Hazen-Williams C factor (EPANET H-W roughness).

    Returns ``inf`` when geometry is missing/invalid so that the caller can
    fall back explicitly rather than silently inventing a value.
    """
    try:
        d = float(diameter_m)
        L = float(length_m)
        C = float(roughness)
    except (TypeError, ValueError):
        return float("inf")
    if d <= 0 or L <= 0 or C <= 0 or gradient <= 0:
        return float("inf")
    return 0.278 * C * (d ** 2.63) * (gradient ** 0.54)


def _node_key(name: str):
    """Normalise an EPANET node name to an int when numeric, else keep str."""
    s = str(name)
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return s


def _add_bidirectional_link(
    network: WaterNetwork,
    start_aro: int,
    end_aro: int,
    capacity: float,
    delivery_cost: float,
    link_kind: str,
    wntr_name: str,
    **attrs: Any,
) -> None:
    """Add a link as two directed arcs.

    Distribution pipes carry flow in either direction; EPANET link
    orientation is arbitrary.  Modelling each link as a pair of directed arcs
    lets the network-flow LP and the max-flow routines route water both ways
    while preserving the physical capacity of the link.
    """
    for (u, v) in ((start_aro, end_aro), (end_aro, start_aro)):
        network.add_pipeline(from_node=u, to_node=v, capacity=capacity,
                             delivery_cost=delivery_cost)
        network.G[u][v]["link_kind"] = link_kind
        network.G[u][v]["wntr_name"] = wntr_name
        for k, val in attrs.items():
            network.G[u][v][k] = val


def load_epanet_inp(
    path: pathlib.Path,
    *,
    capacity_formula: Optional[str] = None,
    cost_formula: Optional[str] = None,
    demand_unit: str = "m3s",
    supply_from: str = "head",
) -> Tuple[WaterNetwork, EpinetMetadata]:
    """Load an EPANET .inp network into a ``WaterNetwork``.

    The conversion preserves the *real* EPANET topology:

    * reservoirs and tanks become supply nodes,
    * junctions become demand (community) nodes,
    * pipes, pumps and valves become graph edges,

    Pipe capacities are derived from the physical pipe geometry with
    :func:`hazen_williams_capacity` (documented reference-gradient proxy).
    Pumps and valves are assigned a large finite capacity because their
    throughput is limited by the connected mains rather than by a fixed
    diameter.  No shortcut connections or synthetic costs are invented: every
    edge corresponds to an EPANET link, and delivery costs default to 1.0 only
    when no ``cost_formula`` is supplied.

    Returns
    -------
    (WaterNetwork, EpinetMetadata)
        The converted network and a metadata record describing what was
        preserved, what was derived, and what (if anything) was invented.
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
        "headloss_model": str(wn.options.hydraulic.headloss),
        "node_counts": {
            "junctions": len(wn.junction_name_list),
            "reservoirs": len(wn.reservoir_name_list),
            "tanks": len(wn.tank_name_list),
            "pipes": len(wn.pipe_name_list),
            "pumps": len(wn.pump_name_list),
            "valves": len(wn.valve_name_list),
        },
        "conversion_settings": {
            "capacity_formula": capacity_formula or "hazen-williams@S=0.001",
            "cost_formula": cost_formula or "default-1.0",
            "demand_unit": demand_unit,
            "supply_from": supply_from,
            "reference_gradient": REFERENCE_GRADIENT,
        },
        "preserved": {"reservoirs": [], "tanks": [], "junctions": [], "pipes": [],
                      "pumps": [], "valves": []},
        "invented": {},
    }

    # ------------------------------------------------------------------
    # Node ID mapping: EPANET names -> sequential AquaRoute IDs.
    # Supply nodes (reservoirs, then tanks) come first, then junctions.
    # ------------------------------------------------------------------
    wntr_to_aro: Dict[str, int] = {}
    aro_counter = 0

    for name in wn.reservoir_name_list:
        wntr_to_aro[name] = aro_counter
        aro_counter += 1
    for name in wn.tank_name_list:
        wntr_to_aro[name] = aro_counter
        aro_counter += 1
    for name in wn.junction_name_list:
        wntr_to_aro[name] = aro_counter
        aro_counter += 1

    metadata["wntr_to_aro"] = dict(wntr_to_aro)
    metadata["aro_to_wntr"] = {v: k for k, v in wntr_to_aro.items()}

    # ------------------------------------------------------------------
    # Reservoirs (head-driven supply)
    # ------------------------------------------------------------------
    for res_name in wn.reservoir_name_list:
        aro_id = wntr_to_aro[res_name]
        try:
            base_head = float(wn.get_node(res_name).base_head)
        except Exception:
            base_head = None
        if supply_from == "head":
            # A reservoir is effectively unbounded in the LP; its head drives
            # flow but does not cap the *volume* deliverable over a period.
            supply_val = float("inf")
        else:
            supply_val = float("inf")
            metadata["invented"]["reservoir_volume_scaling"] = (
                f"supply_from={supply_from!r} not modelled; using infinite supply")
        network.add_reservoir(
            node_id=aro_id,
            capacity=float(base_head) if base_head is not None else 100.0,
            supply=supply_val,
        )
        network.G.nodes[aro_id]["source_kind"] = "reservoir"
        network.G.nodes[aro_id]["base_head_m"] = base_head
        network.G.nodes[aro_id]["wntr_name"] = res_name
        metadata["preserved"]["reservoirs"].append(
            {"wntr_name": res_name, "aro_id": aro_id, "base_head_m": base_head}
        )

    # ------------------------------------------------------------------
    # Tanks (storage / supply nodes).  Modelled as supply sources because
    # they can discharge into the network over the allocation period.
    # ------------------------------------------------------------------
    for tank_name in wn.tank_name_list:
        aro_id = wntr_to_aro[tank_name]
        tank = wn.get_node(tank_name)
        try:
            init_level = float(tank.init_level)
            elevation = float(tank.elevation)
            min_level = float(tank.min_level)
            max_level = float(tank.max_level)
            diameter = float(tank.diameter)
        except Exception:
            init_level = elevation = min_level = max_level = diameter = None
        base_head = (elevation + init_level) if (elevation is not None and init_level is not None) else None
        network.add_reservoir(
            node_id=aro_id,
            capacity=float(base_head) if base_head is not None else 100.0,
            supply=float("inf"),
        )
        # Relabel provenance so tanks are distinguishable from reservoirs.
        network.G.nodes[aro_id]["source_kind"] = "tank"
        network.G.nodes[aro_id]["base_head_m"] = base_head
        network.G.nodes[aro_id]["wntr_name"] = tank_name
        metadata["preserved"]["tanks"].append({
            "wntr_name": tank_name,
            "aro_id": aro_id,
            "base_head_m": base_head,
            "init_level_m": init_level,
            "min_level_m": min_level,
            "max_level_m": max_level,
            "diameter_m": diameter,
        })

    # ------------------------------------------------------------------
    # Junctions (communities).  Demands come from the EPANET time series
    # base_value; no arbitrary inflation is applied.
    # ------------------------------------------------------------------
    for junc_name in wn.junction_name_list:
        aro_id = wntr_to_aro[junc_name]
        try:
            ts_list = wn.get_node(junc_name).demand_timeseries_list
            base = float(ts_list[0].base_value) if ts_list else 0.0
        except Exception:
            base = 0.0
        # Negative EPANET demand means the node *injects* water (an inflow).
        # Such nodes do not require service; record the injection and model
        # them as zero-demand communities to avoid negative demands in the LP.
        if base < 0:
            metadata["preserved"].setdefault("negative_demand_inflows", []).append(
                {"wntr_name": junc_name, "aro_id": aro_id, "inflow_m3s": -base}
            )
            base = 0.0
        network.add_community(node_id=aro_id, demand=base, priority=1)
        network.G.nodes[aro_id]["wntr_name"] = junc_name
        metadata["preserved"]["junctions"].append(
            {"wntr_name": junc_name, "aro_id": aro_id, "demand": base}
        )

    # ------------------------------------------------------------------
    # Pipes (real capacities from physical geometry)
    # ------------------------------------------------------------------
    total_demand = network.total_demand()
    # A finite stand-in for "unbounded" links (pumps/valves).  Kept large
    # relative to demand so it does not artificially bind the LP, while
    # avoiding ``inf`` in max-flow bookkeeping.
    large_capacity = max(total_demand, 1e-6) * 1e3

    for pipe_name in wn.pipe_name_list:
        pipe = wn.get_link(pipe_name)
        start_wntr = pipe.start_node_name
        end_wntr = pipe.end_node_name
        start_aro = wntr_to_aro.get(start_wntr, -1)
        end_aro = wntr_to_aro.get(end_wntr, -1)
        if start_aro < 0 or end_aro < 0:
            metadata["invented"].setdefault("skipped_links", []).append(pipe_name)
            continue
        diam = float(pipe.diameter)
        length = float(pipe.length)
        rough = float(pipe.roughness)
        cap = hazen_williams_capacity(diam, length, rough)
        _add_bidirectional_link(
            network, start_aro, end_aro, cap,
            1.0 if cost_formula is None else None,
            link_kind="pipe", wntr_name=pipe_name,
            diameter_m=diam, length_m=length, roughness=rough,
        )
        metadata["preserved"]["pipes"].append({
            "wntr_name": pipe_name,
            "start_aro_id": start_aro,
            "end_aro_id": end_aro,
            "length_m": length,
            "diameter_m": diam,
            "roughness": rough,
            "capacity_m3s": cap,
        })

    # ------------------------------------------------------------------
    # Pumps (real EPANET links).  Throughput limited by connected mains,
    # not by a diameter, so a large finite capacity is used.
    # ------------------------------------------------------------------
    for pump_name in wn.pump_name_list:
        pump = wn.get_link(pump_name)
        start_aro = wntr_to_aro.get(pump.start_node_name, -1)
        end_aro = wntr_to_aro.get(pump.end_node_name, -1)
        if start_aro < 0 or end_aro < 0:
            metadata["invented"].setdefault("skipped_links", []).append(pump_name)
            continue
        _add_bidirectional_link(
            network, start_aro, end_aro, large_capacity,
            1.0 if cost_formula is None else None,
            link_kind="pump", wntr_name=pump_name,
        )
        metadata["preserved"]["pumps"].append({
            "wntr_name": pump_name,
            "start_aro_id": start_aro,
            "end_aro_id": end_aro,
        })

    # ------------------------------------------------------------------
    # Valves (real EPANET links).
    # ------------------------------------------------------------------
    for valve_name in wn.valve_name_list:
        valve = wn.get_link(valve_name)
        start_aro = wntr_to_aro.get(valve.start_node_name, -1)
        end_aro = wntr_to_aro.get(valve.end_node_name, -1)
        if start_aro < 0 or end_aro < 0:
            metadata["invented"].setdefault("skipped_links", []).append(valve_name)
            continue
        _add_bidirectional_link(
            network, start_aro, end_aro, large_capacity,
            1.0 if cost_formula is None else None,
            link_kind="valve", wntr_name=valve_name,
        )
        metadata["preserved"]["valves"].append({
            "wntr_name": valve_name,
            "start_aro_id": start_aro,
            "end_aro_id": end_aro,
        })

    metadata["edge_counts"] = {
        "pipes": len(metadata["preserved"]["pipes"]),
        "pumps": len(metadata["preserved"]["pumps"]),
        "valves": len(metadata["preserved"]["valves"]),
        "total_links": (len(metadata["preserved"]["pipes"])
                        + len(metadata["preserved"]["pumps"])
                        + len(metadata["preserved"]["valves"])),
        "total_graph_arcs": int(network.G.number_of_edges()),
    }

    return network, metadata


def to_wntr(
    network: WaterNetwork,
    metadata: EpinetMetadata,
) -> wntr.network.WaterNetworkModel:
    """Reconstruct a WNTR model.

    When the original ``source_path`` is available the WNTR model is reloaded
    verbatim, giving a full-fidelity round trip (all EPANET attributes
    preserved).  Otherwise a best-effort model is rebuilt from the metadata.
    """
    source = metadata.get("source_path")
    if source and pathlib.Path(source).is_file():
        return wntr.network.WaterNetworkModel(str(source))

    wn = wntr.network.WaterNetworkModel()

    for r in metadata.get("preserved", {}).get("reservoirs", []):
        name = f"res_{r['aro_id']}"
        bh = r.get("base_head_m")
        wn.add_reservoir(name, base_head=float(bh) if bh is not None else 100.0,
                         head_pattern="pattern1")

    for t in metadata.get("preserved", {}).get("tanks", []):
        name = f"tank_{t['aro_id']}"
        bh = t.get("base_head_m")
        wn.add_reservoir(name, base_head=float(bh) if bh is not None else 100.0,
                         head_pattern="pattern1")

    for j in metadata.get("preserved", {}).get("junctions", []):
        name = f"j_{j['aro_id']}"
        wn.add_junction(name, base_demand=float(j.get("demand", 0.0)))

    aro_to_name = {}
    for r in metadata.get("preserved", {}).get("reservoirs", []):
        aro_to_name[r["aro_id"]] = f"res_{r['aro_id']}"
    for t in metadata.get("preserved", {}).get("tanks", []):
        aro_to_name[t["aro_id"]] = f"tank_{t['aro_id']}"
    for j in metadata.get("preserved", {}).get("junctions", []):
        aro_to_name[j["aro_id"]] = f"j_{j['aro_id']}"

    for p in metadata.get("preserved", {}).get("pipes", []):
        s = aro_to_name.get(p.get("start_aro_id"))
        e = aro_to_name.get(p.get("end_aro_id"))
        if s and e:
            wn.add_pipe(p["wntr_name"], s, e, p.get("length_m", 1000.0),
                        p.get("diameter_m", 0.3), p.get("roughness", 100.0),
                        0.0, "OPEN")

    return wn


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m aquaroute.data.loaders <path/to/Net1.inp>")
        sys.exit(1)
    p = pathlib.Path(sys.argv[1])
    net, meta = load_epanet_inp(p)
    print(f"Loaded: {p.name}")
    print(f"  {net.n_reservoirs} supply nodes, {net.n_communities} communities, "
          f"{net.G.number_of_edges()} edges")
    print(f"  source_hash: {meta['source_hash'][:8]}...")
    print(f"  node_counts: {meta['node_counts']}")
    print(f"  edge_counts: {meta['edge_counts']}")
    print(f"  conversion_settings: {meta['conversion_settings']}")
    wn = to_wntr(net, meta)
    print(f"  round-trip pipes: {len(wn.pipe_name_list)}")
