"""Fault Tree Analysis with Probabilistic Propagation.

Implements Section 9 of the IMC Blueprint v2.0:
- Top-Level Event: Loss of Mission (LOM)
- Level 1: Propulsion, Navigation, Power, Structural, Communication failures
- Level 2: Component-level decomposition with failure rates
- Probability propagation using standard FTA mathematics
- Autonomous recovery protocol library
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from ..propulsion import FAILURE_MODES


@dataclass
class FTANode:
    """A node in the fault tree."""
    id: str
    label: str
    gate_type: str = "basic"  # "or", "and", "basic", "top"
    probability: float = 0.0
    children: List[str] = field(default_factory=list)
    failure_rate_per_hour: float = 0.0
    mitigation: str = ""
    consequence: str = ""


@dataclass
class FaultTree:
    """Complete fault tree with probability propagation."""
    nodes: List[FTANode] = field(default_factory=list)
    top_event_id: str = "lom"
    lom_probability: float = 0.0

    def compute_probabilities(self, mission_hours: float):
        """Propagate probabilities up the tree.

        For basic events: P = 1 - exp(-lambda * t)
        For OR gates: P = 1 - product(1 - P_child)
        For AND gates: P = product(P_child)
        """
        node_map = {n.id: n for n in self.nodes}

        # First pass: compute basic event probabilities
        for node in self.nodes:
            if node.gate_type == "basic":
                node.probability = 1.0 - np.exp(
                    -node.failure_rate_per_hour * mission_hours)

        # Topological sort (bottom-up)
        computed = set()

        def compute_node(node_id):
            if node_id in computed:
                return
            node = node_map[node_id]

            if node.gate_type == "basic":
                computed.add(node_id)
                return

            # Compute children first
            for child_id in node.children:
                compute_node(child_id)

            if node.gate_type == "or":
                # P(OR) = 1 - product(1 - P_i)
                product = 1.0
                for child_id in node.children:
                    product *= (1.0 - node_map[child_id].probability)
                node.probability = 1.0 - product
            elif node.gate_type == "and":
                # P(AND) = product(P_i)
                product = 1.0
                for child_id in node.children:
                    product *= node_map[child_id].probability
                node.probability = product
            elif node.gate_type == "top":
                # Top event: OR of all level-1 events
                product = 1.0
                for child_id in node.children:
                    product *= (1.0 - node_map[child_id].probability)
                node.probability = 1.0 - product

            computed.add(node_id)

        compute_node(self.top_event_id)
        self.lom_probability = node_map[self.top_event_id].probability

    def get_ranked_contributors(self) -> list:
        """Get failure modes ranked by probability contribution."""
        node_map = {n.id: n for n in self.nodes}
        basic_events = [n for n in self.nodes if n.gate_type == "basic"]
        basic_events.sort(key=lambda n: n.probability, reverse=True)
        return basic_events

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "top_event": self.top_event_id,
            "lom_probability": self.lom_probability,
            "nodes": [
                {
                    "id": n.id,
                    "label": n.label,
                    "gate_type": n.gate_type,
                    "probability": n.probability,
                    "children": n.children,
                    "mitigation": n.mitigation,
                    "consequence": n.consequence,
                }
                for n in self.nodes
            ]
        }


# Autonomous recovery protocols (Section 9.3)
RECOVERY_PROTOCOLS = [
    {
        "priority": 1,
        "fault": "Primary XNAV lost",
        "action": "Switch to backup detector; continue on inertial navigation for 30 days awaiting repair",
        "notes": "Trajectory uncertainty grows ~1000 km/day on inertial only",
    },
    {
        "priority": 2,
        "fault": "Primary propulsion lost",
        "action": "Switch to backup thruster pack; recalculate arrival trajectory; transmit new plan to Earth",
        "notes": "Backup thrust is 10% of primary; mission duration extends significantly",
    },
    {
        "priority": 3,
        "fault": "Atomic clock drift > 100 ns",
        "action": "Resynchronize clock from pulsar timing; log anomaly",
        "notes": "All TOA measurements paused during resync",
    },
    {
        "priority": 4,
        "fault": "Radiation-induced memory error",
        "action": "SEU correction via EDAC; if uncorrectable, reload from protected boot ROM",
        "notes": "Boot ROM is write-once at launch; cannot be updated",
    },
    {
        "priority": 5,
        "fault": "All navigation lost",
        "action": "Activate Dead Reckoning mode using last known state + integrated IMU; transmit distress signal",
        "notes": "Position uncertainty grows unboundedly; requires propulsive search pattern at destination",
    },
]


def build_fault_tree(architecture: str, mission_hours: float) -> FaultTree:
    """Build a complete fault tree for the selected mission profile.

    Parameters
    ----------
    architecture : 'fusion', 'acff', or 'sail'
    mission_hours : total mission duration in hours

    Returns
    -------
    FaultTree with computed probabilities
    """
    ft = FaultTree()

    # Top event
    ft.nodes.append(FTANode(
        id="lom",
        label="LOSS OF MISSION (LOM)",
        gate_type="top",
        children=["prop_fail", "nav_fail", "power_fail", "struct_fail", "comm_fail"],
    ))

    # Level 1: Major failure categories
    # Propulsion system failure
    prop_children = []
    failure_modes = FAILURE_MODES.get(architecture, FAILURE_MODES["fusion"])

    for i, fm in enumerate(failure_modes):
        node_id = f"prop_{i}"
        prop_children.append(node_id)
        ft.nodes.append(FTANode(
            id=node_id,
            label=fm["mode"],
            gate_type="basic",
            failure_rate_per_hour=fm["rate_per_hour"],
            mitigation=fm["mitigation"],
            consequence=fm["consequence"],
        ))

    ft.nodes.append(FTANode(
        id="prop_fail",
        label="Propulsion System Failure",
        gate_type="or",
        children=prop_children,
    ))

    # Navigation system failure
    ft.nodes.append(FTANode(
        id="nav_fail",
        label="Navigation System Failure",
        gate_type="and",  # ALL nav systems must fail
        children=["xnav_fail", "attitude_fail", "clock_fail"],
    ))

    ft.nodes.append(FTANode(
        id="xnav_fail", label="XNAV Detector Failure",
        gate_type="basic", failure_rate_per_hour=5e-7,
        mitigation="Redundant detector array",
        consequence="Loss of X-ray pulsar navigation",
    ))
    ft.nodes.append(FTANode(
        id="attitude_fail", label="Attitude Control Failure",
        gate_type="basic", failure_rate_per_hour=3e-7,
        mitigation="Reaction wheel + thruster backup",
        consequence="Loss of pointing knowledge",
    ))
    ft.nodes.append(FTANode(
        id="clock_fail", label="Atomic Clock Failure",
        gate_type="basic", failure_rate_per_hour=1e-7,
        mitigation="Redundant Rb/Cs standards",
        consequence="Loss of timing reference",
    ))

    # Power system failure
    ft.nodes.append(FTANode(
        id="power_fail",
        label="Power System Failure",
        gate_type="or",
        children=["rtg_fail", "bus_fail"],
    ))
    ft.nodes.append(FTANode(
        id="rtg_fail", label="RTG Power Degradation",
        gate_type="basic", failure_rate_per_hour=2e-8,
        mitigation="Multiple RTG units with graceful degradation",
        consequence="Reduced power availability",
    ))
    ft.nodes.append(FTANode(
        id="bus_fail", label="Power Bus Failure",
        gate_type="basic", failure_rate_per_hour=1e-7,
        mitigation="Triple-modular-redundant power buses",
        consequence="Complete power loss to subsystems",
    ))

    # Structural failure
    ft.nodes.append(FTANode(
        id="struct_fail",
        label="Structural Failure (Impact)",
        gate_type="basic",
        failure_rate_per_hour=1e-9,
        mitigation="Shield design with safety margin; trajectory avoidance",
        consequence="Hull breach; mission loss",
    ))

    # Communication failure
    ft.nodes.append(FTANode(
        id="comm_fail",
        label="Communication Loss",
        gate_type="basic",
        failure_rate_per_hour=5e-8,
        mitigation="Multiple antenna systems; store-and-forward capability",
        consequence="Loss of Earth contact (mission continues autonomously)",
    ))

    # Compute probabilities
    ft.compute_probabilities(mission_hours)

    return ft


def compute_lom(architecture: str, earth_transit_yr: float) -> float:
    """Quick LOM probability calculation.

    Parameters
    ----------
    architecture : 'fusion', 'acff', or 'sail'
    earth_transit_yr : mission duration in years

    Returns
    -------
    lom_probability : float
    """
    mission_hours = earth_transit_yr * 365.25 * 24.0
    ft = build_fault_tree(architecture, mission_hours)
    return ft.lom_probability
