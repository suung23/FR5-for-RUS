"""Optical-flow abstraction: backends, backward warping and reliability maps."""

from .backends import FarnebackFlow, IdentityFlow, RaftSmallFlow, build_flow_backend
from .base import FLOW_SCHEMA_VERSION, FlowBackend, FlowPair
from .precomputed import (
    PrecomputedFlow,
    is_valid_flow_file,
    load_flow_pair,
    save_flow_pair,
)
from .reliability import ReliabilityConfig, ReliabilityOutput, compute_reliability, soft_score
from .warp import (
    flow_to_sampling_grid,
    forward_backward_residual,
    sample_flow,
    warp_backward,
)

__all__ = [
    "FlowPair",
    "FlowBackend",
    "FLOW_SCHEMA_VERSION",
    "FarnebackFlow",
    "IdentityFlow",
    "RaftSmallFlow",
    "PrecomputedFlow",
    "build_flow_backend",
    "save_flow_pair",
    "load_flow_pair",
    "is_valid_flow_file",
    "warp_backward",
    "flow_to_sampling_grid",
    "sample_flow",
    "forward_backward_residual",
    "ReliabilityConfig",
    "ReliabilityOutput",
    "compute_reliability",
    "soft_score",
]
