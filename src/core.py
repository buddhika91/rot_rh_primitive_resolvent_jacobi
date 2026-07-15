"""Shared helpers for demos built on the canonical resolvent implementation."""
from __future__ import annotations
import importlib.util
from pathlib import Path

_IMPL = Path(__file__).resolve().parent / "audits" / "rot_rh_primitive_resolvent_gue_audit.py"
_spec = importlib.util.spec_from_file_location("rot_rh_resolvent_impl", _IMPL)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

xi = _mod.xi
xi_taylor_cauchy = _mod.xi_taylor_cauchy
phi_from_xi_coeffs = _mod.phi_from_xi_coeffs
primitive_resolvent = _mod.primitive_resolvent
stieltjes_jacobi = _mod.stieltjes_jacobi
jacobi_matrix = _mod.jacobi_matrix
predict_gamma = _mod.predict_gamma
