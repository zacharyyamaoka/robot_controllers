"""The planar arm's own feed-forward and mass-matrix models, closed form, numpy only (CONTRACT amendment 3,
§3.1).

    from robot_controllers.bench.arm_models import planar_arm_feedforward, planar_arm_mass_matrix
    ff = planar_arm_feedforward(("gravity", "inertia", "coriolis", "viscous"))   # = arm.nominal_dynamics
    tau = ff(q, qd, qdd)                    # (2,) -> (2,), (N, 2) -> (N, 2)
    mass = planar_arm_mass_matrix()
    M = mass(q)                             # (2,) -> (2, 2), (N, 2) -> (N, 2, 2)

Two-link rigid-body dynamics of `arm.py`'s MJCF, with its geometry and sign: both joints about world -y, a
positive angle turns the link from +x toward +z, q = 0 points both links along +x, gravity 9.81 m/s² along -z.
Link 1 and link 2 are uniform rods (inertia m l²/12 about their middle), the tool is a solid sphere of the
tool mass (+ payload) at link 2's end, each joint has its gearbox armature and viscous damping. With the
composite link 2 (rod + tool) about the elbow,

    M2 = m2 + mt                       S2 = m2 l2 / 2 + mt l2           J2 = m2 l2² / 3 + mt l2² + 0.4 mt r²
    S1 = m1 l1 / 2                     J1 = m1 l1² / 3
    M(q) = [[a1 + J1 + J2 + M2 l1² + 2 S2 l1 cos q2,   J2 + S2 l1 cos q2],
            [J2 + S2 l1 cos q2,                         a2 + J2         ]]
    coriolis = S2 l1 sin q2 · (−(2 qd1 qd2 + qd2²), qd1²)
    gravity  = g · ((S1 + M2 l1) cos q1 + S2 cos(q1 + q2),  S2 cos(q1 + q2))
    viscous  = b · qd          coulomb = frictionloss · tanh(qd / 0.01 rad/s)

The oracle is MuJoCo: with {gravity, inertia, coriolis, viscous} this equals `arm.nominal_dynamics`
(`mj_inverse` on the nominal model) to 1e-9 N·m, and M(q) equals its columns
(tests/test_layout_acceptance.py). `coulomb` is the gearbox's dry friction, which the nominal model leaves out
(a constraint force mj_inverse cannot know).

The models are small classes with a stable repr (not closures), so a ledger that records a config shows
`planar_arm_feedforward(('gravity',), payload_kg=0.0)` instead of a memory address.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robot_controllers.bench.arm import ARM, ArmConfig

GRAVITY_M_S2 = 9.81  # the MJCF's `<option gravity="0 0 -9.81">`
COULOMB_SMOOTHING_RAD_S = 0.01  # coulomb = frictionloss · tanh(qd / 0.01 rad/s) (CONTRACT §3.1)
TERMS: tuple[str, ...] = ("gravity", "inertia", "coriolis", "viscous", "coulomb")
RIGID_BODY: tuple[str, ...] = ("gravity", "inertia", "coriolis", "viscous")  # = the nominal model


class _Parameters:
    """The lumped inertial parameters of the two links (see the module docstring)."""

    __slots__ = ("l1", "a1", "a2", "b1", "b2", "f1", "f2", "s1", "m2", "s2", "j1", "j2")

    def __init__(self, arm: ArmConfig, payload_kg: float) -> None:
        l1, l2 = arm.link1_length_m, arm.link2_length_m
        m1, m2, tool = arm.link1_mass_kg, arm.link2_mass_kg, arm.tool_mass_kg + payload_kg
        self.l1 = l1
        self.a1, self.a2 = (float(v) for v in arm.armature_kg_m2)
        self.b1, self.b2 = (float(v) for v in arm.damping_nm_s_rad)
        self.f1, self.f2 = (float(v) for v in arm.frictionloss_nm)
        # WHY the MJCF's own rounding: arm.build_mjcf writes each rod's inertia about its middle with 9
        # significant digits (link 2's m l²/12 = 0.00416666667) and the tool's with 9; the closed form
        # describes the arm the bench simulates, so it takes the same numbers (3e-12 kg·m² otherwise, 1e-10
        # N·m at 20 rad/s²).
        rod1 = float(f"{m1 * l1**2 / 12.0:.9g}")
        rod2 = float(f"{m2 * l2**2 / 12.0:.9g}")
        sphere = float(f"{0.4 * tool * arm.tool_radius_m**2:.9g}")
        self.s1 = m1 * l1 / 2.0
        self.j1 = rod1 + m1 * (l1 / 2.0) ** 2
        self.m2 = m2 + tool
        self.s2 = m2 * l2 / 2.0 + tool * l2
        self.j2 = rod2 + m2 * (l2 / 2.0) ** 2 + tool * l2**2 + sphere


def _arm_text(arm: ArmConfig) -> str:
    return "" if arm == ARM else f", arm={arm!r}"


class PlanarArmMassMatrix:
    """`mass(q) -> M(q)`: the planar arm's joint-space inertia (armature included), (2,) -> (2, 2), (N, 2) ->
    (N, 2, 2). Pure."""

    __slots__ = ("arm", "_p")

    def __init__(self, arm: ArmConfig = ARM) -> None:
        self.arm = arm
        self._p = _Parameters(arm, 0.0)

    def __call__(self, q: Any) -> np.ndarray:
        p = self._p
        q = np.asarray(q, dtype=np.float64)
        coupling = p.s2 * p.l1 * np.cos(q[..., 1])
        m11 = p.a1 + p.j1 + p.j2 + p.m2 * p.l1**2 + 2.0 * coupling
        m12 = p.j2 + coupling
        m22 = np.broadcast_to(p.a2 + p.j2, np.shape(m12))
        return np.stack([np.stack([m11, m12], axis=-1), np.stack([m12, m22], axis=-1)], axis=-2)

    def __repr__(self) -> str:
        return f"planar_arm_mass_matrix({_arm_text(self.arm).lstrip(', ')})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PlanarArmMassMatrix) and other.arm == self.arm

    def __hash__(self) -> int:
        return hash(("planar_arm_mass_matrix", self.arm))


class PlanarArmFeedforward:
    """`ff(q, qd, qdd) -> tau` for the chosen dynamics terms, (2,) -> (2,), (N, 2) -> (N, 2). Pure."""

    __slots__ = ("terms", "payload_kg", "arm", "_p", "_use")

    def __init__(self, terms: tuple[str, ...], arm: ArmConfig = ARM, payload_kg: float = 0.0) -> None:
        terms = tuple(str(t) for t in terms)
        unknown = [t for t in terms if t not in TERMS]
        if not terms or unknown or len(set(terms)) != len(terms):
            raise ValueError(f"terms must be distinct names from {TERMS}, got {terms!r}")
        payload = float(payload_kg)
        if not (np.isfinite(payload) and payload >= 0.0):
            raise ValueError(f"payload_kg must be finite and >= 0, got {payload_kg!r}")
        self.terms = terms
        self.payload_kg = payload
        self.arm = arm
        self._p = _Parameters(arm, payload)
        self._use = frozenset(terms)

    def __call__(self, q: Any, qd: Any, qdd: Any) -> np.ndarray:
        p, use = self._p, self._use
        q = np.asarray(q, dtype=np.float64)
        qd = np.asarray(qd, dtype=np.float64)
        qdd = np.asarray(qdd, dtype=np.float64)
        shape = np.broadcast_shapes(q.shape, qd.shape, qdd.shape)
        tau1 = np.zeros(shape[:-1])
        tau2 = np.zeros(shape[:-1])
        if "gravity" in use:
            c12 = np.cos(q[..., 0] + q[..., 1])
            tau1 = tau1 + GRAVITY_M_S2 * ((p.s1 + p.m2 * p.l1) * np.cos(q[..., 0]) + p.s2 * c12)
            tau2 = tau2 + GRAVITY_M_S2 * p.s2 * c12
        if "inertia" in use:
            coupling = p.s2 * p.l1 * np.cos(q[..., 1])
            m11 = p.a1 + p.j1 + p.j2 + p.m2 * p.l1**2 + 2.0 * coupling
            m12 = p.j2 + coupling
            tau1 = tau1 + (m11 * qdd[..., 0] + m12 * qdd[..., 1])
            tau2 = tau2 + (m12 * qdd[..., 0] + (p.a2 + p.j2) * qdd[..., 1])
        if "coriolis" in use:
            h = p.s2 * p.l1 * np.sin(q[..., 1])
            w1, w2 = qd[..., 0], qd[..., 1]
            tau1 = tau1 - h * (2.0 * w1 * w2 + w2 * w2)
            tau2 = tau2 + h * w1 * w1
        if "viscous" in use:
            tau1 = tau1 + p.b1 * qd[..., 0]
            tau2 = tau2 + p.b2 * qd[..., 1]
        if "coulomb" in use:
            tau1 = tau1 + p.f1 * np.tanh(qd[..., 0] / COULOMB_SMOOTHING_RAD_S)
            tau2 = tau2 + p.f2 * np.tanh(qd[..., 1] / COULOMB_SMOOTHING_RAD_S)
        return np.stack([np.broadcast_to(tau1, shape[:-1]), np.broadcast_to(tau2, shape[:-1])], axis=-1)

    def __repr__(self) -> str:
        return f"planar_arm_feedforward({self.terms!r}, payload_kg={self.payload_kg:g}{_arm_text(self.arm)})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PlanarArmFeedforward) and (other.terms, other.payload_kg, other.arm) == (
            self.terms,
            self.payload_kg,
            self.arm,
        )

    def __hash__(self) -> int:
        return hash(("planar_arm_feedforward", self.terms, self.payload_kg, self.arm))


def planar_arm_feedforward(
    terms: tuple[str, ...] = RIGID_BODY, arm: ArmConfig = ARM, payload_kg: float = 0.0
) -> PlanarArmFeedforward:
    """The planar arm's feed-forward for `terms` ⊆ {gravity, inertia, coriolis, viscous, coulomb}.

    `payload_kg` is a payload on the tool the model is told about (0: the rungs' payload stays unknown).
    """
    return PlanarArmFeedforward(tuple(terms), arm=arm, payload_kg=payload_kg)


def planar_arm_mass_matrix(arm: ArmConfig = ARM) -> PlanarArmMassMatrix:
    """The planar arm's joint-space inertia M(q), armature included, no payload."""
    return PlanarArmMassMatrix(arm)
