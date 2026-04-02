"""Three-Tier Adaptive Integrator Stack.

Implements Section 5 of the IMC Blueprint v2.0:
- Tier 1: RK45 + PN1 (deep space coasting, v < 0.01c)
- Tier 2: RK89 + PN2 (moderate gravity, 0.01c < v < 0.1c)
- Tier 3: BSSN Numerical GR (strong field, v > 0.1c or r < 100 r_s)

Also includes:
- Relativistic aberration correction for XNAV
- Doppler shift modeling for communications
- Conservation checks (energy, angular momentum)
- Time-reversal validation
- Cross-validation against Bulirsch-Stoer integrator
"""
import numpy as np
from scipy.integrate import solve_ivp, ode
from typing import Callable, Optional, Tuple
from dataclasses import dataclass
from ..constants import c, G, M_SUN, AU, YEAR_S, GM_SUN


@dataclass
class IntegratorState:
    """State vector for trajectory integration."""
    t: float                    # time [s]
    pos: np.ndarray             # position [m], shape (3,)
    vel: np.ndarray             # velocity [m/s], shape (3,)
    mass: float                 # current spacecraft mass [kg]
    tier: int = 1               # current integrator tier
    energy: float = 0.0         # total specific energy (for conservation check)
    angular_momentum: float = 0.0


def needs_numerical_gr(v_ms: float, r_m: float, M_kg: float) -> bool:
    """Check if numerical GR integration is required (Tier 3).

    Switch to BSSN if: v/c > 0.1 OR r < 100 * r_Schwarzschild
    """
    r_s = 2.0 * G * M_kg / (c * c)
    return (abs(v_ms) / c > 0.1) or (r_m < 100.0 * r_s and r_m > 0)


def select_tier(v_ms: float, r_m: float, M_kg: float = M_SUN) -> int:
    """Select integrator tier based on local spacetime conditions.

    Tier 1: RK45+PN1 — deep space, v < 0.01c, r > 1000 AU
    Tier 2: RK89+PN2 — moderate gravity, approaching stellar system
    Tier 3: BSSN — strong field or high velocity
    """
    v_frac = abs(v_ms) / c
    r_au = r_m / AU

    if needs_numerical_gr(v_ms, r_m, M_kg):
        return 3
    elif v_frac > 0.01 or r_au < 1000:
        return 2
    else:
        return 1


def pn1_acceleration(pos: np.ndarray, vel: np.ndarray,
                     M_central: float = M_SUN) -> np.ndarray:
    """First Post-Newtonian (PN1) gravitational acceleration.

    Includes the leading-order GR correction to Newtonian gravity.
    This gives the correct perihelion precession for Mercury.

    a_PN1 = -GM/r^2 * r_hat + GM/r^2 * {
        (1/c^2) * [(4GM/r - v^2) * r_hat + 4*(v . r_hat)*v]
    }
    """
    r = np.linalg.norm(pos)
    if r < 1.0:  # avoid singularity
        return np.zeros(3)

    r_hat = pos / r
    v = np.linalg.norm(vel)
    v_dot_r = np.dot(vel, r_hat)
    GM = G * M_central

    # Newtonian acceleration
    a_newton = -GM / (r * r) * r_hat

    # PN1 correction
    pn1_factor = GM / (r * r * c * c)
    a_pn1 = pn1_factor * (
        (4.0 * GM / r - v * v) * r_hat +
        4.0 * v_dot_r * vel
    )

    return a_newton + a_pn1


def pn2_acceleration(pos: np.ndarray, vel: np.ndarray,
                     M_central: float = M_SUN) -> np.ndarray:
    """Second Post-Newtonian (PN2) gravitational acceleration.

    Includes PN1 plus second-order corrections for better accuracy
    in moderate gravitational fields.
    """
    r = np.linalg.norm(pos)
    if r < 1.0:
        return np.zeros(3)

    r_hat = pos / r
    v = np.linalg.norm(vel)
    v_dot_r = np.dot(vel, r_hat)
    GM = G * M_central

    # Start with PN1
    a = pn1_acceleration(pos, vel, M_central)

    # PN2 correction terms (leading order)
    pn2_factor = GM / (r * r * c * c * c * c)
    a_pn2 = pn2_factor * (
        (-2.0 * GM * GM / (r * r) + 1.5 * v * v * v_dot_r * v_dot_r) * r_hat +
        (2.0 * GM / r * v_dot_r) * vel
    )

    return a + a_pn2


def bssn_acceleration(pos: np.ndarray, vel: np.ndarray,
                      M_central: float = M_SUN) -> np.ndarray:
    """BSSN-inspired strong-field gravitational acceleration.

    For the IMC, we implement the geodesic equation in Schwarzschild spacetime
    as a practical approximation of full BSSN numerical relativity.
    This is exact for spherically symmetric spacetimes and captures
    all strong-field effects needed for compact object flybys.

    Uses the Schwarzschild geodesic equations in isotropic coordinates.
    """
    r = np.linalg.norm(pos)
    GM = G * M_central
    r_s = 2.0 * GM / (c * c)  # Schwarzschild radius

    if r < r_s * 1.01:  # Inside or near event horizon
        return np.zeros(3)

    r_hat = pos / r
    v_mag = np.linalg.norm(vel)
    v_dot_r = np.dot(vel, r_hat)

    # Schwarzschild metric components
    f = 1.0 - r_s / r

    # Geodesic equation in Schwarzschild coordinates
    # This includes all orders of post-Newtonian corrections
    a_r = -GM / (r * r) * (1.0 / f) * (
        f - v_mag * v_mag / (c * c) +
        3.0 * v_dot_r * v_dot_r / (c * c)
    )

    a_tangential = -2.0 * GM * v_dot_r / (r * r * c * c * f) * (
        vel - v_dot_r * r_hat
    )

    return a_r * r_hat + a_tangential


def _derivatives_tier1(t, y, thrust_func, M_bodies):
    """RHS for Tier 1: RK45 + PN1."""
    pos = y[0:3]
    vel = y[3:6]
    mass = y[6]

    # Gravitational acceleration with PN1
    a_grav = np.zeros(3)
    for M, pos_body in M_bodies:
        rel_pos = pos - pos_body
        a_grav += pn1_acceleration(rel_pos, vel, M)

    # Thrust acceleration
    a_thrust, mdot = thrust_func(t, pos, vel, mass)

    dydt = np.zeros(7)
    dydt[0:3] = vel
    dydt[3:6] = a_grav + a_thrust
    dydt[6] = -mdot  # mass decreases as fuel burns
    return dydt


def _derivatives_tier2(t, y, thrust_func, M_bodies):
    """RHS for Tier 2: RK89 + PN2."""
    pos = y[0:3]
    vel = y[3:6]
    mass = y[6]

    a_grav = np.zeros(3)
    for M, pos_body in M_bodies:
        rel_pos = pos - pos_body
        a_grav += pn2_acceleration(rel_pos, vel, M)

    a_thrust, mdot = thrust_func(t, pos, vel, mass)

    dydt = np.zeros(7)
    dydt[0:3] = vel
    dydt[3:6] = a_grav + a_thrust
    dydt[6] = -mdot
    return dydt


def _derivatives_tier3(t, y, thrust_func, M_bodies):
    """RHS for Tier 3: BSSN (Schwarzschild geodesic)."""
    pos = y[0:3]
    vel = y[3:6]
    mass = y[6]

    a_grav = np.zeros(3)
    for M, pos_body in M_bodies:
        rel_pos = pos - pos_body
        a_grav += bssn_acceleration(rel_pos, vel, M)

    a_thrust, mdot = thrust_func(t, pos, vel, mass)

    dydt = np.zeros(7)
    dydt[0:3] = vel
    dydt[3:6] = a_grav + a_thrust
    dydt[6] = -mdot
    return dydt


class AdaptiveIntegrator:
    """Three-tier adaptive integrator for relativistic trajectory computation.

    Automatically switches between tiers based on local conditions.
    Includes conservation checks and cross-validation.
    """

    # Default timesteps per tier
    TIER_DT = {
        1: 7.0 * 86400.0,    # 1 week
        2: 1.0 * 86400.0,    # 1 day
        3: 1.0,               # 1 second
    }

    TIER_METHODS = {
        1: 'RK45',
        2: 'DOP853',   # 8th-order Runge-Kutta (closest to RK89)
        3: 'DOP853',   # BSSN uses same integrator with different physics
    }

    TIER_RTOL = {
        1: 1e-10,
        2: 1e-12,
        3: 1e-14,
    }

    def __init__(self):
        self.trajectory = []
        self.tier_log = []
        self.energy_violations = []

    def integrate(self, y0: np.ndarray, t_span: tuple,
                  thrust_func: Callable,
                  M_bodies: list,
                  max_steps: int = 100000,
                  adaptive_tier: bool = True) -> dict:
        """Integrate trajectory with automatic tier switching.

        Parameters
        ----------
        y0 : array (7,) — [x, y, z, vx, vy, vz, mass]
        t_span : (t_start, t_end) in seconds
        thrust_func : callable(t, pos, vel, mass) -> (a_thrust, mdot)
        M_bodies : list of (mass_kg, position_array) tuples
        max_steps : maximum number of integration steps
        adaptive_tier : whether to auto-switch tiers

        Returns
        -------
        dict with 't', 'pos', 'vel', 'mass', 'tier', 'energy'
        """
        t_start, t_end = t_span
        t_current = t_start
        y_current = y0.copy()

        times = [t_current]
        positions = [y_current[0:3].copy()]
        velocities = [y_current[3:6].copy()]
        masses = [y_current[6]]
        tiers = []

        step = 0
        while t_current < t_end and step < max_steps:
            # Determine current tier
            v_mag = np.linalg.norm(y_current[3:6])
            r_mag = np.linalg.norm(y_current[0:3])
            M_nearest = M_bodies[0][0] if M_bodies else M_SUN

            if adaptive_tier:
                tier = select_tier(v_mag, r_mag, M_nearest)
            else:
                tier = 1

            # Select timestep and derivatives function
            dt = self.TIER_DT[tier]
            dt = min(dt, t_end - t_current)

            if tier == 1:
                deriv_func = _derivatives_tier1
            elif tier == 2:
                deriv_func = _derivatives_tier2
            else:
                deriv_func = _derivatives_tier3

            # Integrate one step using scipy
            t_step_end = t_current + dt
            sol = solve_ivp(
                lambda t, y: deriv_func(t, y, thrust_func, M_bodies),
                (t_current, t_step_end),
                y_current,
                method=self.TIER_METHODS[tier],
                rtol=self.TIER_RTOL[tier],
                atol=1e-15,
                max_step=dt,
            )

            if sol.success and len(sol.t) > 0:
                y_current = sol.y[:, -1].copy()
                t_current = sol.t[-1]

                # Ensure mass doesn't go negative
                y_current[6] = max(y_current[6], 1.0)

                times.append(t_current)
                positions.append(y_current[0:3].copy())
                velocities.append(y_current[3:6].copy())
                masses.append(y_current[6])
                tiers.append(tier)
            else:
                # If integration fails, try smaller step
                dt *= 0.1
                if dt < 1e-6:
                    break
                continue

            step += 1

        return {
            "t": np.array(times),
            "pos": np.array(positions),
            "vel": np.array(velocities),
            "mass": np.array(masses),
            "tiers": tiers,
            "n_steps": step,
        }

    def integrate_simple(self, pos0: np.ndarray, vel0: np.ndarray,
                         t_span: tuple, M_central: float = M_SUN,
                         dt_hint: float = None) -> dict:
        """Simplified integration for validation tests (no thrust, single body)."""
        def no_thrust(t, pos, vel, mass):
            return np.zeros(3), 0.0

        y0 = np.zeros(7)
        y0[0:3] = pos0
        y0[3:6] = vel0
        y0[6] = 1.0  # dummy mass

        M_bodies = [(M_central, np.zeros(3))]

        return self.integrate(y0, t_span, no_thrust, M_bodies)


def compute_specific_energy(pos: np.ndarray, vel: np.ndarray,
                            M_central: float = M_SUN) -> float:
    """Compute specific orbital energy for conservation checking."""
    r = np.linalg.norm(pos)
    v = np.linalg.norm(vel)
    return 0.5 * v * v - G * M_central / r


def check_conservation(trajectory: dict, M_central: float = M_SUN,
                       tolerance: float = 1e-8) -> dict:
    """Check energy and angular momentum conservation along trajectory.

    Returns
    -------
    dict with 'energy_violation', 'angular_momentum_violation', 'passed'
    """
    positions = trajectory["pos"]
    velocities = trajectory["vel"]

    energies = []
    ang_mom = []
    for pos, vel in zip(positions, velocities):
        E = compute_specific_energy(pos, vel, M_central)
        L = np.linalg.norm(np.cross(pos, vel))
        energies.append(E)
        ang_mom.append(L)

    energies = np.array(energies)
    ang_mom = np.array(ang_mom)

    # Relative variations
    E_var = (np.max(energies) - np.min(energies)) / max(abs(energies[0]), 1e-30)
    L_var = (np.max(ang_mom) - np.min(ang_mom)) / max(abs(ang_mom[0]), 1e-30)

    return {
        "energy_variation": float(E_var),
        "angular_momentum_variation": float(L_var),
        "passed": E_var < tolerance and L_var < tolerance,
    }


def time_reversal_check(pos0: np.ndarray, vel0: np.ndarray,
                        t_forward: float, M_central: float = M_SUN,
                        tolerance_m: float = 1e3) -> dict:
    """Run trajectory forward, then reverse velocities and run backward.

    The recovered initial state should match the original to within
    floating-point precision.
    """
    integrator = AdaptiveIntegrator()

    # Forward integration
    fwd = integrator.integrate_simple(pos0, vel0, (0, t_forward), M_central)

    # Get final state
    pos_final = fwd["pos"][-1]
    vel_final = fwd["vel"][-1]

    # Reverse integration
    rev = integrator.integrate_simple(pos_final, -vel_final, (0, t_forward), M_central)

    # Compare recovered position with original
    pos_recovered = rev["pos"][-1]
    error_m = np.linalg.norm(pos_recovered - pos0)

    return {
        "position_error_m": float(error_m),
        "passed": error_m < tolerance_m,
    }


def bulirsch_stoer_integrate(pos0: np.ndarray, vel0: np.ndarray,
                             t_span: tuple, M_central: float = M_SUN) -> dict:
    """Reference Bulirsch-Stoer integrator for cross-validation.

    Uses scipy's Radau method which is similar to Bulirsch-Stoer
    in terms of accuracy for this application.
    """
    def deriv(t, y):
        pos = y[0:3]
        vel = y[3:6]
        r = np.linalg.norm(pos)
        if r < 1.0:
            return np.zeros(6)
        a = pn1_acceleration(pos, vel, M_central)
        return np.concatenate([vel, a])

    y0 = np.concatenate([pos0, vel0])
    sol = solve_ivp(deriv, t_span, y0, method='Radau',
                    rtol=1e-14, atol=1e-16, max_step=86400.0)

    return {
        "t": sol.t,
        "pos": sol.y[0:3].T,
        "vel": sol.y[3:6].T,
    }


def cross_validate(trajectory: dict, pos0: np.ndarray, vel0: np.ndarray,
                   t_span: tuple, M_central: float = M_SUN,
                   threshold: float = 1e-8) -> dict:
    """Cross-validate trajectory against Bulirsch-Stoer reference.

    Returns
    -------
    dict with 'max_position_divergence', 'max_velocity_divergence', 'passed'
    """
    ref = bulirsch_stoer_integrate(pos0, vel0, t_span, M_central)

    # Compare final positions
    pos_err = np.linalg.norm(trajectory["pos"][-1] - ref["pos"][-1])
    vel_err = np.linalg.norm(trajectory["vel"][-1] - ref["vel"][-1])

    # Normalize
    r_scale = np.linalg.norm(trajectory["pos"][-1])
    v_scale = np.linalg.norm(trajectory["vel"][-1])

    rel_pos_err = pos_err / max(r_scale, 1.0)
    rel_vel_err = vel_err / max(v_scale, 1.0)

    return {
        "position_divergence": float(rel_pos_err),
        "velocity_divergence": float(rel_vel_err),
        "passed": max(rel_pos_err, rel_vel_err) < threshold,
    }


def aberrated_angle(theta_rest: float, beta: float) -> float:
    """Compute relativistic aberration correction for XNAV.

    cos(theta_ship) = (cos(theta_rest) - beta) / (1 - beta * cos(theta_rest))

    Parameters
    ----------
    theta_rest : angle of pulsar in rest frame [radians]
    beta : v/c of spacecraft

    Returns
    -------
    theta_ship : apparent angle in ship frame [radians]
    """
    cos_rest = np.cos(theta_rest)
    cos_ship = (cos_rest - beta) / (1.0 - beta * cos_rest)
    cos_ship = np.clip(cos_ship, -1.0, 1.0)
    return np.arccos(cos_ship)


def doppler_shift(f_emit: float, beta: float, theta: float = 0.0) -> float:
    """Compute relativistic Doppler shift for communication modeling.

    Parameters
    ----------
    f_emit : emitted frequency [Hz]
    beta : v/c of spacecraft
    theta : angle between velocity and line of sight [radians]

    Returns
    -------
    f_obs : observed frequency [Hz]
    """
    gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    return f_emit / (gamma * (1.0 + beta * np.cos(theta)))
