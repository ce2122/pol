"""Propulsion Physics Module — First-Principles Drive Models.

Implements Section 4 of the IMC Blueprint v2.0:
- Architecture A: Pulsed Fusion Drive (ICF) — D/He-3 pellets, magnetic nozzle
- Architecture B: Antimatter Catalyzed Fission-Fusion (ACFF)
- Architecture C: Laser-Driven Lightsail
- Failure mode library for all architectures
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional
from ..constants import c, g0, eV, MeV, M_H, AMU


@dataclass
class FusionDriveParams:
    """Pulsed Fusion Drive (ICF) parameters derived from first principles.

    Based on Project Longshot/Daedalus heritage with contemporary ICF data.
    """
    pellet_mass_kg: float = 3.0e-4         # Mass per D/He-3 pellet [kg]
    pellet_frequency_hz: float = 250.0     # Ignition rate [Hz]
    driver_efficiency: float = 0.03        # Laser/ion driver wall-plug efficiency
    fusion_gain_Q: float = 12.0            # Fusion energy gain factor
    nozzle_efficiency: float = 0.67        # Magnetic nozzle thrust efficiency
    nozzle_erosion_rate_mm_hr: float = 0.01  # Nozzle erosion rate

    # Derived quantities (computed in __post_init__)
    exhaust_velocity_ms: float = 0.0
    specific_impulse_s: float = 0.0
    thrust_N: float = 0.0
    mdot_kgs: float = 0.0

    def __post_init__(self):
        # D/He-3 fusion: Q-value = 18.3 MeV per reaction
        E_fusion_per_kg = 18.3e6 * eV / (5 * AMU)  # J/kg of D-He3 fuel

        # Overall efficiency: driver * gain * nozzle
        eta = self.driver_efficiency * self.fusion_gain_Q * self.nozzle_efficiency

        # Exhaust velocity from energy balance: v_e = sqrt(2 * eta * E_fusion_per_kg)
        self.exhaust_velocity_ms = np.sqrt(2.0 * eta * E_fusion_per_kg)

        # Cap at speed of light
        self.exhaust_velocity_ms = min(self.exhaust_velocity_ms, 0.10 * c)

        # Specific impulse
        self.specific_impulse_s = self.exhaust_velocity_ms / g0

        # Mass flow rate
        self.mdot_kgs = self.pellet_frequency_hz * self.pellet_mass_kg

        # Thrust
        self.thrust_N = self.mdot_kgs * self.exhaust_velocity_ms


@dataclass
class ACFFDriveParams:
    """Antimatter Catalyzed Fission-Fusion drive parameters.

    Antiproton-triggered subcritical U-238 → fission → D-T fusion.
    """
    pellet_mass_kg: float = 7.5e-4        # Total pellet mass (U-238 + D-T)
    charged_fraction: float = 0.30         # Fraction of energy in charged particles
    pellet_frequency_hz: float = 100.0     # Hz
    trap_redundancy: int = 3               # Number of Penning trap arrays

    # Derived
    exhaust_velocity_ms: float = 0.0
    specific_impulse_s: float = 0.0
    thrust_N: float = 0.0
    mdot_kgs: float = 0.0
    antimatter_per_pellet_kg: float = 1e-12  # ~1 nanogram antiprotons per pellet

    def __post_init__(self):
        # Energy budget per pellet:
        # E_fission ~ 200 MeV per U-238 fission (from antiproton catalysis)
        # E_fusion  ~ 17.6 MeV per D-T pair
        E_fission_J = 200.0 * MeV
        E_fusion_J = 17.6 * MeV
        E_total_per_reaction = E_fission_J + E_fusion_J

        # Number of reactions per pellet (approximate)
        # U-238 mass = 0.25g = 2.5e-4 kg; each U-238 nucleus ~ 238 AMU
        n_fissions = 2.5e-4 / (238 * AMU)

        E_total_pellet = n_fissions * E_total_per_reaction
        E_charged = self.charged_fraction * E_total_pellet

        # Exhaust velocity from kinetic energy of charged fragments
        self.exhaust_velocity_ms = np.sqrt(2.0 * E_charged / self.pellet_mass_kg)

        # Cap at 0.5c (physical limit for ACFF)
        self.exhaust_velocity_ms = min(self.exhaust_velocity_ms, 0.5 * c)

        self.specific_impulse_s = self.exhaust_velocity_ms / g0
        self.mdot_kgs = self.pellet_frequency_hz * self.pellet_mass_kg
        self.thrust_N = self.mdot_kgs * self.exhaust_velocity_ms


@dataclass
class LightsailParams:
    """Laser-Driven Lightsail parameters.

    Beamed-energy architecture with phased laser array.
    """
    laser_power_W: float = 100.0e9          # Total laser array power [W]
    sail_area_m2: float = 1.0e4             # Sail area [m^2]
    sail_mass_kg: float = 10.0              # Sail + payload mass [kg]
    reflectivity: float = 0.9999            # Sail reflectivity
    acceleration_time_s: float = 10 * 86400  # Acceleration window [s]
    deceleration_method: str = "magnetic_braking"  # or "retro_sail", "dest_laser"

    # Derived
    thrust_N: float = 0.0
    acceleration_ms2: float = 0.0
    max_velocity_ms: float = 0.0
    max_velocity_frac_c: float = 0.0

    def __post_init__(self):
        # Photon momentum: F = 2 * P * R / c (factor 2 for reflection)
        self.thrust_N = 2.0 * self.laser_power_W * self.reflectivity / c

        # Acceleration
        self.acceleration_ms2 = self.thrust_N / self.sail_mass_kg

        # Maximum velocity (limited by acceleration time and 0.25c cap)
        v_achievable = self.acceleration_ms2 * self.acceleration_time_s
        self.max_velocity_ms = min(v_achievable, 0.25 * c)
        self.max_velocity_frac_c = self.max_velocity_ms / c


# Failure mode library
FAILURE_MODES = {
    "fusion": [
        {"mode": "Pellet ignition misfire",
         "consequence": "Thrust interruption; trajectory deviation",
         "mitigation": "Redundant igniter arrays; missed-pellet compensation burn",
         "rate_per_hour": 1e-5},
        {"mode": "Nozzle throat erosion failure",
         "consequence": "Loss of thrust direction control",
         "mitigation": "Spare nozzle sections; mission-abort delta-V budget",
         "rate_per_hour": 5e-7},
        {"mode": "Fuel line fracture",
         "consequence": "Propellant leak, fire/explosion risk",
         "mitigation": "Cryogenic isolation valves; redundant feed lines",
         "rate_per_hour": 1e-7},
        {"mode": "Power bus failure",
         "consequence": "Drive and all avionics at risk",
         "mitigation": "Triple-modular-redundant power buses; RTG backup",
         "rate_per_hour": 2e-7},
    ],
    "acff": [
        {"mode": "Antiproton trap quench",
         "consequence": "Immediate loss of catalytic propellant",
         "mitigation": "Triple-redundant traps; fission-only degraded mode",
         "rate_per_hour": 3e-7},
        {"mode": "Pellet ignition misfire",
         "consequence": "Thrust interruption",
         "mitigation": "Redundant igniter arrays",
         "rate_per_hour": 1e-5},
        {"mode": "Nozzle throat erosion failure",
         "consequence": "Loss of thrust direction control",
         "mitigation": "Purely magnetic nozzle design; spare segments",
         "rate_per_hour": 3e-7},
        {"mode": "Power bus failure",
         "consequence": "Drive and all avionics at risk",
         "mitigation": "Triple-modular-redundant power buses; RTG backup",
         "rate_per_hour": 2e-7},
    ],
    "sail": [
        {"mode": "Laser array defocus",
         "consequence": "Sail heating, structural failure",
         "mitigation": "Adaptive optics correction; emergency sail jettison",
         "rate_per_hour": 1e-6},
        {"mode": "Sail material degradation",
         "consequence": "Reduced reflectivity, thermal buildup",
         "mitigation": "Multi-layer sail design; thermal monitoring",
         "rate_per_hour": 5e-7},
        {"mode": "Power bus failure",
         "consequence": "Drive and all avionics at risk",
         "mitigation": "Triple-modular-redundant power buses; RTG backup",
         "rate_per_hour": 2e-7},
    ],
}


def get_propulsion_model(architecture: str, **kwargs):
    """Factory function to create propulsion model.

    Parameters
    ----------
    architecture : str — 'fusion', 'acff', or 'sail'
    **kwargs — override default parameters

    Returns
    -------
    Propulsion parameters dataclass
    """
    if architecture == "fusion":
        return FusionDriveParams(**kwargs)
    elif architecture == "acff":
        return ACFFDriveParams(**kwargs)
    elif architecture == "sail":
        return LightsailParams(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def relativistic_delta_v(ve_ms: float, m0_kg: float, mf_kg: float) -> float:
    """Compute relativistic delta-V using the relativistic rocket equation.

    dV = c * tanh(ve/c * ln(m0/mf))

    Parameters
    ----------
    ve_ms : exhaust velocity [m/s]
    m0_kg : initial mass (with fuel) [kg]
    mf_kg : final mass (without fuel) [kg]

    Returns
    -------
    delta_v : float — achievable delta-V [m/s]
    """
    if ve_ms <= 0 or m0_kg <= mf_kg:
        return 0.0
    beta_e = ve_ms / c
    ln_mr = np.log(m0_kg / mf_kg)
    beta_ship = np.tanh(beta_e * ln_mr)
    return beta_ship * c


def fuel_fraction_needed(ve_ms: float, dv_ms: float) -> float:
    """Compute fuel fraction needed for a given delta-V.

    Inverse of the relativistic rocket equation.
    """
    if ve_ms <= 0:
        return 0.99
    beta = dv_ms / c
    if beta >= 1.0:
        return 0.9999
    beta_e = ve_ms / c
    # dv/c = tanh(ve/c * ln(m0/mf))
    # atanh(beta) = beta_e * ln(m0/mf)
    mr = np.exp(np.arctanh(min(beta, 0.9999)) / max(beta_e, 1e-20))
    return 1.0 - 1.0 / mr


def lorentz_factor(v_ms: float) -> float:
    """Compute Lorentz factor gamma = 1/sqrt(1 - v^2/c^2)."""
    beta = v_ms / c
    beta = min(abs(beta), 0.9999999)
    return 1.0 / np.sqrt(1.0 - beta * beta)
