"""5-Layer Probabilistic Interstellar Medium Model.

Implements Section 3 of the IMC Blueprint v2.0:
- Layer 1: Bulk Neutral Hydrogen (HI) from HI4PI survey parameters
- Layer 2: Warm Ionized Medium (WIM) from Reynolds Layer model
- Layer 3: Dust grain distribution (Draine & Li 2007)
- Layer 4: Magnetic field structure (Faraday rotation measures)
- Layer 5: Uncertainty envelope for all layers

Also implements the ablative shielding degradation model (Section 3.3).
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from ..constants import (c, M_H, eV, LY, PC, AU,
                         SHIELD_MATERIALS)


@dataclass
class ISMProfile:
    """Full probabilistic ISM profile along a trajectory corridor."""
    path_pc: np.ndarray         # Path positions [parsecs]
    nH_cm3: np.ndarray          # Neutral H density [cm^-3]
    nH_sigma: np.ndarray        # 1-sigma uncertainty on nH
    ne_cm3: np.ndarray          # Free electron density [cm^-3]
    ne_sigma: np.ndarray        # 1-sigma uncertainty on ne
    dust_opacity: np.ndarray    # Dust optical depth per parsec
    dust_sigma: np.ndarray      # 1-sigma uncertainty on dust
    B_field_uG: np.ndarray      # Magnetic field magnitude [microGauss]
    B_sigma: np.ndarray         # 1-sigma uncertainty on B
    CR_flux: np.ndarray         # Cosmic ray flux [GeV^-1 cm^-2 s^-1 sr^-1]
    CR_sigma: np.ndarray        # 1-sigma uncertainty on CR flux


class ISMEngine:
    """Multi-layer probabilistic ISM model.

    Based on published survey data and models:
    - HI4PI (HI Collaboration 2016): neutral hydrogen column density
    - NE2001 / YMW16: free electron density model
    - Draine & Li (2007): dust grain distribution
    - Van Eck et al. (2011): Faraday rotation measure compilations
    """

    # Local Bubble parameters (within ~100 pc of Sun)
    # Based on Lallement et al. (2003), Redfield & Linsky (2008)
    LOCAL_BUBBLE_nH_BASE = 0.005     # cm^-3 (very low inside bubble)
    LOCAL_FLUFF_nH = 0.10            # cm^-3 (Local Interstellar Cloud)
    WARM_ISM_nH = 0.30               # cm^-3 (beyond Local Bubble wall)

    # Local Bubble radius varies by direction, ~60-250 pc
    BUBBLE_RADIUS_PC = 100.0

    # WIM parameters (Reynolds 1991)
    WIM_ne_BASE = 0.02               # cm^-3 midplane
    WIM_SCALE_HEIGHT_PC = 900.0      # pc

    # Dust parameters (Draine & Li 2007)
    DUST_TAU_PER_PC = 1.8e-3         # optical depth per parsec (V-band, local)
    DUST_GRAIN_POWER_LAW = -3.5      # dN/da ~ a^(-3.5)

    # Magnetic field (local bubble)
    B_LOCAL_uG = 3.0                 # microGauss (typical local field)
    B_SIGMA_FRAC = 0.30              # 30% uncertainty

    # Cosmic ray flux at 1 GeV
    CR_FLUX_1GEV = 1.8e-9            # GeV^-1 cm^-2 s^-1 sr^-1

    def query(self, origin_ly: np.ndarray, dest_ly: np.ndarray,
              num_samples: int = 100) -> ISMProfile:
        """Query the ISM along a trajectory corridor.

        Parameters
        ----------
        origin_ly : array (3,) — origin position in light-years (typically [0,0,0] for Sol)
        dest_ly : array (3,) — destination position in light-years
        num_samples : int — number of sample points along the path

        Returns
        -------
        ISMProfile with all 5 layers populated
        """
        origin_pc = origin_ly / 3.2616
        dest_pc = dest_ly / 3.2616
        total_dist_pc = np.linalg.norm(dest_pc - origin_pc)
        direction = (dest_pc - origin_pc) / max(total_dist_pc, 1e-10)

        path_pc = np.linspace(0, total_dist_pc, num_samples)

        # Layer 1: Neutral Hydrogen (HI)
        nH, nH_sigma = self._layer_hi(path_pc, direction, total_dist_pc)

        # Layer 2: Warm Ionized Medium
        ne, ne_sigma = self._layer_wim(path_pc, direction)

        # Layer 3: Dust grain distribution
        dust, dust_sigma = self._layer_dust(path_pc, direction, total_dist_pc)

        # Layer 4: Magnetic field
        B, B_sigma = self._layer_bfield(path_pc, direction)

        # Layer 5: Cosmic ray flux
        CR, CR_sigma = self._layer_cosmic_rays(path_pc, direction)

        return ISMProfile(
            path_pc=path_pc,
            nH_cm3=nH, nH_sigma=nH_sigma,
            ne_cm3=ne, ne_sigma=ne_sigma,
            dust_opacity=dust, dust_sigma=dust_sigma,
            B_field_uG=B, B_sigma=B_sigma,
            CR_flux=CR, CR_sigma=CR_sigma,
        )

    def _layer_hi(self, path_pc: np.ndarray, direction: np.ndarray,
                  total_dist_pc: float) -> tuple:
        """Layer 1: Neutral hydrogen density profile from HI4PI-like model.

        Models the Local Bubble structure:
        - Very low density inside the bubble cavity
        - Local Interstellar Cloud (LIC) with moderate density
        - Transition to warm/hot ISM beyond the bubble wall
        """
        n = len(path_pc)
        nH = np.zeros(n)
        nH_sigma = np.zeros(n)

        for i, s in enumerate(path_pc):
            if s < 5.0:
                # Very local: Local Interstellar Cloud
                nH[i] = self.LOCAL_FLUFF_nH * (1.0 + 0.3 * np.sin(s * 0.5))
                nH_sigma[i] = nH[i] * 0.20
            elif s < self.BUBBLE_RADIUS_PC:
                # Inside Local Bubble: low density with filaments
                base = self.LOCAL_BUBBLE_nH_BASE
                # Model density enhancements (filaments, cloudlets)
                filament = 0.02 * np.exp(-((s - 30) / 10) ** 2)
                filament += 0.015 * np.exp(-((s - 70) / 15) ** 2)
                nH[i] = base + filament
                nH_sigma[i] = max(nH[i] * 0.40, 0.005)
            else:
                # Beyond bubble: transition to warm ISM
                wall_transition = np.tanh((s - self.BUBBLE_RADIUS_PC) / 20.0)
                nH[i] = self.LOCAL_BUBBLE_nH_BASE + \
                    (self.WARM_ISM_nH - self.LOCAL_BUBBLE_nH_BASE) * wall_transition
                nH_sigma[i] = nH[i] * 0.35

            # Add direction-dependent modulation (galactic plane is denser)
            gal_factor = 1.0 + 0.2 * abs(direction[2])
            nH[i] *= gal_factor

        return nH, nH_sigma

    def _layer_wim(self, path_pc: np.ndarray, direction: np.ndarray) -> tuple:
        """Layer 2: Warm Ionized Medium (free electron density).

        Based on NE2001/YMW16 electron density models.
        """
        n = len(path_pc)
        ne = np.zeros(n)
        ne_sigma = np.zeros(n)

        for i, s in enumerate(path_pc):
            # Base WIM density with exponential scale height
            z_pc = s * direction[2]  # height above galactic plane
            ne_base = self.WIM_ne_BASE * np.exp(-abs(z_pc) / self.WIM_SCALE_HEIGHT_PC)

            # Local enhancement from known HII regions
            if s < 5.0:
                ne[i] = ne_base * 0.5  # Local cavity has lower electron density
            else:
                ne[i] = ne_base

            ne_sigma[i] = ne[i] * 0.25

        return ne, ne_sigma

    def _layer_dust(self, path_pc: np.ndarray, direction: np.ndarray,
                    total_dist_pc: float) -> tuple:
        """Layer 3: Dust grain distribution (Draine & Li 2007).

        Grain size distribution: dN/da ~ a^(-3.5)
        Optical depth scales approximately linearly with distance
        but with local enhancements.
        """
        n = len(path_pc)
        dust = np.zeros(n)
        dust_sigma = np.zeros(n)

        for i, s in enumerate(path_pc):
            # Base dust opacity per parsec
            tau_base = self.DUST_TAU_PER_PC

            # Reduced in Local Bubble
            if s < self.BUBBLE_RADIUS_PC:
                reduction = 0.2 + 0.8 * (s / self.BUBBLE_RADIUS_PC)
                tau_base *= reduction

            # Cumulative optical depth to this point
            if i > 0:
                ds = path_pc[i] - path_pc[i - 1]
                dust[i] = dust[i - 1] + tau_base * ds
            else:
                dust[i] = 0.0

            dust_sigma[i] = dust[i] * 0.30 + 1e-6

        return dust, dust_sigma

    def _layer_bfield(self, path_pc: np.ndarray, direction: np.ndarray) -> tuple:
        """Layer 4: Magnetic field structure from Faraday rotation measures."""
        n = len(path_pc)
        B = np.zeros(n)
        B_sigma = np.zeros(n)

        for i, s in enumerate(path_pc):
            # Local magnetic field with coherence length ~10 pc
            B_coherent = self.B_LOCAL_uG * np.cos(s * 2 * np.pi / 50.0)
            B_random = self.B_LOCAL_uG * 0.5  # turbulent component

            B[i] = np.sqrt(B_coherent ** 2 + B_random ** 2)
            B_sigma[i] = B[i] * self.B_SIGMA_FRAC

        return B, B_sigma

    def _layer_cosmic_rays(self, path_pc: np.ndarray,
                           direction: np.ndarray) -> tuple:
        """Layer 5: Cosmic ray flux profile."""
        n = len(path_pc)
        CR = np.zeros(n)
        CR_sigma = np.zeros(n)

        for i, s in enumerate(path_pc):
            # CR flux modulated by heliosphere (reduced near Sun)
            # and galactic gradient
            if s < 0.01:  # within heliosphere
                CR[i] = self.CR_FLUX_1GEV * 0.3
            else:
                # Slight increase with distance from Sun due to
                # loss of heliospheric shielding
                CR[i] = self.CR_FLUX_1GEV * (1.0 + 0.05 * np.log(1 + s))
            CR_sigma[i] = CR[i] * 0.20

        return CR, CR_sigma


def compute_sputter_yield(E_ion_eV: float, material: str = "be") -> float:
    """Compute sputter yield using the Yamamura empirical model.

    Parameters
    ----------
    E_ion_eV : float — kinetic energy of impacting H atom [eV]
    material : str — shield material key ('be', 'w', 'c')

    Returns
    -------
    Ysp : float — sputter yield (atoms removed per incident ion)
    """
    mat = SHIELD_MATERIALS.get(material, SHIELD_MATERIALS["be"])
    U_s = mat["U_s_eV"]  # surface binding energy
    Z_t = mat["Z"]       # target atomic number
    A_t = mat["A"]       # target atomic mass

    # Threshold energy for sputtering
    E_th = U_s * (1 + 5.7 * (1.0 / A_t)) ** 2  # Simplified threshold

    if E_ion_eV < E_th:
        return 0.0

    # Yamamura formula (simplified for H -> heavy target)
    # Y = 0.042 * alpha * Sn(E) / U_s * [1 - (E_th/E)^(2/3)] * [1 - (E_th/E)]^2
    # alpha depends on mass ratio
    alpha = 0.2  # for H -> Be/W/C (light projectile)

    # Nuclear stopping cross-section (reduced energy formalism)
    epsilon = E_ion_eV * A_t / (Z_t * (1 + A_t) * 30.0)
    Sn = 3.441 * np.sqrt(epsilon) * np.log(epsilon + 2.718) / \
        (1 + 6.355 * np.sqrt(epsilon) + epsilon * (6.882 * np.sqrt(epsilon) - 1.708))

    ratio = E_th / E_ion_eV
    if ratio >= 1.0:
        return 0.0

    Ysp = 0.042 * alpha * Sn / U_s * (1.0 - ratio ** (2.0 / 3.0)) * (1.0 - ratio) ** 2
    return max(0.0, Ysp)


def compute_shield_degradation(velocity_profile_ms: np.ndarray,
                               nH_profile_cm3: np.ndarray,
                               path_m: np.ndarray,
                               material: str = "be",
                               initial_thickness_cm: float = 15.0) -> np.ndarray:
    """Compute shield thickness remaining along trajectory.

    Integrates the sputtering erosion rate along the full trajectory.

    Parameters
    ----------
    velocity_profile_ms : array — spacecraft velocity at each path point [m/s]
    nH_profile_cm3 : array — neutral H density at each point [cm^-3]
    path_m : array — path positions [m]
    material : str — shield material key
    initial_thickness_cm : float — initial shield thickness [cm]

    Returns
    -------
    shield_fraction : array — remaining shield fraction at each point
    """
    mat = SHIELD_MATERIALS.get(material, SHIELD_MATERIALS["be"])
    rho = mat["rho_gcc"]  # g/cm^3
    A_t = mat["A"]
    m_atom_g = A_t * 1.66054e-24  # g

    n_points = len(path_m)
    eroded_cm = np.zeros(n_points)

    for i in range(1, n_points):
        v = velocity_profile_ms[i]
        nH = nH_profile_cm3[i] * 1e6  # cm^-3 -> m^-3 (nH is already in cm^-3)

        # Ion energy in ship frame
        E_ion_J = 0.5 * M_H * v * v
        E_ion_eV = E_ion_J / eV

        # Sputter yield
        Ysp = compute_sputter_yield(E_ion_eV, material)

        # Erosion rate: dE/ds = nH * Ysp * m_atom / rho
        # nH in cm^-3, we need consistent units
        # erosion in cm per cm of path
        erosion_rate_cm_per_cm = nH_profile_cm3[i] * Ysp * m_atom_g / rho

        # Path increment
        ds_m = path_m[i] - path_m[i - 1]
        ds_cm = ds_m * 100.0  # m -> cm

        eroded_cm[i] = eroded_cm[i - 1] + erosion_rate_cm_per_cm * ds_cm

    shield_fraction = np.maximum(0.0, 1.0 - eroded_cm / initial_thickness_cm)
    return shield_fraction
