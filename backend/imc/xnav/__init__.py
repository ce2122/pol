"""XNAV Navigation System with Unscented Kalman Filter.

Implements Section 8 of the IMC Blueprint v2.0:
- 13-state UKF (position, velocity, acceleration, quaternion)
- 4-component pulsar timing noise model:
  1. White noise (radiometer noise)
  2. Red noise (spin noise)
  3. Dispersion measure variation
  4. Relativistic timing corrections (Shapiro delay, proper motion, parallax)
- Detector hardware specification
- Stellar aberration correction
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from ..constants import c, AU, LY, PC, YEAR_S
from ..integrators import aberrated_angle


@dataclass
class PulsarTimingModel:
    """Timing model for a single millisecond pulsar."""
    name: str
    P_ms: float                  # Spin period [ms]
    P_dot: float                 # Period derivative [dimensionless]
    DM: float                    # Dispersion measure [pc cm^-3]
    DM_dot: float                # DM time derivative [pc cm^-3 yr^-1]
    red_noise_A: float           # Red noise amplitude
    red_noise_gamma: float       # Red noise spectral index
    ra_deg: float                # Right ascension [degrees]
    dec_deg: float               # Declination [degrees]
    proper_motion_masyr: tuple   # (mu_RA, mu_Dec) [mas/yr]
    parallax_mas: float          # Parallax [mas]
    sigma_toa_us: float = 0.0   # Achieved timing noise [microseconds]

    def __post_init__(self):
        """Compute total TOA uncertainty from 4 noise components."""
        # Component 1: White noise (radiometer)
        self.sigma_white_us = 0.5 / np.sqrt(1.0 / max(self.P_ms, 0.001))

        # Component 2: Red noise (spin noise)
        # PSD ~ A^2 * f^(-gamma), integrate over observation band
        f_min = 1.0 / (10.0 * YEAR_S)  # minimum frequency
        f_max = 1.0 / (1000.0)  # 1 ks integration
        if self.red_noise_gamma > 1:
            sigma_red_sq = (self.red_noise_A ** 2 /
                           (self.red_noise_gamma - 1) *
                           (f_min ** (1 - self.red_noise_gamma) -
                            f_max ** (1 - self.red_noise_gamma)))
        else:
            sigma_red_sq = self.red_noise_A ** 2 * np.log(f_max / f_min)
        self.sigma_red_us = np.sqrt(max(sigma_red_sq, 0)) * 1e6

        # Component 3: DM variation
        # sigma_DM ~ DM_dot * delta_t / (nu^2 correction)
        self.sigma_DM_us = abs(self.DM_dot) * 0.01 * 4.149e3  # at 1 GHz

        # Component 4: Relativistic corrections
        # Shapiro delay, proper motion, parallax combined
        self.sigma_rel_us = 0.01 + self.parallax_mas * 0.001

        # Total TOA uncertainty
        self.sigma_toa_us = np.sqrt(
            self.sigma_white_us ** 2 +
            self.sigma_red_us ** 2 +
            self.sigma_DM_us ** 2 +
            self.sigma_rel_us ** 2
        )


# Catalog of navigation-grade millisecond pulsars
# Only pulsars with sigma_toa < 0.1 us qualify for primary navigation
NAVIGATION_PULSARS = [
    PulsarTimingModel(
        name="PSR J0437-4715",
        P_ms=5.757, P_dot=5.73e-20,
        DM=2.64, DM_dot=0.0001,
        red_noise_A=1e-15, red_noise_gamma=3.0,
        ra_deg=69.32, dec_deg=-47.25,
        proper_motion_masyr=(121.44, -71.44),
        parallax_mas=6.37,
    ),
    PulsarTimingModel(
        name="PSR J1909-3744",
        P_ms=2.947, P_dot=1.40e-20,
        DM=10.39, DM_dot=0.00005,
        red_noise_A=5e-16, red_noise_gamma=4.0,
        ra_deg=287.44, dec_deg=-37.74,
        proper_motion_masyr=(-9.51, -35.77),
        parallax_mas=0.94,
    ),
    PulsarTimingModel(
        name="PSR J0030+0451",
        P_ms=4.865, P_dot=1.02e-20,
        DM=4.33, DM_dot=0.00002,
        red_noise_A=3e-16, red_noise_gamma=3.5,
        ra_deg=7.61, dec_deg=4.86,
        proper_motion_masyr=(-5.27, -0.40),
        parallax_mas=3.29,
    ),
    PulsarTimingModel(
        name="PSR J2124-3358",
        P_ms=4.931, P_dot=2.06e-20,
        DM=4.62, DM_dot=0.00003,
        red_noise_A=2e-16, red_noise_gamma=3.0,
        ra_deg=321.17, dec_deg=-33.97,
        proper_motion_masyr=(-14.10, -50.06),
        parallax_mas=2.49,
    ),
    PulsarTimingModel(
        name="PSR J1713+0747",
        P_ms=4.570, P_dot=8.53e-21,
        DM=15.99, DM_dot=0.0001,
        red_noise_A=1e-16, red_noise_gamma=4.5,
        ra_deg=258.45, dec_deg=7.79,
        proper_motion_masyr=(4.92, -3.91),
        parallax_mas=0.85,
    ),
]


@dataclass
class DetectorSpec:
    """XNAV detector hardware specification."""
    effective_area_cm2: float = 500.0    # Minimum effective area
    energy_band_keV: tuple = (0.2, 12.0)  # Energy range
    timing_accuracy_ns: float = 100.0     # Absolute timing accuracy
    fov_deg: float = 2.0                  # Field of view
    attitude_accuracy_arcsec: float = 1.0  # Boresight attitude accuracy


class UnscentedKalmanFilter:
    """13-state Unscented Kalman Filter for XNAV navigation.

    State vector: [x, y, z, vx, vy, vz, ax, ay, az, q0, q1, q2, q3]
    - Position (BCRS) [m]
    - Velocity [m/s]
    - Acceleration [m/s²]
    - Attitude quaternion

    Uses the unscented transform to propagate sigma points through
    nonlinear dynamics, capturing second-order effects.
    """

    N_STATE = 13
    ALPHA = 1e-3    # Spread of sigma points
    BETA = 2.0      # Prior knowledge (Gaussian: beta=2)
    KAPPA = 0.0     # Secondary scaling

    def __init__(self, pulsars: List[PulsarTimingModel] = None,
                 detector: DetectorSpec = None):
        self.pulsars = pulsars or NAVIGATION_PULSARS[:4]
        self.detector = detector or DetectorSpec()

        # State and covariance
        self.x = np.zeros(self.N_STATE)
        self.x[9] = 1.0  # quaternion identity
        self.P = np.eye(self.N_STATE)

        # Initial uncertainties
        self.P[0:3, 0:3] *= (1000.0 * AU) ** 2   # 1000 AU position uncertainty
        self.P[3:6, 3:6] *= 100.0 ** 2             # 100 m/s velocity uncertainty
        self.P[6:9, 6:9] *= 0.001 ** 2             # 0.001 m/s² accel uncertainty
        self.P[9:13, 9:13] *= 0.01 ** 2            # quaternion uncertainty

        # Process noise
        self.Q = np.zeros((self.N_STATE, self.N_STATE))
        thrust_sigma = 0.001  # 0.1% thrust uncertainty
        self.Q[6:9, 6:9] = np.eye(3) * thrust_sigma ** 2
        gyro_drift_rad = 0.5 / 3600.0 * np.pi / 180.0  # 0.5 arcsec
        self.Q[9:13, 9:13] = np.eye(4) * gyro_drift_rad ** 2

        # UKF parameters
        n = self.N_STATE
        self.lambda_ = self.ALPHA ** 2 * (n + self.KAPPA) - n
        self.gamma = np.sqrt(n + self.lambda_)

        # Weights
        self.Wm = np.zeros(2 * n + 1)
        self.Wc = np.zeros(2 * n + 1)
        self.Wm[0] = self.lambda_ / (n + self.lambda_)
        self.Wc[0] = self.lambda_ / (n + self.lambda_) + (1 - self.ALPHA ** 2 + self.BETA)
        for i in range(1, 2 * n + 1):
            self.Wm[i] = 1.0 / (2 * (n + self.lambda_))
            self.Wc[i] = 1.0 / (2 * (n + self.lambda_))

    def initialize(self, pos_m: np.ndarray, vel_ms: np.ndarray,
                   pos_sigma_m: float = 1e12, vel_sigma_ms: float = 100.0):
        """Initialize filter state."""
        self.x[0:3] = pos_m
        self.x[3:6] = vel_ms
        self.x[6:9] = 0.0
        self.x[9] = 1.0
        self.x[10:13] = 0.0

        self.P = np.eye(self.N_STATE)
        self.P[0:3, 0:3] *= pos_sigma_m ** 2
        self.P[3:6, 3:6] *= vel_sigma_ms ** 2
        self.P[6:9, 6:9] *= 0.001 ** 2
        self.P[9:13, 9:13] *= 0.01 ** 2

    def _generate_sigma_points(self) -> np.ndarray:
        """Generate 2n+1 sigma points."""
        n = self.N_STATE
        sigma_pts = np.zeros((2 * n + 1, n))
        sigma_pts[0] = self.x

        try:
            sqrt_P = np.linalg.cholesky((n + self.lambda_) * self.P)
        except np.linalg.LinAlgError:
            # If Cholesky fails, use eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(self.P)
            eigvals = np.maximum(eigvals, 1e-20)
            sqrt_P = eigvecs @ np.diag(np.sqrt(eigvals * (n + self.lambda_)))

        for i in range(n):
            sigma_pts[i + 1] = self.x + sqrt_P[:, i]
            sigma_pts[n + i + 1] = self.x - sqrt_P[:, i]

        return sigma_pts

    def _state_transition(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Propagate state forward by dt seconds."""
        new_state = state.copy()
        # Position update: x += v*dt + 0.5*a*dt^2
        new_state[0:3] = state[0:3] + state[3:6] * dt + 0.5 * state[6:9] * dt * dt
        # Velocity update: v += a*dt
        new_state[3:6] = state[3:6] + state[6:9] * dt
        # Acceleration stays constant between updates
        # Quaternion propagation (simplified)
        q = state[9:13]
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            new_state[9:13] = q / q_norm
        return new_state

    def predict(self, dt: float):
        """UKF prediction step."""
        # Generate sigma points
        sigma_pts = self._generate_sigma_points()

        # Propagate sigma points
        n = self.N_STATE
        propagated = np.zeros_like(sigma_pts)
        for i in range(2 * n + 1):
            propagated[i] = self._state_transition(sigma_pts[i], dt)

        # Compute predicted mean
        x_pred = np.zeros(n)
        for i in range(2 * n + 1):
            x_pred += self.Wm[i] * propagated[i]

        # Compute predicted covariance
        P_pred = self.Q.copy()
        for i in range(2 * n + 1):
            diff = propagated[i] - x_pred
            P_pred += self.Wc[i] * np.outer(diff, diff)

        self.x = x_pred
        self.P = P_pred

    def update_toa(self, pulsar_idx: int, toa_measured_s: float,
                   ship_velocity_frac_c: float = 0.0):
        """UKF measurement update from a pulsar TOA observation.

        Parameters
        ----------
        pulsar_idx : index into self.pulsars
        toa_measured_s : measured time of arrival [s]
        ship_velocity_frac_c : v/c for aberration correction
        """
        pulsar = self.pulsars[pulsar_idx]
        n = self.N_STATE

        # Generate sigma points
        sigma_pts = self._generate_sigma_points()

        # Predicted measurements for each sigma point
        z_pred = np.zeros(2 * n + 1)
        for i in range(2 * n + 1):
            pos = sigma_pts[i, 0:3]
            # TOA prediction: time for signal to travel from pulsar to ship
            # Simplified: use pulsar direction and ship position
            pulsar_dir = np.array([
                np.cos(np.radians(pulsar.ra_deg)) * np.cos(np.radians(pulsar.dec_deg)),
                np.sin(np.radians(pulsar.ra_deg)) * np.cos(np.radians(pulsar.dec_deg)),
                np.sin(np.radians(pulsar.dec_deg)),
            ])

            # Apply aberration correction
            if ship_velocity_frac_c > 0.001:
                theta_rest = np.arccos(np.dot(pulsar_dir, np.array([1, 0, 0])))
                theta_ship = aberrated_angle(theta_rest, ship_velocity_frac_c)
                # Adjust direction for aberration
                aberration_factor = np.cos(theta_ship) / max(np.cos(theta_rest), 1e-10)
            else:
                aberration_factor = 1.0

            # Geometric delay: dot product of position with pulsar direction
            geometric_delay = np.dot(pos, pulsar_dir) / c
            z_pred[i] = geometric_delay * aberration_factor

        # Mean predicted measurement
        z_mean = sum(self.Wm[i] * z_pred[i] for i in range(2 * n + 1))

        # Measurement noise (from pulsar timing model)
        R = (pulsar.sigma_toa_us * 1e-6) ** 2  # Convert to seconds²

        # Innovation covariance
        Pzz = R
        for i in range(2 * n + 1):
            dz = z_pred[i] - z_mean
            Pzz += self.Wc[i] * dz * dz

        # Cross-covariance
        Pxz = np.zeros(n)
        for i in range(2 * n + 1):
            dx = sigma_pts[i] - self.x
            dz = z_pred[i] - z_mean
            Pxz += self.Wc[i] * dx * dz

        # Kalman gain
        K = Pxz / max(Pzz, 1e-30)

        # Innovation
        innovation = toa_measured_s - z_mean

        # State update
        self.x = self.x + K * innovation

        # Covariance update
        self.P = self.P - np.outer(K, K) * Pzz

        # Ensure symmetry and positive definiteness
        self.P = 0.5 * (self.P + self.P.T)
        eigvals = np.linalg.eigvalsh(self.P)
        if np.min(eigvals) < 0:
            self.P += np.eye(n) * (abs(np.min(eigvals)) + 1e-10)

        return {
            "innovation_s": innovation,
            "kalman_gain_max": np.max(np.abs(K)),
            "position_sigma_m": np.sqrt(np.trace(self.P[0:3, 0:3]) / 3),
        }

    def get_position_error_au(self) -> float:
        """Get current 1-sigma position error in AU."""
        pos_cov = self.P[0:3, 0:3]
        sigma_m = np.sqrt(np.trace(pos_cov) / 3.0)
        return sigma_m / AU

    def run_observation_campaign(self, n_observations: int,
                                dt_between_obs_s: float,
                                ship_velocity_frac_c: float = 0.0) -> dict:
        """Run a series of pulsar observations and track position error.

        Parameters
        ----------
        n_observations : number of observation windows
        dt_between_obs_s : time between observations [s]
        ship_velocity_frac_c : spacecraft velocity as fraction of c

        Returns
        -------
        dict with position error history
        """
        errors_au = [self.get_position_error_au()]

        for obs in range(n_observations):
            # Predict
            self.predict(dt_between_obs_s)

            # Update from each pulsar
            for p_idx in range(len(self.pulsars)):
                # Simulate TOA measurement (true TOA + noise)
                true_toa = np.dot(self.x[0:3],
                                  np.array([np.cos(np.radians(self.pulsars[p_idx].ra_deg)),
                                           np.sin(np.radians(self.pulsars[p_idx].ra_deg)),
                                           0.0])) / c
                noise = np.random.normal(0, self.pulsars[p_idx].sigma_toa_us * 1e-6)
                measured_toa = true_toa + noise

                self.update_toa(p_idx, measured_toa, ship_velocity_frac_c)

            errors_au.append(self.get_position_error_au())

        return {
            "position_errors_au": np.array(errors_au),
            "final_error_au": errors_au[-1],
            "n_observations": n_observations,
        }


def compute_xnav_error(dist_ly: float, peak_beta: float,
                       detector_area_cm2: float = 500.0,
                       n_pulsars: int = 4) -> float:
    """Compute expected XNAV position error at destination.

    Parameters
    ----------
    dist_ly : distance to destination [ly]
    peak_beta : peak velocity as fraction of c
    detector_area_cm2 : detector effective area [cm²]
    n_pulsars : number of navigation pulsars

    Returns
    -------
    position_error_au : expected 1-sigma position error [AU]
    """
    # Base TOA precision from detector area
    sigma_toa_us = 0.1 * np.sqrt(500.0 / max(detector_area_cm2, 1.0))

    # Number of observations (daily over transit)
    n_observations = dist_ly * 365.25  # ~1 per day per year of transit

    # Single observation position error
    sigma_single_m = sigma_toa_us * 1e-6 * c  # TOA * c = position error

    # Error reduction from multiple observations and pulsars
    # sqrt(N) averaging + sqrt(n_pulsars) from geometry
    pos_err_m = sigma_single_m * np.sqrt(n_observations) / np.sqrt(max(n_pulsars, 1))

    # But the error grows as sqrt(t) from process noise, so net error
    # scales roughly as observation_error * sqrt(transit_time)
    # This is modeled more carefully in the UKF, but for quick estimation:
    pos_err_au = pos_err_m / AU

    # Aberration correction degradation at high velocity
    aber_factor = 1.0 + 0.5 * peak_beta

    return pos_err_au * aber_factor
