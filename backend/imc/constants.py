"""Physical constants used throughout the IMC engine (SI units)."""
import numpy as np

# Fundamental constants
c = 299_792_458.0            # speed of light [m/s]
G = 6.67430e-11              # gravitational constant [m^3 kg^-1 s^-2]
g0 = 9.80665                 # standard gravity [m/s^2]
AU = 1.495978707e11          # astronomical unit [m]
LY = 9.4607304725808e15      # light-year [m]
PC = 3.0856775814913673e16   # parsec [m]
M_H = 1.67262192e-27         # hydrogen atom mass [kg]
M_SUN = 1.98892e30           # solar mass [kg]
eV = 1.602176634e-19         # electron-volt [J]
MeV = 1.602176634e-13        # mega-electron-volt [J]
AMU = 1.66053906660e-27      # atomic mass unit [kg]
k_B = 1.380649e-23           # Boltzmann constant [J/K]
h_planck = 6.62607015e-34    # Planck constant [J s]
YEAR_S = 365.25 * 86400.0    # Julian year [s]

# Solar system
R_SUN = 6.9634e8             # solar radius [m]
L_SUN = 3.828e26             # solar luminosity [W]
GM_SUN = G * M_SUN           # solar gravitational parameter [m^3/s^2]

# Mercury orbital elements
MERCURY_a = 5.790905e10      # semi-major axis [m]
MERCURY_e = 0.20563          # eccentricity
MERCURY_T = 87.9691 * 86400  # orbital period [s]
MERCURY_M = 3.3011e23        # mass [kg]

# Voyager 1
VOYAGER1_V_INF = 16_950.0    # hyperbolic excess velocity [m/s] (post all flybys)
VOYAGER1_LAUNCH_JD = 2443387.5  # 1977-09-05 Julian Date
VOYAGER1_CURRENT_R_AU = 163.0   # approximate current heliocentric distance [AU] (2024)

# Pioneer 10
PIONEER_MASS = 259.0         # kg
PIONEER_RTG_POWER = 63.5     # thermal power [W] at anomaly epoch
PIONEER_ANOM_ACCEL = 8.74e-10  # observed anomalous acceleration [m/s^2]

# New Horizons
NH_C3 = 157.0                # characteristic energy [km^2/s^2]
NH_LAUNCH_JD = 2453755.5     # 2006-01-19
NH_PLUTO_CA_JD = 2457218.0   # 2015-07-14
NH_PLUTO_R_AU = 32.9         # Pluto heliocentric distance at encounter [AU]

# Alpha Centauri AB binary
ALPHA_CEN_PERIOD_YR = 79.91
ALPHA_CEN_a_AU = 23.4
ALPHA_CEN_e = 0.5179
ALPHA_CEN_MA = 1.100 * M_SUN
ALPHA_CEN_MB = 0.907 * M_SUN

# Shield material properties
SHIELD_MATERIALS = {
    "be": {"name": "Beryllium", "rho_gcc": 1.85, "A": 9.012, "Z": 4,
           "U_s_eV": 3.32, "rho_kgm3": 1850.0},
    "w":  {"name": "Tungsten",  "rho_gcc": 19.3, "A": 183.84, "Z": 74,
           "U_s_eV": 8.90, "rho_kgm3": 19300.0},
    "c":  {"name": "Carbon Composite", "rho_gcc": 1.50, "A": 12.011, "Z": 6,
           "U_s_eV": 7.41, "rho_kgm3": 1500.0},
}
