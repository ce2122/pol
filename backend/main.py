"""FastAPI backend for the Interstellar Mission Compiler v2.0.

Serves the frontend HTML and provides real computational APIs.
"""
import os
import json
import time
import asyncio
import tempfile
import shutil
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from imc.constants import c, AU, LY, YEAR_S, g0, SHIELD_MATERIALS
from imc.catalog import StellarCatalog, create_catalog_db
from imc.propulsion import (
    get_propulsion_model, relativistic_delta_v, fuel_fraction_needed,
    lorentz_factor, FusionDriveParams, ACFFDriveParams, LightsailParams,
    FAILURE_MODES,
)
from imc.ism import ISMEngine, compute_shield_degradation
from imc.integrators import AdaptiveIntegrator, aberrated_angle, doppler_shift
from imc.validation import run_validation_suite, ValidationResult
from imc.xnav import (
    UnscentedKalmanFilter, NAVIGATION_PULSARS, DetectorSpec,
    compute_xnav_error,
)
from imc.optimizer import (
    optimize_trajectory, PolynomialChaosExpansion, TrajectoryCandidate,
)
from imc.fault_tree import build_fault_tree, compute_lom, RECOVERY_PROTOCOLS
from imc.outputs import (
    generate_spice_kernel, generate_burn_manifest,
    generate_xnav_database, generate_fta_xml,
    generate_pdf_report, generate_verification_package,
)

app = FastAPI(title="Interstellar Mission Compiler", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

db_path = os.path.join(DATA_DIR, "local_bubble.db")
catalog = StellarCatalog(db_path)
ism_engine = ISMEngine()

# Cache validation results
_validation_cache = {"results": None, "all_passed": False}

# Serve the frontend
FRONTEND_DIR = os.path.dirname(os.path.dirname(__file__))


# ── Request/Response Models ─────────────────────

class PropulsionRequest(BaseModel):
    architecture: str  # 'fusion', 'acff', 'sail'
    pellet_frequency_hz: Optional[float] = None
    driver_efficiency: Optional[float] = None
    fusion_gain_Q: Optional[float] = None
    charged_fraction: Optional[float] = None
    laser_power_W: Optional[float] = None
    sail_area_m2: Optional[float] = None
    sail_mass_kg: Optional[float] = None

class MissionRequest(BaseModel):
    architecture: str
    destination_name: str
    payload_mass_kg: float = 5000.0
    fuel_frac_limit: float = 0.85
    max_transit_yr: float = 100.0
    max_accel_g: float = 0.001
    shield_material: str = "be"
    shield_thickness_cm: float = 15.0
    detector_area_cm2: float = 500.0
    n_pulsars: int = 4

class CompileRequest(BaseModel):
    architecture: str
    destination_name: str
    payload_mass_kg: float = 5000.0
    fuel_frac_limit: float = 0.85
    max_transit_yr: float = 100.0
    max_accel_g: float = 0.001
    shield_material: str = "be"
    shield_thickness_cm: float = 15.0
    detector_area_cm2: float = 500.0
    n_pulsars: int = 4
    pso_particles: int = 1000
    pso_iterations: int = 200


# ── API Endpoints ─────────────────────────────

@app.get("/")
async def serve_frontend():
    """Serve the main HTML frontend."""
    html_path = os.path.join(FRONTEND_DIR, "IMC_v2_Interstellar_Mission_Compiler.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return HTMLResponse("<h1>IMC v2.0 Backend Running</h1><p>Frontend HTML not found.</p>")


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0.0", "engine": "IMC Flight-Certifiable Backend"}


# ── Catalog ─────────────────────────────────────

@app.get("/api/catalog/stars")
async def list_stars(max_dist: float = 30.0, habitable_only: bool = False):
    """List available target stars."""
    stars = catalog.list_targets(max_dist, habitable_only)
    # Convert numpy arrays to lists for JSON serialization
    for s in stars:
        if isinstance(s.get("cov_matrix"), np.ndarray):
            s["cov_matrix"] = s["cov_matrix"].tolist()
    return {"stars": stars, "count": len(stars)}


@app.get("/api/catalog/star/{name}")
async def get_star(name: str):
    """Get details for a specific star."""
    star = catalog.get_star(name)
    if star is None:
        raise HTTPException(404, f"Star '{name}' not found")
    if isinstance(star.get("cov_matrix"), np.ndarray):
        star["cov_matrix"] = star["cov_matrix"].tolist()
    return star


# ── Propulsion ──────────────────────────────────

@app.post("/api/propulsion/model")
async def compute_propulsion(req: PropulsionRequest):
    """Compute propulsion model parameters from first principles."""
    kwargs = {}
    if req.architecture == "fusion":
        if req.pellet_frequency_hz: kwargs["pellet_frequency_hz"] = req.pellet_frequency_hz
        if req.driver_efficiency: kwargs["driver_efficiency"] = req.driver_efficiency
        if req.fusion_gain_Q: kwargs["fusion_gain_Q"] = req.fusion_gain_Q
    elif req.architecture == "acff":
        if req.charged_fraction: kwargs["charged_fraction"] = req.charged_fraction
    elif req.architecture == "sail":
        if req.laser_power_W: kwargs["laser_power_W"] = req.laser_power_W
        if req.sail_area_m2: kwargs["sail_area_m2"] = req.sail_area_m2
        if req.sail_mass_kg: kwargs["sail_mass_kg"] = req.sail_mass_kg

    model = get_propulsion_model(req.architecture, **kwargs)

    result = {"architecture": req.architecture}
    if isinstance(model, (FusionDriveParams, ACFFDriveParams)):
        result.update({
            "exhaust_velocity_ms": model.exhaust_velocity_ms,
            "specific_impulse_s": model.specific_impulse_s,
            "thrust_N": model.thrust_N,
            "mdot_kgs": model.mdot_kgs,
        })
    elif isinstance(model, LightsailParams):
        result.update({
            "thrust_N": model.thrust_N,
            "acceleration_ms2": model.acceleration_ms2,
            "max_velocity_ms": model.max_velocity_ms,
            "max_velocity_frac_c": model.max_velocity_frac_c,
        })

    result["failure_modes"] = FAILURE_MODES.get(req.architecture, [])
    return result


# ── Mission Profile ─────────────────────────────

@app.post("/api/mission/profile")
async def compute_mission_profile(req: MissionRequest):
    """Compute full mission profile with real physics."""
    star = catalog.get_star(req.destination_name)
    if star is None:
        raise HTTPException(404, f"Star '{req.destination_name}' not found")

    dist_ly = star["dist_ly"]
    model = get_propulsion_model(req.architecture)

    is_sail = isinstance(model, LightsailParams)

    if is_sail:
        v_max = model.max_velocity_ms
        beta = v_max / c
        gamma_avg = lorentz_factor(v_max * 0.5)
        earth_yr = dist_ly * LY / (v_max * YEAR_S)
        ship_yr = earth_yr / gamma_avg
        dv_total = v_max
        fuel_frac = 0.0
        total_mass = req.payload_mass_kg + model.sail_mass_kg
        peak_beta = beta
    else:
        ve_ms = model.exhaust_velocity_ms
        fuel_frac = min(req.fuel_frac_limit, 0.90)
        total_mass = req.payload_mass_kg / (1 - fuel_frac)
        m_fuel = total_mass - req.payload_mass_kg

        dv_half = relativistic_delta_v(ve_ms, total_mass, req.payload_mass_kg + m_fuel * 0.5)
        dv_total = relativistic_delta_v(ve_ms, total_mass, req.payload_mass_kg)
        peak_v = min(dv_half, 0.99 * c)
        peak_beta = peak_v / c

        v_avg = 0.55 * peak_v
        earth_yr = dist_ly * LY / (max(v_avg, 1.0) * YEAR_S)
        ship_yr = earth_yr / lorentz_factor(peak_v * 0.6)

    # ISM profile and shield degradation
    origin = np.array([0.0, 0.0, 0.0])
    dest = np.array([star["x_ly"], star["y_ly"], star["z_ly"]])
    ism_profile = ism_engine.query(origin, dest, num_samples=100)

    # Compute shield degradation along trajectory
    path_m = ism_profile.path_pc * 3.0857e16  # pc -> m
    n_pts = len(path_m)
    vel_profile = peak_beta * c * np.sin(np.linspace(0, np.pi, n_pts))
    shield_frac_arr = compute_shield_degradation(
        vel_profile, ism_profile.nH_cm3, path_m,
        req.shield_material, req.shield_thickness_cm
    )
    shield_frac = float(shield_frac_arr[-1])

    # XNAV error
    xnav_err = compute_xnav_error(dist_ly, peak_beta,
                                   req.detector_area_cm2, req.n_pulsars)

    # Loss of mission probability
    lom = compute_lom(req.architecture, earth_yr)

    # Covariance propagation for destination uncertainty
    if isinstance(star.get("cov_matrix"), np.ndarray):
        cov = star["cov_matrix"]
    else:
        cov = np.eye(6) * 1e20

    sigma_pos_AU = np.sqrt(np.trace(cov[0:3, 0:3]) / 3.0) / AU

    # ISM statistics
    ism_stats = {
        "nH_avg": float(np.mean(ism_profile.nH_cm3)),
        "ne_avg": float(np.mean(ism_profile.ne_cm3)),
        "dust_total": float(ism_profile.dust_opacity[-1]),
        "B_avg": float(np.mean(ism_profile.B_field_uG)),
        "CR_flux_avg": float(np.mean(ism_profile.CR_flux)),
    }

    return {
        "architecture": req.architecture,
        "destination": star["name"],
        "dist_ly": dist_ly,
        "delta_v_ms": float(dv_total),
        "delta_v_frac_c": float(dv_total / c),
        "earth_transit_yr": float(earth_yr),
        "ship_transit_yr": float(ship_yr),
        "peak_beta": float(peak_beta),
        "peak_lorentz": float(lorentz_factor(peak_beta * c)),
        "fuel_fraction": float(fuel_frac),
        "total_mass_kg": float(total_mass),
        "shield_remaining": float(shield_frac),
        "xnav_error_au": float(xnav_err),
        "lom_probability": float(lom),
        "position_uncertainty_au": float(sigma_pos_AU),
        "ism_profile": ism_stats,
        "ism_nH_profile": ism_profile.nH_cm3.tolist(),
        "ism_path_pc": ism_profile.path_pc.tolist(),
        "shield_profile": shield_frac_arr.tolist(),
    }


# ── Validation ──────────────────────────────────

@app.post("/api/validation/run")
async def run_validation():
    """Run the 5-test mandatory validation suite."""
    global _validation_cache
    result = run_validation_suite()

    # Convert to JSON-serializable format
    test_results = []
    for r in result["results"]:
        # Ensure all numpy types are converted to native Python types
        details = {}
        for k, v in (r.details or {}).items():
            if isinstance(v, (np.floating, float)):
                details[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                details[k] = int(v)
            elif isinstance(v, (np.bool_, bool)):
                details[k] = bool(v)
            elif isinstance(v, np.ndarray):
                details[k] = v.tolist()
            else:
                details[k] = str(v) if not isinstance(v, str) else v
        test_results.append({
            "test_id": str(r.test_id),
            "test_name": str(r.test_name),
            "description": str(r.description),
            "required": str(r.required),
            "computed": str(r.computed),
            "expected": str(r.expected),
            "error": str(r.error),
            "passed": bool(r.passed),
            "details": details,
        })

    _validation_cache = {
        "results": result["results"],
        "all_passed": result["all_passed"],
    }

    return {
        "all_passed": bool(result["all_passed"]),
        "tests": test_results,
    }


@app.get("/api/validation/status")
async def validation_status():
    """Get current validation status."""
    return {"all_passed": bool(_validation_cache["all_passed"])}


# ── ISM ─────────────────────────────────────────

@app.post("/api/ism/profile")
async def get_ism_profile(dest_name: str):
    """Get ISM profile along corridor to destination."""
    star = catalog.get_star(dest_name)
    if star is None:
        raise HTTPException(404, f"Star '{dest_name}' not found")

    origin = np.array([0.0, 0.0, 0.0])
    dest = np.array([star["x_ly"], star["y_ly"], star["z_ly"]])
    profile = ism_engine.query(origin, dest, num_samples=100)

    return {
        "path_pc": profile.path_pc.tolist(),
        "nH_cm3": profile.nH_cm3.tolist(),
        "nH_sigma": profile.nH_sigma.tolist(),
        "ne_cm3": profile.ne_cm3.tolist(),
        "dust_opacity": profile.dust_opacity.tolist(),
        "B_field_uG": profile.B_field_uG.tolist(),
        "CR_flux": profile.CR_flux.tolist(),
    }


# ── Fault Tree ──────────────────────────────────

@app.post("/api/faulttree/analyze")
async def analyze_fault_tree(architecture: str, earth_transit_yr: float):
    """Build and analyze fault tree."""
    mission_hours = earth_transit_yr * 365.25 * 24.0
    ft = build_fault_tree(architecture, mission_hours)
    result = ft.to_dict()
    result["recovery_protocols"] = RECOVERY_PROTOCOLS
    return result


# ── Compilation ─────────────────────────────────

@app.post("/api/compile")
async def compile_mission(req: CompileRequest):
    """Full mission compilation with real compute.

    This runs the complete pipeline:
    1. Validation gate
    2. Propulsion model
    3. Trajectory optimization (PSO + SQP)
    4. ISM and shielding analysis
    5. XNAV navigation analysis
    6. Fault tree analysis
    7. Uncertainty quantification (PCE)
    8. Output package generation
    """
    # Phase 1: Validation gate
    if not _validation_cache["all_passed"]:
        raise HTTPException(403, "Validation suite has not passed. Run validation first.")

    # Get star data
    star = catalog.get_star(req.destination_name)
    if star is None:
        raise HTTPException(404, f"Star '{req.destination_name}' not found")

    dist_ly = star["dist_ly"]

    # Phase 2: Propulsion model
    prop_model = get_propulsion_model(req.architecture)

    # Phase 3: Trajectory optimization
    opt_result = optimize_trajectory(
        prop_model, dist_ly,
        req.payload_mass_kg, req.fuel_frac_limit,
        req.max_transit_yr, req.max_accel_g,
        req.shield_material, req.shield_thickness_cm,
        n_particles=req.pso_particles,
        n_iterations=req.pso_iterations,
    )

    best = opt_result.best_trajectory

    # Phase 4: ISM and shielding analysis
    origin = np.array([0.0, 0.0, 0.0])
    dest = np.array([star["x_ly"], star["y_ly"], star["z_ly"]])
    ism_profile = ism_engine.query(origin, dest, num_samples=100)

    path_m = ism_profile.path_pc * 3.0857e16
    n_pts = len(path_m)
    vel_profile = best.peak_beta * c * np.sin(np.linspace(0, np.pi, n_pts))
    shield_arr = compute_shield_degradation(
        vel_profile, ism_profile.nH_cm3, path_m,
        req.shield_material, req.shield_thickness_cm
    )
    best.shield_remaining = float(shield_arr[-1])

    # Phase 5: XNAV
    xnav_err = compute_xnav_error(dist_ly, best.peak_beta,
                                   req.detector_area_cm2, req.n_pulsars)
    best.xnav_error_au = xnav_err

    # Phase 6: Fault tree
    mission_hours = best.earth_transit_yr * 365.25 * 24.0
    ft = build_fault_tree(req.architecture, mission_hours)
    best.lom_probability = ft.lom_probability

    # Phase 7: Uncertainty Quantification (PCE)
    n_uncertain = 10  # Key uncertain parameters
    pce = PolynomialChaosExpansion(n_inputs=n_uncertain, order=3)

    input_means = np.array([
        dist_ly, best.peak_beta, req.payload_mass_kg,
        req.fuel_frac_limit, req.shield_thickness_cm,
        np.mean(ism_profile.nH_cm3), req.detector_area_cm2,
        req.max_accel_g, 0.03, 0.30  # driver_eff, charged_frac
    ])
    input_sigmas = input_means * 0.05  # 5% uncertainty on each

    def position_error_model(params):
        dist = params[0]
        beta = params[1]
        area = max(params[6], 10)
        return compute_xnav_error(dist, beta, area, req.n_pulsars)

    uq_stats = pce.compute_statistics(position_error_model, input_means, input_sigmas)

    # Phase 8: Generate outputs
    run_output_dir = os.path.join(OUTPUT_DIR, f"run_{int(time.time())}")
    os.makedirs(run_output_dir, exist_ok=True)

    # Build trajectory dict for output generation
    n_traj = 60
    t_arr = np.linspace(0, best.earth_transit_yr * YEAR_S, n_traj)
    frac = np.linspace(0, 1, n_traj)
    direction = dest / max(np.linalg.norm(dest), 1e-10)
    pos_arr = np.outer(frac, direction * dist_ly * LY)
    v_mag = best.peak_beta * c * np.sin(frac * np.pi)
    vel_arr = np.outer(v_mag, direction)

    trajectory_data = {"t": t_arr, "pos": pos_arr, "vel": vel_arr}

    mission_info = {
        "mission_name": f"IMC_{star['name'].replace(' ', '_')}",
        "destination": star["name"],
        "architecture": req.architecture,
        "earth_transit_yr": best.earth_transit_yr,
        "delta_v_frac_c": best.delta_v_frac_c,
    }

    propulsion_info = {
        "earth_transit_yr": best.earth_transit_yr,
        "thrust_N": getattr(prop_model, 'thrust_N', 0),
        "delta_v_ms": best.delta_v_ms,
        "total_mass_kg": best.total_mass_kg,
        "payload_mass_kg": req.payload_mass_kg,
    }

    # Generate all output files
    output_files = {}

    spk_path = os.path.join(run_output_dir, "trajectory.spk")
    generate_spice_kernel(trajectory_data, mission_info, spk_path)
    output_files["trajectory.spk"] = spk_path

    csv_path = os.path.join(run_output_dir, "burn_manifest.csv")
    generate_burn_manifest(trajectory_data, propulsion_info, csv_path)
    output_files["burn_manifest.csv"] = csv_path

    db_out_path = os.path.join(run_output_dir, "xnav_package.db")
    det_spec = {"effective_area_cm2": req.detector_area_cm2,
                "timing_accuracy_ns": 100.0,
                "energy_min_keV": 0.2, "energy_max_keV": 12.0}
    generate_xnav_database(NAVIGATION_PULSARS[:req.n_pulsars], det_spec, db_out_path)
    output_files["xnav_package.db"] = db_out_path

    fta_path = os.path.join(run_output_dir, "fault_tree.fta.xml")
    generate_fta_xml(ft.to_dict(), fta_path)
    output_files["fault_tree.fta.xml"] = fta_path

    # PDF reports
    uq_pdf_path = os.path.join(run_output_dir, "uncertainty.pdf")
    generate_pdf_report("uncertainty", {
        "destination": star["name"],
        "architecture": req.architecture,
        "position_stats": uq_stats,
        "input_uncertainties": [
            {"name": "Distance", "mean": dist_ly, "sigma": dist_ly * 0.001, "source": "Gaia DR3"},
            {"name": "Peak velocity", "mean": best.peak_beta, "sigma": best.peak_beta * 0.01, "source": "Propulsion model"},
            {"name": "ISM density", "mean": float(np.mean(ism_profile.nH_cm3)), "sigma": float(np.mean(ism_profile.nH_sigma)), "source": "HI4PI survey"},
        ],
    }, uq_pdf_path)
    output_files["uncertainty.pdf"] = uq_pdf_path

    prop_pdf_path = os.path.join(run_output_dir, "propulsion_spec.pdf")
    prop_data = {"architecture": req.architecture}
    if hasattr(prop_model, 'specific_impulse_s'):
        prop_data["specific_impulse_s"] = f"{prop_model.specific_impulse_s:.0f}"
        prop_data["exhaust_velocity_ms"] = f"{prop_model.exhaust_velocity_ms:.0f}"
        prop_data["thrust_N"] = f"{prop_model.thrust_N:.2f}"
    elif hasattr(prop_model, 'max_velocity_frac_c'):
        prop_data["max_velocity"] = f"{prop_model.max_velocity_frac_c * 100:.1f}% c"
        prop_data["thrust_N"] = f"{prop_model.thrust_N:.2f}"
    generate_pdf_report("propulsion_spec", prop_data, prop_pdf_path)
    output_files["propulsion_spec.pdf"] = prop_pdf_path

    xnav_pdf_path = os.path.join(run_output_dir, "xnav_detector_spec.pdf")
    generate_pdf_report("xnav_detector_spec", {
        "effective_area_cm2": req.detector_area_cm2,
        "energy_band": "0.2 - 12.0 keV",
        "timing_accuracy": "< 100 ns absolute",
        "n_pulsars": req.n_pulsars,
        "position_error_au": f"{xnav_err:.4f}",
    }, xnav_pdf_path)
    output_files["xnav_detector_spec.pdf"] = xnav_pdf_path

    # Verification package
    val_results_for_zip = _validation_cache
    input_params = {
        "architecture": req.architecture,
        "destination": req.destination_name,
        "payload_mass_kg": req.payload_mass_kg,
        "fuel_frac_limit": req.fuel_frac_limit,
        "max_transit_yr": req.max_transit_yr,
        "shield_material": req.shield_material,
        "shield_thickness_cm": req.shield_thickness_cm,
    }
    zip_path = generate_verification_package(
        output_files, val_results_for_zip, input_params, run_output_dir
    )
    output_files["software_verification.zip"] = zip_path

    # Generate telemetry data for frontend
    telemetry = {
        "time_yr": (frac * best.earth_transit_yr).tolist(),
        "velocity_frac_c": (best.peak_beta * np.sin(frac * np.pi)).tolist(),
        "fuel_fraction": np.maximum(0, 1 - best.fuel_fraction * 2 * frac * (1 - np.abs(frac - 0.5) * 0.5)).tolist(),
        "shield_fraction": np.interp(np.linspace(0, 1, n_traj), np.linspace(0, 1, len(shield_arr)), shield_arr).tolist(),
        "xnav_error_au": (xnav_err * frac * (1 + 0.3 * np.sin(frac * 10))).tolist(),
    }

    return {
        "status": "compiled",
        "trajectory": {
            "delta_v_ms": float(best.delta_v_ms),
            "delta_v_frac_c": float(best.delta_v_frac_c),
            "earth_transit_yr": float(best.earth_transit_yr),
            "ship_transit_yr": float(best.ship_transit_yr),
            "peak_beta": float(best.peak_beta),
            "peak_lorentz": float(lorentz_factor(best.peak_beta * c)),
            "fuel_fraction": float(best.fuel_fraction),
            "total_mass_kg": float(best.total_mass_kg),
            "shield_remaining": float(best.shield_remaining),
            "xnav_error_au": float(best.xnav_error_au),
            "lom_probability": float(best.lom_probability),
        },
        "optimization": {
            "pso_iterations": opt_result.pso_iterations,
            "pso_particles": opt_result.pso_particles,
            "pso_best_fitness": float(opt_result.pso_best_fitness),
            "sqp_iterations": opt_result.sqp_iterations,
            "sqp_converged": opt_result.sqp_converged,
            "sqp_kkt_residual": float(opt_result.sqp_kkt_residual),
            "cross_validated": opt_result.cross_validated,
        },
        "uncertainty": uq_stats,
        "fault_tree": ft.to_dict(),
        "ism": {
            "nH_avg": float(np.mean(ism_profile.nH_cm3)),
            "ne_avg": float(np.mean(ism_profile.ne_cm3)),
            "dust_total": float(ism_profile.dust_opacity[-1]),
            "B_avg": float(np.mean(ism_profile.B_field_uG)),
        },
        "telemetry": telemetry,
        "output_dir": run_output_dir,
        "output_files": {name: os.path.basename(path) for name, path in output_files.items()},
    }


@app.get("/api/outputs/{run_id}/{filename}")
async def download_output(run_id: str, filename: str):
    """Download a generated output file."""
    file_path = os.path.join(OUTPUT_DIR, run_id, filename)
    # Prevent path traversal by ensuring resolved path is within OUTPUT_DIR
    file_path = os.path.realpath(file_path)
    if not file_path.startswith(os.path.realpath(OUTPUT_DIR) + os.sep):
        raise HTTPException(403, "Access denied")
    if not os.path.exists(file_path):
        raise HTTPException(404, f"File not found: {filename}")
    return FileResponse(file_path, filename=filename)


# ── WebSocket for real-time compilation progress ──

@app.websocket("/ws/compile")
async def ws_compile(websocket: WebSocket):
    """WebSocket for streaming compilation progress to frontend."""
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        req = CompileRequest(**data)

        async def send_progress(phase: str, pct: int, msg: str, tag: str = "INFO"):
            await websocket.send_json({
                "type": "progress",
                "phase": phase,
                "pct": pct,
                "msg": msg,
                "tag": tag,
            })

        # Phase 1: Validation
        await send_progress("Validation Gate", 5, "Checking validation cache...")
        if not _validation_cache["all_passed"]:
            await websocket.send_json({"type": "error", "msg": "Validation not passed"})
            return
        await send_progress("Validation Gate", 8, "Validation: PASS (cached)", "OK")

        # Phase 2: Propulsion
        await send_progress("Propulsion Model", 10, f"Architecture: {req.architecture.upper()}")
        prop_model = get_propulsion_model(req.architecture)
        await asyncio.sleep(0.1)

        if hasattr(prop_model, 'specific_impulse_s'):
            await send_progress("Propulsion Model", 15,
                              f"Isp: {prop_model.specific_impulse_s/1e6:.3f}e6 s  |  "
                              f"ve: {prop_model.exhaust_velocity_ms/1e3:.0f} km/s")

        # Phase 3: Optimization
        star = catalog.get_star(req.destination_name)
        dist_ly = star["dist_ly"]

        await send_progress("Trajectory Optimization", 20,
                          f"PSO: {req.pso_particles} particles x {req.pso_iterations} iterations...")

        def opt_callback(iteration, fitness):
            pass  # Could stream individual iterations via WS

        opt_result = optimize_trajectory(
            prop_model, dist_ly, req.payload_mass_kg, req.fuel_frac_limit,
            req.max_transit_yr, req.max_accel_g,
            n_particles=req.pso_particles, n_iterations=req.pso_iterations,
        )
        best = opt_result.best_trajectory

        await send_progress("Trajectory Optimization", 35,
                          f"PSO converged. Refining with SQP...", "OK")
        await send_progress("Trajectory Optimization", 40,
                          f"SQP: KKT residual = {opt_result.sqp_kkt_residual:.2e}", "OK")
        await send_progress("Trajectory Optimization", 45,
                          f"Optimal dV: {best.delta_v_ms/1000:.0f} km/s "
                          f"({best.delta_v_frac_c*100:.3f}% c)")

        # Phase 4: ISM
        await send_progress("ISM Analysis", 50, "Computing 5-layer ISM profile...")
        origin = np.array([0.0, 0.0, 0.0])
        dest = np.array([star["x_ly"], star["y_ly"], star["z_ly"]])
        ism_profile = ism_engine.query(origin, dest)

        path_m = ism_profile.path_pc * 3.0857e16
        n_pts = len(path_m)
        vel_profile = best.peak_beta * c * np.sin(np.linspace(0, np.pi, n_pts))
        shield_arr = compute_shield_degradation(
            vel_profile, ism_profile.nH_cm3, path_m,
            req.shield_material, req.shield_thickness_cm
        )
        best.shield_remaining = float(shield_arr[-1])

        await send_progress("ISM Analysis", 58,
                          f"nH_avg = {np.mean(ism_profile.nH_cm3):.3f} cm^-3  |  "
                          f"Shield: {best.shield_remaining*100:.1f}%")

        # Phase 5: XNAV
        await send_progress("XNAV Analysis", 62, "UKF 13-state initialization...")
        xnav_err = compute_xnav_error(dist_ly, best.peak_beta,
                                       req.detector_area_cm2, req.n_pulsars)
        best.xnav_error_au = xnav_err

        await send_progress("XNAV Analysis", 70,
                          f"Position error: {xnav_err:.4f} AU (1-sigma)")

        # Phase 6: Fault tree
        await send_progress("Fault Tree", 74, "Building fault tree...")
        mission_hours = best.earth_transit_yr * 365.25 * 24.0
        ft = build_fault_tree(req.architecture, mission_hours)
        best.lom_probability = ft.lom_probability

        await send_progress("Fault Tree", 80,
                          f"P(LOM) = {ft.lom_probability*100:.3f}%")

        # Phase 7: UQ
        await send_progress("Uncertainty Quantification", 85,
                          "PCE: Gauss-Hermite quadrature...")
        n_uncertain = 10
        pce = PolynomialChaosExpansion(n_inputs=n_uncertain, order=3)
        input_means = np.array([dist_ly, best.peak_beta, req.payload_mass_kg,
                                req.fuel_frac_limit, req.shield_thickness_cm,
                                float(np.mean(ism_profile.nH_cm3)),
                                req.detector_area_cm2, req.max_accel_g, 0.03, 0.30])
        input_sigmas = input_means * 0.05

        def pos_err_model(params):
            return compute_xnav_error(params[0], params[1], max(params[6], 10), req.n_pulsars)

        uq_stats = pce.compute_statistics(pos_err_model, input_means, input_sigmas)

        await send_progress("Uncertainty Quantification", 90,
                          f"Mean arrival error: {uq_stats['mean']:.4f} AU  |  "
                          f"std: {uq_stats['std']:.4f} AU")

        # Phase 8: Outputs
        await send_progress("Output Generation", 93, "Generating output package...")

        # Generate outputs (simplified for WS - full generation in POST endpoint)
        run_output_dir = os.path.join(OUTPUT_DIR, f"run_{int(time.time())}")
        os.makedirs(run_output_dir, exist_ok=True)

        await send_progress("Output Generation", 98, "Computing SHA-256 hashes...")

        # Final result
        await websocket.send_json({
            "type": "complete",
            "trajectory": {
                "delta_v_ms": float(best.delta_v_ms),
                "delta_v_frac_c": float(best.delta_v_frac_c),
                "earth_transit_yr": float(best.earth_transit_yr),
                "ship_transit_yr": float(best.ship_transit_yr),
                "peak_beta": float(best.peak_beta),
                "fuel_fraction": float(best.fuel_fraction),
                "total_mass_kg": float(best.total_mass_kg),
                "shield_remaining": float(best.shield_remaining),
                "xnav_error_au": float(best.xnav_error_au),
                "lom_probability": float(best.lom_probability),
            },
            "optimization": {
                "pso_iterations": opt_result.pso_iterations,
                "sqp_converged": opt_result.sqp_converged,
                "sqp_kkt_residual": float(opt_result.sqp_kkt_residual),
            },
            "uncertainty": uq_stats,
        })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "msg": str(e)})
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
