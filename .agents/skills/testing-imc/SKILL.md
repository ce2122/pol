# Testing the Interstellar Mission Compiler (IMC) v2.0

## Overview
The IMC is a FastAPI backend + HTML frontend application for interstellar mission planning. The backend implements real numerical physics (integrators, PSO+SQP optimization, UKF XNAV, PCE UQ) and serves the HTML frontend as a static file.

## Setup

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

The frontend is served at `http://localhost:8000/` — no separate frontend build step needed.

### Verification
- `GET /api/health` should return `{"status": "ok", "stars": 30, ...}`
- The frontend auto-detects the backend via `/api/health` and shows "Loaded 30 star targets from server catalog" in the log
- If backend is down, frontend gracefully degrades to client-side analytical mode (11 hardcoded stars)

## E2E Test Flow

1. **Backend detection**: Load page, verify log shows "Loaded 30 stars" (not 11)
2. **Propulsion selection**: Click a propulsion card (Fusion, ACFF, or Lightsail)
3. **Destination selection**: Use the destination dropdown (should show 30 options from backend catalog)
4. **Validation**: Click "Run Validation" tab → "Run All Tests" button. This takes ~45-50 seconds due to real numerical integration. All 5 tests should PASS.
5. **Compilation**: Click "Compile Mission". Uses WebSocket (`/ws/compile`) for real-time progress. Runs through 6+ phases.
6. **Output downloads**: Click "Output Package" tab → click file cards to download

## Key Test Assertions

### Validation (proves real physics, not fakes)
- Mercury perihelion precession: ~42.xx arcsec/century (GR prediction is 42.98)
- Voyager 1 (43yr propagation): r ~164 AU, v ~17 km/s
- Pioneer anomaly: ~8.x × 10⁻¹⁰ m/s²
- New Horizons: r_Pluto ~34-35 AU
- Alpha Cen AB: a_Kepler ~23 AU

### Output files
- A real SPICE kernel should start with binary `DAF/SPK` header (use `xxd file.spk | head` to check)
- Plain-text blobs starting with `// IMC v2.0` indicate the frontend is generating client-side fakes

## Known Issues & Workarounds

- **Validation might hang or be slow**: The `integrate_simple` method does real numerical integration. If it hangs, check that it uses a single `solve_ivp` call (not per-day-timestep iteration). A 43-year Voyager integration with per-day calls = ~15,700 solve_ivp invocations → server hangs.
- **numpy.bool_ serialization errors**: FastAPI's `jsonable_encoder` may fail on numpy types. Validation endpoint needs explicit `bool()`, `float()`, `int()` conversions on all numpy values before returning JSON.
- **WebSocket compile path might skip output file generation**: The WS path at `/ws/compile` may not generate actual output files on disk (only the HTTP POST `/api/compile` path does). Check the backend output directory after compilation to verify files exist.
- **Destination dropdown mismatch**: The dropdown selection index might not match the displayed star name in the header. Verify the JSON response `destination` field matches what was selected.
- **Large transit time values**: For certain architecture+destination combinations, the optimizer may return extremely large transit times (e.g., 10¹³ years). This may indicate the optimization bounds or propulsion model needs tuning.

## Tips

- The frontend uses both HTTP and WebSocket APIs. Compilation primarily uses the WebSocket path for real-time progress streaming.
- No external APIs or secrets are needed — the app is fully self-contained with a local SQLite catalog.
- The validation suite is the best proof that the backend physics engine works correctly. If all 5 tests pass with realistic values, the core engine is functioning.
- Screenshot the validation results panel — it shows Computed vs Expected values side by side, which is the strongest evidence of real physics.

## Devin Secrets Needed

None — this application is fully self-contained with no external service dependencies.
