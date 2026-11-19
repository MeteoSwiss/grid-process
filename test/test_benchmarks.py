"""
Smoke tests of the benchmark script benchmarks/operational_cases.py at a small scale, so that it keeps working
"""

import importlib.util
import json
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import grid_process.grid_interp as gri

SCRIPT = Path(__file__).parents[1] / "benchmarks" / "operational_cases.py"
SPEC = importlib.util.spec_from_file_location("operational_cases", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
oc = importlib.util.module_from_spec(SPEC)
sys.modules["operational_cases"] = oc  # for the dataclasses
SPEC.loader.exec_module(oc)


@pytest.mark.parametrize("case", oc.CASES, ids=lambda case: case.name)
def test_operational_case(case):
    """
    Each case runs on its synthetic data at a small scale
    """

    source, target, real = oc.case_inputs(case, scale=0.05)
    assert not any(real.values())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = case.call(gri, source, target)
    assert out.data_vars
    assert all(var.size > 0 for var in out.data_vars.values())


def test_synthetic_nan():
    """
    The synthetic data has the NaN patterns of the operational data
    """

    snow = oc.grid_database("icon_ch1_snow", scale=0.3).snow_limit.isnull()
    assert 0.25 < float(snow.mean()) < 0.35
    assert 0.2 < float(snow.all(["forecast_reference_time", "lead_time", "realization"]).mean()) < 0.28
    radiation = oc.grid_database("icon_ch1", scale=0.3).surface_diffuse_downwelling_shortwave_flux_in_air
    assert radiation.isel(lead_time=0).isnull().all() and radiation.isel(lead_time=slice(1, None)).notnull().all()


def test_run_and_compare(tmp_path):
    """
    The command line runs a case in worker processes and compares two runs
    """

    for name in ("a", "b"):
        command = [sys.executable, str(SCRIPT), "run", "--out", str(tmp_path / name), "--scale", "0.05"]
        command += ["--repeat", "1", "--case", "poi_icon_ch1_idw_k49"]
        subprocess.run(command, check=True, capture_output=True)
    results = json.loads((tmp_path / "a" / "results.json").read_text())["cases"]["poi_icon_ch1_idw_k49"]
    assert results["first_s"] > 0 and len(results["repeat_s"]) == 1 and results["traced_first_gb"] >= 0
    compared = subprocess.run(
        [sys.executable, str(SCRIPT), "compare", str(tmp_path / "a"), str(tmp_path / "b")],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "identical" in compared.stdout
