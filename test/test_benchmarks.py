"""
Smoke tests of the benchmark script benchmarks/operational_cases.py at a small scale, so that it keeps working
"""

import importlib.util
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

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


def test_synthetic_native():
    """
    The native ICON cut-outs have the NaN patterns of the grid database and fewer points than the whole domain
    """

    snow = oc.icon_native("icon_ch1_native_snow", scale=0.3).SNOWLMT.isnull()
    assert 0.25 < float(snow.mean()) < 0.35
    assert 0.2 < float(snow.all(["forecast_reference_time", "step", "realization"]).mean()) < 0.28
    assert oc.icon_native("icon_ch1_native", scale=0.3).ASWDIFD_S.notnull().all()
    lake = oc.icon_native("icon_ch2_native_lake", scale=0.3).T_2M.isnull().all(["step", "realization"])
    assert 0.01 < float(lake.mean()) < 0.03
    domain = oc.icon_native("icon_ch2_native_domain", scale=0.3)
    assert 0.08 < lake.sizes["values"] / domain.sizes["values"] < 0.12


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


def test_real_native(tmp_path):
    """
    A run of the open data in the structure of earthkit is written as a native source file and read back: lakes
    masked with its land fraction, smaller sizes than the operational ones noted
    """

    synthetic = oc.icon_native("icon_ch2_native_domain", scale=0.05)
    n_points = synthetic.sizes["values"]
    values = synthetic.T_2M.transpose("realization", "forecast_reference_time", "step", "values").values
    ogd = xr.Dataset(
        {"T_2M": (("member", "forecast_reference_time", "step", "level", "level_type", "values"),
                  values[:, :, :, None, None, :].astype(np.float64))},
        coords={
            "member": [str(i) for i in range(values.shape[0])][::-1],
            "forecast_reference_time": synthetic.forecast_reference_time.values,
            "step": synthetic.step.values,
            "level": [2],
            "level_type": ["height_above_ground_level"],
            "latitude": ("values", synthetic.latitude.values),
            "longitude": ("values", synthetic.longitude.values),
        },
    )  # fmt: skip
    land = np.ones(n_points)
    land[:10] = 0.2
    ds = oc._normalize_ogd(ogd, {"HSURF": synthetic.HSURF.values, "FR_LAND": land})
    assert ds.T_2M.dims == ("forecast_reference_time", "step", "realization", "values")
    assert ds.T_2M.dtype == np.float32 and ds.realization.values.tolist() == list(range(values.shape[0]))
    ds.to_netcdf(tmp_path / "icon_ch2_native.nc")

    lake, real = oc.load_input("icon_ch2_native_lake", 0.05, tmp_path)
    domain, _ = oc.load_input("icon_ch2_native_domain", 0.05, tmp_path)
    assert real and "land_area_fraction" not in lake.coords and "land_area_fraction" not in domain.coords
    assert domain.T_2M.isel(values=slice(10)).notnull().all()
    in_cut = oc._in_cut_out(*oc.Transformer.from_crs("EPSG:4326", oc.CRS.from_cf(oc.ROTATED), always_xy=True)
                            .transform(synthetic.longitude.values[:10], synthetic.latitude.values[:10]))  # fmt: skip
    assert int(lake.T_2M.isnull().all(["forecast_reference_time", "step", "realization"]).sum()) == in_cut.sum()
    assert "realization instead of 21" in oc._native_sizes_note("icon_ch2_native_domain", domain)
