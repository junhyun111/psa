from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


R_UNIVERSAL = 8.314462618
MW_DRY_GAS = 0.02897
MW_WATER = 0.01801528

DEFAULT_BASE_DIR = Path(__file__).resolve().parents[1]
PREPROCESSING_CSV = DEFAULT_BASE_DIR / "preprocessing" / "test.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "guard_bed_synthetic_dataset.csv"


@dataclass(frozen=True)
class ReferenceStats:
    flow_mol_s_min: float
    flow_mol_s_max: float
    temp_c_min: float
    temp_c_max: float
    pressure_kpa_min: float
    pressure_kpa_max: float
    rh_min: float
    rh_max: float
    delta_p_kpa_mean: float
    delta_p_kpa_std: float
    delta_p_kpa_min: float
    delta_p_kpa_max: float
    superficial_velocity_mean: float
    superficial_velocity_std: float


def load_reference_stats() -> ReferenceStats:
    if PREPROCESSING_CSV.exists():
        df = pd.read_csv(
            PREPROCESSING_CSV,
            usecols=[
                "flow_mol_s",
                "T_in_C",
                "P_in_kPa",
                "RH_in",
                "deltaP_kPa",
                "superficial_velocity_m_s",
            ],
        )
        return ReferenceStats(
            flow_mol_s_min=float(df["flow_mol_s"].min()),
            flow_mol_s_max=float(df["flow_mol_s"].max()),
            temp_c_min=float(df["T_in_C"].min()),
            temp_c_max=float(df["T_in_C"].max()),
            pressure_kpa_min=float(df["P_in_kPa"].min()),
            pressure_kpa_max=float(df["P_in_kPa"].max()),
            rh_min=float(df["RH_in"].min()),
            rh_max=float(df["RH_in"].max()),
            delta_p_kpa_mean=float(df["deltaP_kPa"].mean()),
            delta_p_kpa_std=float(df["deltaP_kPa"].std()),
            delta_p_kpa_min=float(df["deltaP_kPa"].min()),
            delta_p_kpa_max=float(df["deltaP_kPa"].max()),
            superficial_velocity_mean=float(df["superficial_velocity_m_s"].mean()),
            superficial_velocity_std=float(df["superficial_velocity_m_s"].std()),
        )

    return ReferenceStats(
        flow_mol_s_min=480.0,
        flow_mol_s_max=760.0,
        temp_c_min=25.0,
        temp_c_max=50.0,
        pressure_kpa_min=98.0,
        pressure_kpa_max=112.0,
        rh_min=0.55,
        rh_max=0.98,
        delta_p_kpa_mean=38.0,
        delta_p_kpa_std=8.0,
        delta_p_kpa_min=15.0,
        delta_p_kpa_max=66.0,
        superficial_velocity_mean=3.1,
        superficial_velocity_std=0.45,
    )


def saturation_pressure_water_kpa(temp_c: float) -> float:
    return 0.61094 * np.exp((17.625 * temp_c) / (temp_c + 243.04))


def clamp(value: float, lower: float, upper: float) -> float:
    return float(np.clip(value, lower, upper))


def sample_truncated_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    lower: float,
    upper: float,
) -> float:
    for _ in range(100):
        value = rng.normal(mean, std)
        if lower <= value <= upper:
            return float(value)
    return clamp(mean, lower, upper)


def compute_water_mole_fraction(temp_c: float, pressure_kpa: float, rh: float) -> float:
    p_sat = saturation_pressure_water_kpa(temp_c)
    p_h2o = min(rh * p_sat, 0.95 * pressure_kpa)
    return clamp(p_h2o / pressure_kpa, 1e-4, 0.25)


def compute_gas_density_kg_m3(temp_c: float, pressure_kpa: float, water_y: float) -> float:
    temp_k = temp_c + 273.15
    avg_mw = (1.0 - water_y) * MW_DRY_GAS + water_y * MW_WATER
    return (pressure_kpa * 1000.0) * avg_mw / (R_UNIVERSAL * temp_k)


def compute_gas_viscosity_pa_s(temp_c: float) -> float:
    temp_k = temp_c + 273.15
    mu_ref = 1.716e-5
    t_ref = 273.15
    sutherland = 111.0
    return mu_ref * ((temp_k / t_ref) ** 1.5) * ((t_ref + sutherland) / (temp_k + sutherland))


def estimate_superficial_velocity(
    flow_mol_s: float,
    temp_c: float,
    pressure_kpa: float,
    bed_diameter_m: float,
) -> float:
    temp_k = temp_c + 273.15
    flow_actual_m3_s = flow_mol_s * R_UNIVERSAL * temp_k / (pressure_kpa * 1000.0)
    area = np.pi * (bed_diameter_m / 2.0) ** 2
    return flow_actual_m3_s / area


def estimate_reynolds_particle(
    density_kg_m3: float,
    velocity_m_s: float,
    particle_diameter_m: float,
    viscosity_pa_s: float,
) -> float:
    return density_kg_m3 * velocity_m_s * particle_diameter_m / viscosity_pa_s


def estimate_pressure_drop_kpa(row: dict[str, float], ref: ReferenceStats) -> float:
    eps = row["bed_porosity"]
    dp = row["particle_diameter_m"]
    mu = row["gas_viscosity_Pa_s"]
    rho = row["gas_density_kg_m3"]
    v = row["superficial_velocity_m_s"]
    l_bed = row["bed_length_m"]

    ergun_pa = l_bed * (
        (150.0 * mu * (1.0 - eps) ** 2 / (dp**2 * eps**3)) * v
        + (1.75 * rho * (1.0 - eps) / (dp * eps**3)) * v**2
    )
    ergun_kpa = ergun_pa / 1000.0

    humidity_penalty = 1.0 + 0.25 * row["relative_humidity"] + 0.04 * row["water_loading_factor"]
    fouling_penalty = 1.0 + 0.12 * row["pm25_mg_m3"] / 20.0 + 0.08 * row["mist_mg_m3"] / 10.0
    aging_penalty = 1.0 + 0.00012 * row["guard_bed_age_h"] + 0.0015 * row["cycles_since_regen"]
    calibrated = ergun_kpa * humidity_penalty * fouling_penalty * aging_penalty

    calibration_factor = 0.22
    adjusted = calibrated * calibration_factor
    return clamp(adjusted, ref.delta_p_kpa_min, ref.delta_p_kpa_max)


def simulate_single_case(
    rng: np.random.Generator,
    ref: ReferenceStats,
    simulation_id: int,
) -> dict[str, Any]:
    flow_mol_s = rng.uniform(ref.flow_mol_s_min, ref.flow_mol_s_max)
    temp_c = rng.uniform(ref.temp_c_min, ref.temp_c_max)
    pressure_kpa = rng.uniform(ref.pressure_kpa_min, ref.pressure_kpa_max)
    rh = rng.uniform(ref.rh_min, ref.rh_max)

    bed_diameter_m = rng.uniform(2.6, 3.8)
    bed_length_m = rng.uniform(1.2, 2.5)
    particle_diameter_m = rng.uniform(0.0030, 0.0060)
    bed_porosity = rng.uniform(0.36, 0.48)
    particle_sphericity = rng.uniform(0.72, 0.93)
    adsorbent_bulk_density_kg_m3 = rng.uniform(540.0, 760.0)

    pm25_mg_m3 = rng.uniform(1.0, 18.0)
    so2_ppmv = rng.uniform(0.05, 12.0)
    nox_ppmv = rng.uniform(0.2, 35.0)
    hcl_ppmv = rng.uniform(0.01, 4.0)
    h2s_ppmv = rng.uniform(0.01, 3.5)
    mist_mg_m3 = rng.uniform(0.1, 10.0)

    co2_vol_pct = rng.uniform(8.0, 18.0)
    o2_vol_pct = rng.uniform(1.0, 8.0)
    n2_vol_pct = clamp(100.0 - co2_vol_pct - o2_vol_pct, 70.0, 91.0)

    guard_bed_age_h = rng.uniform(48.0, 2500.0)
    cycles_since_regen = rng.integers(0, 25)
    inlet_dew_point_c = temp_c - rng.uniform(2.0, 12.0)
    residence_time_target_s = rng.uniform(0.15, 0.45)
    pretreatment_pm_efficiency = rng.uniform(0.75, 0.98)
    pretreatment_acid_efficiency = rng.uniform(0.10, 0.55)

    water_y_in = compute_water_mole_fraction(temp_c, pressure_kpa, rh)
    gas_density = compute_gas_density_kg_m3(temp_c, pressure_kpa, water_y_in)
    gas_viscosity = compute_gas_viscosity_pa_s(temp_c)
    superficial_velocity = estimate_superficial_velocity(flow_mol_s, temp_c, pressure_kpa, bed_diameter_m)
    reynolds_particle = estimate_reynolds_particle(
        density_kg_m3=gas_density,
        velocity_m_s=superficial_velocity,
        particle_diameter_m=particle_diameter_m,
        viscosity_pa_s=gas_viscosity,
    )

    bed_volume_m3 = np.pi * (bed_diameter_m / 2.0) ** 2 * bed_length_m
    void_volume_m3 = bed_volume_m3 * bed_porosity
    flow_actual_m3_s = flow_mol_s * R_UNIVERSAL * (temp_c + 273.15) / (pressure_kpa * 1000.0)
    residence_time_s = void_volume_m3 / max(flow_actual_m3_s, 1e-6)

    pretreated_pm_mg_m3 = pm25_mg_m3 * (1.0 - pretreatment_pm_efficiency)
    pretreated_so2_ppmv = so2_ppmv * (1.0 - pretreatment_acid_efficiency)
    pretreated_nox_ppmv = nox_ppmv * (1.0 - 0.35 * pretreatment_acid_efficiency)
    pretreated_hcl_ppmv = hcl_ppmv * (1.0 - 1.15 * pretreatment_acid_efficiency)
    pretreated_h2s_ppmv = h2s_ppmv * (1.0 - 0.85 * pretreatment_acid_efficiency)

    total_inlet_impurity_ppmv = (
        pretreated_so2_ppmv + pretreated_nox_ppmv + pretreated_hcl_ppmv + pretreated_h2s_ppmv
    )
    water_loading_factor = 1000.0 * water_y_in + 0.35 * pretreated_pm_mg_m3 + 0.2 * mist_mg_m3

    row = {
        "bed_porosity": bed_porosity,
        "particle_diameter_m": particle_diameter_m,
        "gas_viscosity_Pa_s": gas_viscosity,
        "gas_density_kg_m3": gas_density,
        "superficial_velocity_m_s": superficial_velocity,
        "bed_length_m": bed_length_m,
        "relative_humidity": rh,
        "water_loading_factor": water_loading_factor,
        "pm25_mg_m3": pretreated_pm_mg_m3,
        "mist_mg_m3": mist_mg_m3,
        "guard_bed_age_h": guard_bed_age_h,
        "cycles_since_regen": float(cycles_since_regen),
    }
    delta_p_kpa = estimate_pressure_drop_kpa(row, ref)

    humidity_competition = clamp((rh - 0.45) / 0.55, 0.0, 1.0)
    acid_load_index = total_inlet_impurity_ppmv / 30.0
    pm_fouling_index = pretreated_pm_mg_m3 / 5.0
    age_index = guard_bed_age_h / 2000.0
    flow_stress = superficial_velocity / max(ref.superficial_velocity_mean, 1e-6)
    pressure_penalty = delta_p_kpa / max(ref.delta_p_kpa_mean, 1e-6)

    moisture_removal_eff = (
        0.84
        + 0.10 * np.exp(-((residence_time_s - residence_time_target_s) ** 2) / 0.5)
        - 0.22 * humidity_competition
        - 0.05 * age_index
        - 0.03 * pressure_penalty
    )
    moisture_removal_eff = clamp(moisture_removal_eff, 0.05, 0.95)

    impurity_removal_eff = (
        0.90
        + 0.08 * np.tanh(0.9 * residence_time_s)
        - 0.10 * humidity_competition
        - 0.08 * acid_load_index
        - 0.05 * pm_fouling_index
        - 0.05 * age_index
        - 0.04 * max(flow_stress - 1.0, 0.0)
    )
    impurity_removal_eff = clamp(impurity_removal_eff, 0.10, 0.98)

    h2o_out_y = water_y_in * (1.0 - moisture_removal_eff)
    so2_out_ppmv = pretreated_so2_ppmv * (1.0 - impurity_removal_eff * 1.05)
    nox_out_ppmv = pretreated_nox_ppmv * (1.0 - impurity_removal_eff * 0.85)
    hcl_out_ppmv = pretreated_hcl_ppmv * (1.0 - impurity_removal_eff * 1.10)
    h2s_out_ppmv = pretreated_h2s_ppmv * (1.0 - impurity_removal_eff * 0.95)

    so2_out_ppmv = max(so2_out_ppmv, 0.0)
    nox_out_ppmv = max(nox_out_ppmv, 0.0)
    hcl_out_ppmv = max(hcl_out_ppmv, 0.0)
    h2s_out_ppmv = max(h2s_out_ppmv, 0.0)
    impurity_out_ppmv = so2_out_ppmv + nox_out_ppmv + hcl_out_ppmv + h2s_out_ppmv

    h2o_out_ppmv = h2o_out_y * 1e6
    h2o_in_ppmv = water_y_in * 1e6

    return {
        "simulation_id": simulation_id,
        "flow_mol_s": flow_mol_s,
        "T_in_C": temp_c,
        "P_in_kPa": pressure_kpa,
        "relative_humidity": rh,
        "H2O_in_ppmv": h2o_in_ppmv,
        "inlet_dew_point_C": inlet_dew_point_c,
        "CO2_in_vol_pct": co2_vol_pct,
        "N2_in_vol_pct": n2_vol_pct,
        "O2_in_vol_pct": o2_vol_pct,
        "PM2_5_in_mg_m3": pm25_mg_m3,
        "mist_in_mg_m3": mist_mg_m3,
        "SO2_in_ppmv": so2_ppmv,
        "NOx_in_ppmv": nox_ppmv,
        "HCl_in_ppmv": hcl_ppmv,
        "H2S_in_ppmv": h2s_ppmv,
        "pretreatment_pm_efficiency": pretreatment_pm_efficiency,
        "pretreatment_acid_efficiency": pretreatment_acid_efficiency,
        "PM2_5_after_pretreatment_mg_m3": pretreated_pm_mg_m3,
        "SO2_after_pretreatment_ppmv": pretreated_so2_ppmv,
        "NOx_after_pretreatment_ppmv": pretreated_nox_ppmv,
        "HCl_after_pretreatment_ppmv": pretreated_hcl_ppmv,
        "H2S_after_pretreatment_ppmv": pretreated_h2s_ppmv,
        "bed_length_m": bed_length_m,
        "bed_diameter_m": bed_diameter_m,
        "particle_diameter_m": particle_diameter_m,
        "bed_porosity": bed_porosity,
        "particle_sphericity": particle_sphericity,
        "adsorbent_bulk_density_kg_m3": adsorbent_bulk_density_kg_m3,
        "guard_bed_age_h": guard_bed_age_h,
        "cycles_since_regen": int(cycles_since_regen),
        "gas_density_kg_m3": gas_density,
        "gas_viscosity_Pa_s": gas_viscosity,
        "flow_actual_m3_s": flow_actual_m3_s,
        "superficial_velocity_m_s": superficial_velocity,
        "Re_particle": reynolds_particle,
        "residence_time_s": residence_time_s,
        "deltaP_guard_bed_kPa": delta_p_kpa,
        "moisture_removal_efficiency": moisture_removal_eff,
        "impurity_removal_efficiency": impurity_removal_eff,
        "H2O_out": h2o_out_ppmv,
        "H2O_out_ppmv": h2o_out_ppmv,
        "SO2_out_ppmv": so2_out_ppmv,
        "NOx_out_ppmv": nox_out_ppmv,
        "HCl_out_ppmv": hcl_out_ppmv,
        "H2S_out_ppmv": h2s_out_ppmv,
        "impurity_out": impurity_out_ppmv,
        "impurity_out_ppmv": impurity_out_ppmv,
        "pressure_drop_model_source": "calibrated_ergun_with_preprocessing_reference",
    }


def build_dataset(n_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ref = load_reference_stats()
    rows = [simulate_single_case(rng, ref, idx + 1) for idx in range(n_samples)]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic guard bed outlet data for MOF CO2 adsorption pretreatment studies."
    )
    parser.add_argument("--n-samples", type=int, default=100, help="Number of simulations to run.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV output path.",
    )
    args = parser.parse_args()

    df = build_dataset(n_samples=args.n_samples, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
