from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from optimize_policy import (
    CO2_MW_KG_PER_MOL,
    DECISION_COLS,
    CostConfig,
    PSAEconomicsOptimizer,
    ProcessConstants,
    load_training_data,
    normalize_fixed_input,
)


def locate_project_paths(module_dir: Path | None = None) -> dict[str, Path]:
    if module_dir is None:
        cwd = Path.cwd().resolve()
        candidates = [
            cwd,
            cwd / "mof" / "optimization",
            cwd.parent / "optimization",
        ]
        module_dir = next((path for path in candidates if (path / "optimize_policy.py").exists()), None)
        if module_dir is None:
            raise FileNotFoundError("optimize_policy.py를 찾을 수 없습니다.")

    module_dir = module_dir.resolve()
    model_dir = module_dir.parent / "model"
    dataset_dir = module_dir.parent / "dataset"
    train_candidates = [
        model_dir / "output.csv",
        dataset_dir / "data.csv",
        dataset_dir / "test.csv",
    ]
    train_csv = next((path for path in train_candidates if path.exists()), None)
    if train_csv is None:
        raise FileNotFoundError("학습용 CSV를 찾지 못했습니다.")

    purity_model_path = model_dir / "CO2_purity_dnn.keras"
    recovery_model_path = model_dir / "CO2_recovery_dnn.keras"
    if not purity_model_path.exists() or not recovery_model_path.exists():
        raise FileNotFoundError("DNN surrogate 모델 파일을 찾지 못했습니다.")

    return {
        "module_dir": module_dir,
        "model_dir": model_dir,
        "dataset_dir": dataset_dir,
        "train_csv": train_csv,
        "purity_model_path": purity_model_path,
        "recovery_model_path": recovery_model_path,
    }


def build_default_cost_config(train_df: pd.DataFrame) -> CostConfig:
    purity_target = float(min(0.90, train_df["CO2_purity"].quantile(0.90)))
    recovery_target = float(min(0.90, train_df["CO2_recovery"].quantile(0.90)))
    return CostConfig(
        electricity_cost_per_kwh=110.0,
        carbon_price_per_tco2=50_000.0,
        co2_credit_per_tco2=0.0,
        annualized_capex_per_m3_bed=200_000.0,
        annual_fixed_om_per_m3_bed=40_000.0,
        annual_adsorbent_budget_per_m3_bed=80_000.0,
        purity_target=purity_target,
        recovery_target=recovery_target,
        purity_penalty_weight=120_000.0,
        recovery_penalty_weight=150_000.0,
        uncertainty_penalty_weight=0.0,
    )


def initialize_optimizer(module_dir: Path | None = None) -> tuple[PSAEconomicsOptimizer, dict[str, Path], CostConfig, ProcessConstants]:
    paths = locate_project_paths(module_dir)
    train_df = load_training_data(paths["train_csv"])
    base_cost = build_default_cost_config(train_df)
    base_constants = ProcessConstants()
    optimizer = PSAEconomicsOptimizer(
        train_df=train_df,
        purity_model_path=paths["purity_model_path"],
        recovery_model_path=paths["recovery_model_path"],
        constants=base_constants,
        cost=base_cost,
    )
    return optimizer, paths, base_cost, base_constants


def _build_profile_with_step_limits(
    baseline: np.ndarray,
    rng: np.random.Generator,
    min_value: float,
    max_value: float,
    max_step_frac: float | None = None,
    max_step_abs: float | None = None,
    noise_scale: float = 0.0,
) -> np.ndarray:
    values = np.zeros_like(baseline, dtype=float)
    values[0] = float(np.clip(baseline[0], min_value, max_value))
    for idx in range(1, len(baseline)):
        candidate = baseline[idx] + rng.normal(0.0, noise_scale)
        lower = min_value
        upper = max_value
        prev = values[idx - 1]
        if max_step_frac is not None:
            lower = max(lower, prev * (1.0 - max_step_frac))
            upper = min(upper, prev * (1.0 + max_step_frac))
        if max_step_abs is not None:
            lower = max(lower, prev - max_step_abs)
            upper = min(upper, prev + max_step_abs)
        values[idx] = float(np.clip(candidate, lower, upper))
    return values


def _time_of_use_price_profile(length: int, low: float, normal: float, high: float, peak: float) -> np.ndarray:
    slots_per_day = 96
    profile = np.zeros(length, dtype=float)
    for idx in range(length):
        hour = (idx % slots_per_day) / 4.0
        if 0 <= hour < 6:
            profile[idx] = low
        elif 6 <= hour < 12:
            profile[idx] = normal
        elif 12 <= hour < 18:
            profile[idx] = high
        elif 18 <= hour < 22:
            profile[idx] = peak
        else:
            profile[idx] = normal
    return profile


def _load_block_profile(length: int, base: float, low_mult: float, med_mult: float, high_mult: float) -> np.ndarray:
    slots_per_day = 96
    profile = np.zeros(length, dtype=float)
    for idx in range(length):
        hour = (idx % slots_per_day) / 4.0
        if 0 <= hour < 8:
            mult = low_mult
        elif 8 <= hour < 16:
            mult = med_mult
        elif 16 <= hour < 20:
            mult = high_mult
        else:
            mult = 0.95
        profile[idx] = base * mult
    smooth = profile.copy()
    for idx in range(1, len(profile)):
        smooth[idx] = 0.65 * smooth[idx - 1] + 0.35 * profile[idx]
    return smooth


def generate_synthetic_dataset(
    scenario: str,
    start: str = "2026-01-01 00:00",
    hours: int = 48,
    freq: str = "15min",
    seed: int = 42,
) -> pd.DataFrame:
    periods = int(hours * 60 / 15)
    timestamps = pd.date_range(start=start, periods=periods, freq=freq)
    t = np.arange(periods)
    day_wave = np.sin(2.0 * np.pi * t / 96.0)
    half_day_wave = np.sin(2.0 * np.pi * t / 48.0 + 0.75)
    rng = np.random.default_rng(seed + abs(hash(scenario)) % 10_000)

    flow_base = 1000.0
    conc_base = 0.20
    ambient_base = 22.0
    heat_base = 42.0
    capacity_base = 7.0

    if scenario == "normal":
        flow_baseline = flow_base * (1.0 + 0.08 * day_wave + 0.02 * half_day_wave)
        conc_baseline = conc_base + 0.015 * day_wave - 0.005 * half_day_wave
        price_baseline = _time_of_use_price_profile(periods, 95.0, 120.0, 165.0, 210.0)
        ambient_baseline = ambient_base + 5.0 * day_wave + 1.2 * half_day_wave
        capture_baseline = np.full(periods, 0.72)
        capacity_baseline = np.full(periods, capacity_base)
    elif scenario == "price_volatile":
        flow_baseline = flow_base * (1.0 + 0.06 * day_wave)
        conc_baseline = conc_base + 0.012 * day_wave
        price_baseline = _time_of_use_price_profile(periods, 85.0, 135.0, 210.0, 285.0)
        price_baseline[56:72] += np.linspace(0.0, 35.0, 16)
        price_baseline[72:84] += 35.0
        ambient_baseline = ambient_base + 4.5 * day_wave
        capture_baseline = np.full(periods, 0.72)
        capacity_baseline = np.full(periods, capacity_base)
    elif scenario == "load_volatile":
        flow_baseline = _load_block_profile(periods, flow_base, 0.90, 1.10, 1.20)
        conc_baseline = conc_base + 0.018 * day_wave + 0.010 * (flow_baseline / flow_base - 1.0)
        price_baseline = _time_of_use_price_profile(periods, 95.0, 118.0, 160.0, 205.0)
        ambient_baseline = ambient_base + 5.5 * day_wave
        capture_baseline = np.full(periods, 0.72)
        capacity_baseline = np.full(periods, capacity_base + 0.25 * day_wave)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    flow = _build_profile_with_step_limits(
        baseline=flow_baseline,
        rng=rng,
        min_value=800.0,
        max_value=1200.0,
        max_step_frac=0.05,
        noise_scale=18.0,
    )
    co2_concentration = _build_profile_with_step_limits(
        baseline=conc_baseline,
        rng=rng,
        min_value=0.16,
        max_value=0.24,
        max_step_abs=0.005,
        noise_scale=0.0025,
    )
    electricity_price = _build_profile_with_step_limits(
        baseline=price_baseline,
        rng=rng,
        min_value=80.0,
        max_value=300.0,
        max_step_frac=0.10,
        noise_scale=6.0,
    )
    heat_price = _build_profile_with_step_limits(
        baseline=heat_base + 4.0 * day_wave + 1.0 * half_day_wave,
        rng=rng,
        min_value=25.0,
        max_value=80.0,
        max_step_frac=0.06,
        noise_scale=1.5,
    )
    ambient_temp = _build_profile_with_step_limits(
        baseline=ambient_baseline,
        rng=rng,
        min_value=10.0,
        max_value=35.0,
        max_step_abs=0.7,
        noise_scale=0.25,
    )
    capture_target = _build_profile_with_step_limits(
        baseline=capture_baseline,
        rng=rng,
        min_value=0.68,
        max_value=0.75,
        max_step_abs=0.01,
        noise_scale=0.003,
    )
    max_capacity = _build_profile_with_step_limits(
        baseline=capacity_baseline,
        rng=rng,
        min_value=6.4,
        max_value=7.8,
        max_step_abs=0.12,
        noise_scale=0.04,
    )

    h2o_ppmv = _build_profile_with_step_limits(
        baseline=52.0 + 18.0 * day_wave + 0.70 * (ambient_temp - ambient_temp.mean()),
        rng=rng,
        min_value=5.0,
        max_value=95.0,
        max_step_abs=4.5,
        noise_scale=2.0,
    )
    sox_ppmv = _build_profile_with_step_limits(
        baseline=0.48 + 0.06 * half_day_wave,
        rng=rng,
        min_value=0.05,
        max_value=0.95,
        max_step_abs=0.05,
        noise_scale=0.015,
    )
    nox_ppmv = _build_profile_with_step_limits(
        baseline=5.1 + 0.45 * day_wave + 0.70 * (flow / flow_base - 1.0),
        rng=rng,
        min_value=1.0,
        max_value=9.5,
        max_step_abs=0.35,
        noise_scale=0.10,
    )
    dust_mg_nm3 = _build_profile_with_step_limits(
        baseline=0.050 + 0.006 * half_day_wave + 0.008 * (flow / flow_base - 1.0),
        rng=rng,
        min_value=0.010,
        max_value=0.095,
        max_step_abs=0.005,
        noise_scale=0.0015,
    )

    df = pd.DataFrame({
        "timestamp": timestamps,
        "scenario": scenario,
        "co2_flow_rate": flow,
        "co2_concentration": co2_concentration,
        "electricity_price": electricity_price,
        "heat_price": heat_price,
        "ambient_temp": ambient_temp,
        "capture_target": capture_target,
        "max_capacity": max_capacity,
        "co2_mol_frac": co2_concentration,
        "h2o_ppmv": h2o_ppmv,
        "sox_ppmv": sox_ppmv,
        "nox_ppmv": nox_ppmv,
        "dust_mg_Nm3": dust_mg_nm3,
    })
    return df


def generate_all_scenarios(
    start: str = "2026-01-01 00:00",
    hours: int = 48,
    freq: str = "15min",
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    scenarios = ["normal", "price_volatile", "load_volatile"]
    return {
        scenario: generate_synthetic_dataset(
            scenario=scenario,
            start=start,
            hours=hours,
            freq=freq,
            seed=seed,
        )
        for scenario in scenarios
    }


def row_to_fixed_input(row: pd.Series | dict[str, Any]) -> dict[str, float]:
    row = dict(row)
    return normalize_fixed_input({
        "co2_mol_frac": float(row["co2_mol_frac"]),
        "h2o_ppmv": float(row["h2o_ppmv"]),
        "sox_ppmv": float(row["sox_ppmv"]),
        "nox_ppmv": float(row["nox_ppmv"]),
        "dust_mg_Nm3": float(row["dust_mg_Nm3"]),
    })


def row_to_cost_config(row: pd.Series | dict[str, Any], base_cost: CostConfig) -> CostConfig:
    row = dict(row)
    return replace(
        base_cost,
        electricity_cost_per_kwh=float(row["electricity_price"]),
        recovery_target=float(row["capture_target"]),
    )


def row_to_process_constants(row: pd.Series | dict[str, Any], base_constants: ProcessConstants) -> ProcessConstants:
    row = dict(row)
    process_temperature_k = 313.15 + 0.25 * (float(row["ambient_temp"]) - 20.0)
    return replace(
        base_constants,
        temperature_k=float(np.clip(process_temperature_k, 308.15, 318.15)),
    )


def auxiliary_heat_cost_per_tco2(decision_eval: pd.Series | dict[str, Any], row: pd.Series | dict[str, Any]) -> float:
    decision_eval = dict(decision_eval)
    row = dict(row)
    deactivation = float(decision_eval["deactivation_index"])
    capture_target = float(row["capture_target"])
    ambient_temp = float(row["ambient_temp"])
    heat_price = float(row["heat_price"])

    specific_heat_mj_per_tco2 = (
        0.18
        + 0.35 * deactivation
        + 0.25 * max(0.0, capture_target - 0.70)
        + 0.02 * max(0.0, 20.0 - ambient_temp)
    )
    return specific_heat_mj_per_tco2 * heat_price


def _evaluate_single_decision(
    optimizer: PSAEconomicsOptimizer,
    decision: dict[str, float],
    row: pd.Series,
    base_cost: CostConfig,
    base_constants: ProcessConstants,
) -> pd.Series:
    fixed_input = row_to_fixed_input(row)
    scenario_cost = row_to_cost_config(row, base_cost)
    scenario_constants = row_to_process_constants(row, base_constants)

    old_cost = optimizer.cost
    old_constants = optimizer.constants
    optimizer.cost = scenario_cost
    optimizer.constants = scenario_constants
    try:
        evaluated = optimizer.evaluate_candidates(pd.DataFrame([decision]), fixed_input).iloc[0].copy()
    finally:
        optimizer.cost = old_cost
        optimizer.constants = old_constants

    evaluated["aux_heat_cost_per_tco2"] = auxiliary_heat_cost_per_tco2(evaluated, row)
    evaluated["adjusted_total_cost_per_tco2"] = evaluated["total_cost_per_tco2"] + evaluated["aux_heat_cost_per_tco2"]

    inlet_co2_tph = float(row["co2_flow_rate"]) * float(row["co2_concentration"]) * CO2_MW_KG_PER_MOL
    captured_tph = min(inlet_co2_tph * float(row["capture_target"]), float(row["max_capacity"]))
    interval_hours = 0.25
    evaluated["inlet_co2_tph"] = inlet_co2_tph
    evaluated["captured_tph"] = captured_tph
    evaluated["captured_tons_interval"] = captured_tph * interval_hours
    evaluated["interval_total_cost"] = evaluated["adjusted_total_cost_per_tco2"] * evaluated["captured_tons_interval"]
    return evaluated


def optimize_row_decision(
    optimizer: PSAEconomicsOptimizer,
    row: pd.Series,
    base_cost: CostConfig,
    base_constants: ProcessConstants,
    n_samples: int,
    local_steps: int,
    elite_size: int,
    seed: int,
) -> dict[str, float]:
    fixed_input = row_to_fixed_input(row)
    scenario_cost = row_to_cost_config(row, base_cost)
    scenario_constants = row_to_process_constants(row, base_constants)

    old_cost = optimizer.cost
    old_constants = optimizer.constants
    optimizer.cost = scenario_cost
    optimizer.constants = scenario_constants
    try:
        best, _ = optimizer.optimize_for_fixed_input(
            fixed_input=fixed_input,
            n_samples=n_samples,
            local_steps=local_steps,
            elite_size=elite_size,
            seed=seed,
        )
    finally:
        optimizer.cost = old_cost
        optimizer.constants = old_constants

    return {col: float(best[col]) for col in DECISION_COLS}


def mean_operating_row(scenario_df: pd.DataFrame) -> pd.Series:
    numeric_mean = scenario_df.mean(numeric_only=True)
    numeric_mean["timestamp"] = scenario_df["timestamp"].iloc[0]
    numeric_mean["scenario"] = scenario_df["scenario"].iloc[0]
    return numeric_mean


def run_single_scenario_experiment(
    optimizer: PSAEconomicsOptimizer,
    scenario_df: pd.DataFrame,
    base_cost: CostConfig,
    base_constants: ProcessConstants,
    n_samples: int = 200,
    local_steps: int = 2,
    elite_size: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    scenario_df = scenario_df.reset_index(drop=True).copy()
    first_row = scenario_df.iloc[0]
    mean_row = mean_operating_row(scenario_df)

    first_decision = optimize_row_decision(
        optimizer=optimizer,
        row=first_row,
        base_cost=base_cost,
        base_constants=base_constants,
        n_samples=n_samples,
        local_steps=local_steps,
        elite_size=elite_size,
        seed=seed + 10_000,
    )
    average_decision = optimize_row_decision(
        optimizer=optimizer,
        row=mean_row,
        base_cost=base_cost,
        base_constants=base_constants,
        n_samples=n_samples,
        local_steps=local_steps,
        elite_size=elite_size,
        seed=seed + 20_000,
    )

    rows: list[dict[str, Any]] = []
    for idx, row in scenario_df.iterrows():
        dynamic_decision = optimize_row_decision(
            optimizer=optimizer,
            row=row,
            base_cost=base_cost,
            base_constants=base_constants,
            n_samples=n_samples,
            local_steps=local_steps,
            elite_size=elite_size,
            seed=seed + idx,
        )
        strategy_decisions = {
            "dynamic_reoptimized": dynamic_decision,
            "static_first_step": first_decision,
            "static_average_condition": average_decision,
        }

        for strategy_name, decision in strategy_decisions.items():
            evaluated = _evaluate_single_decision(
                optimizer=optimizer,
                decision=decision,
                row=row,
                base_cost=base_cost,
                base_constants=base_constants,
            )
            result_row = {
                "timestamp": row["timestamp"],
                "scenario": row["scenario"],
                "strategy": strategy_name,
                "co2_flow_rate": float(row["co2_flow_rate"]),
                "co2_concentration": float(row["co2_concentration"]),
                "electricity_price": float(row["electricity_price"]),
                "heat_price": float(row["heat_price"]),
                "ambient_temp": float(row["ambient_temp"]),
                "capture_target": float(row["capture_target"]),
                "max_capacity": float(row["max_capacity"]),
            }
            result_row.update({col: decision[col] for col in DECISION_COLS})
            result_row.update(evaluated.to_dict())
            rows.append(result_row)

    results = pd.DataFrame(rows).sort_values(["strategy", "timestamp"]).reset_index(drop=True)
    results["cumulative_total_cost"] = results.groupby("strategy")["interval_total_cost"].cumsum()
    results["cumulative_captured_tons"] = results.groupby("strategy")["captured_tons_interval"].cumsum()
    results["cumulative_cost_per_tco2"] = (
        results["cumulative_total_cost"] / results["cumulative_captured_tons"].clip(lower=1e-9)
    )
    return results


def run_all_scenarios_experiment(
    optimizer: PSAEconomicsOptimizer,
    scenarios: dict[str, pd.DataFrame],
    base_cost: CostConfig,
    base_constants: ProcessConstants,
    n_samples: int = 200,
    local_steps: int = 2,
    elite_size: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    frames = []
    for offset, (scenario_name, scenario_df) in enumerate(scenarios.items()):
        frame = run_single_scenario_experiment(
            optimizer=optimizer,
            scenario_df=scenario_df,
            base_cost=base_cost,
            base_constants=base_constants,
            n_samples=n_samples,
            local_steps=local_steps,
            elite_size=elite_size,
            seed=seed + offset * 100_000,
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_summary_table(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby(["scenario", "strategy"], as_index=False)
        .agg(
            avg_cost_per_tco2=("adjusted_total_cost_per_tco2", "mean"),
            final_cumulative_cost_per_tco2=("cumulative_cost_per_tco2", "last"),
            total_captured_tons=("captured_tons_interval", "sum"),
            total_cost=("interval_total_cost", "sum"),
        )
        .sort_values(["scenario", "final_cumulative_cost_per_tco2"])
        .reset_index(drop=True)
    )
    return summary
