from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


R = 8.3145
CO2_MW_KG_PER_MOL = 44.01e-3
N2_MW_KG_PER_MOL = 28.0134e-3

DECISION_COLS = [
    "tADS",
    "PL",
    "v0",
    "t_press",
    "t_depress",
]

GAS_STATE_COLS = [
    "co2_mol_frac",
    "h2o_ppmv",
    "sox_ppmv",
    "nox_ppmv",
    "dust_mg_Nm3",
]

FACTOR_COLS = [
    "capacity_factor",
    "affinity_factor_co2",
    "mtc_factor",
    "dax_factor",
    "deactivation_index",
]

FEATURE_COLS = DECISION_COLS + GAS_STATE_COLS + FACTOR_COLS
TARGET_COLS = ["CO2_purity", "CO2_recovery"]

DECISION_BOUNDS = {
    "tADS": (70.0, 180.0),
    "PL": (0.12, 0.35),
    "v0": (0.010, 0.032),
    "t_press": (20.0, 70.0),
    "t_depress": (25.0, 75.0),
}


def clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def lhs_sampling(bounds: dict[str, tuple[float, float]], n_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    keys = list(bounds.keys())
    samples = np.zeros((n_samples, len(keys)))

    for j, key in enumerate(keys):
        low, high = bounds[key]
        cut = np.linspace(0.0, 1.0, n_samples + 1)
        u = rng.uniform(cut[:-1], cut[1:])
        rng.shuffle(u)
        samples[:, j] = low + u * (high - low)

    return pd.DataFrame(samples, columns=keys)


def compute_pretreatment_factors(
    h2o_ppmv: float,
    sox_ppmv: float,
    nox_ppmv: float,
    dust_mg_Nm3: float,
) -> dict[str, float]:
    h2o_ppmv = clip(h2o_ppmv, 1.0, 100.0)
    sox_ppmv = clip(sox_ppmv, 0.0, 1.0)
    nox_ppmv = clip(nox_ppmv, 0.0, 10.0)
    dust_mg_Nm3 = clip(dust_mg_Nm3, 0.0, 0.1)

    h2o_load = np.log10(h2o_ppmv + 1.0) / np.log10(101.0)
    sox_load = clip(sox_ppmv / 1.0, 0.0, 1.0)
    nox_load = clip(nox_ppmv / 10.0, 0.0, 1.0)
    dust_load = clip(dust_mg_Nm3 / 0.1, 0.0, 1.0)

    deactivation_index = clip(
        0.35 * h2o_load + 0.15 * sox_load + 0.10 * nox_load + 0.05 * dust_load,
        0.0,
        0.65,
    )
    capacity_factor = clip(
        1.0 - (0.02 + 0.08 * h2o_load + 0.05 * sox_load + 0.025 * nox_load + 0.015 * dust_load),
        0.82,
        0.98,
    )
    affinity_factor_co2 = clip(
        1.0 - (0.015 + 0.045 * h2o_load + 0.045 * sox_load + 0.015 * nox_load),
        0.84,
        0.99,
    )
    mtc_factor = clip(
        1.0 - (0.025 * h2o_load + 0.025 * sox_load + 0.015 * dust_load),
        0.88,
        0.99,
    )
    dax_factor = clip(
        1.0 + 0.045 * dust_load + 0.020 * nox_load + 0.010 * h2o_load,
        1.0,
        1.08,
    )

    return {
        "capacity_factor": capacity_factor,
        "affinity_factor_co2": affinity_factor_co2,
        "mtc_factor": mtc_factor,
        "dax_factor": dax_factor,
        "deactivation_index": deactivation_index,
    }


def compute_effective_feed_fractions(
    co2_mol_frac: float,
    h2o_ppmv: float,
    sox_ppmv: float,
    nox_ppmv: float,
) -> tuple[float, float]:
    co2_mol_frac = clip(co2_mol_frac, 0.15, 0.25)
    trace_mol_frac = h2o_ppmv * 1e-6 + sox_ppmv * 1e-6 + nox_ppmv * 1e-6
    n2_major = max(1.0 - co2_mol_frac - trace_mol_frac, 1e-8)
    major_total = co2_mol_frac + n2_major
    y_co2 = co2_mol_frac / major_total
    y_n2 = n2_major / major_total
    return y_co2, y_n2


def normalize_fixed_input(fixed_input: dict[str, float]) -> dict[str, float]:
    normalized = dict(fixed_input)
    missing_gas_cols = [col for col in GAS_STATE_COLS if col not in normalized]
    if missing_gas_cols:
        raise ValueError(f"Missing fixed-input gas-state columns: {missing_gas_cols}")

    needs_factor_fill = any(col not in normalized or pd.isna(normalized[col]) for col in FACTOR_COLS)
    if needs_factor_fill:
        normalized.update(
            compute_pretreatment_factors(
                normalized["h2o_ppmv"],
                normalized["sox_ppmv"],
                normalized["nox_ppmv"],
                normalized["dust_mg_Nm3"],
            )
        )

    missing_after_fill = [col for col in GAS_STATE_COLS + FACTOR_COLS if col not in normalized]
    if missing_after_fill:
        raise ValueError(f"Missing fixed-input columns: {missing_after_fill}")
    return normalized


def build_feature_row(decision: dict[str, float], fixed_input: dict[str, float]) -> dict[str, float]:
    normalized_fixed = normalize_fixed_input(fixed_input)
    return {
        **decision,
        **{col: normalized_fixed[col] for col in GAS_STATE_COLS + FACTOR_COLS},
    }


@dataclass
class ProcessConstants:
    temperature_k: float = 313.15
    high_pressure_bar: float = 1.2
    bed_length_m: float = 1.0
    area_m2: float = 0.01
    bed_void_fraction: float = 0.39
    particle_diameter_m: float = 1.2e-3
    desorb_time_factor: float = 0.90
    min_desorb_time_s: float = 45.0
    operating_hours_per_year: float = 8000.0


@dataclass
class CostConfig:
    electricity_cost_per_kwh: float = 110.0
    carbon_price_per_tco2: float = 50000.0
    co2_credit_per_tco2: float = 0.0
    annualized_capex_per_m3_bed: float = 200_000.0
    annual_fixed_om_per_m3_bed: float = 40_000.0
    annual_adsorbent_budget_per_m3_bed: float = 80_000.0
    purity_target: float = 0.90
    recovery_target: float = 0.90
    purity_penalty_weight: float = 200_000.0
    recovery_penalty_weight: float = 250_000.0
    uncertainty_penalty_weight: float = 0.0
    min_capture_tpy_per_bed: float = 0.10
    vacuum_efficiency: float = 0.65
    blower_efficiency: float = 0.72
    compressor_efficiency: float = 0.75
    gas_gamma: float = 1.30
    vacuum_discharge_bar: float = 1.01325
    product_compression_outlet_bar: float = 110.0


class PSAEconomicsOptimizer:
    def __init__(
        self,
        train_df: pd.DataFrame,
        purity_model_path: Path,
        recovery_model_path: Path,
        constants: ProcessConstants | None = None,
        cost: CostConfig | None = None,
    ) -> None:
        self.constants = constants or ProcessConstants()
        self.cost = cost or CostConfig()
        self.train_df = self._prepare_training_data(train_df)
        self.x_scaler, self.purity_y_scaler, self.recovery_y_scaler = self._fit_scalers(self.train_df)
        self.purity_model = keras.models.load_model(purity_model_path)
        self.recovery_model = keras.models.load_model(recovery_model_path)
        self.feature_ranges = {
            col: (float(self.train_df[col].min()), float(self.train_df[col].max()))
            for col in FEATURE_COLS
        }

    @staticmethod
    def _prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
        if "status" in df.columns:
            df = df[df["status"].eq("ok")].copy()
        elif "valid_physics" in df.columns:
            df = df[df["valid_physics"].fillna(False)].copy()
        return df.dropna(subset=FEATURE_COLS + TARGET_COLS).reset_index(drop=True)

    @staticmethod
    def _fit_scalers(train_df: pd.DataFrame) -> tuple[StandardScaler, StandardScaler, StandardScaler]:
        x_scaler = StandardScaler()
        purity_y_scaler = StandardScaler()
        recovery_y_scaler = StandardScaler()

        x_scaler.fit(train_df[FEATURE_COLS].to_numpy(dtype=np.float32))
        purity_y_scaler.fit(train_df[["CO2_purity"]].to_numpy(dtype=np.float32))
        recovery_y_scaler.fit(train_df[["CO2_recovery"]].to_numpy(dtype=np.float32))
        return x_scaler, purity_y_scaler, recovery_y_scaler

    def predict_performance(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        x_scaled = self.x_scaler.transform(feature_df[FEATURE_COLS].to_numpy(dtype=np.float32)).astype(np.float32)

        purity_scaled = self.purity_model.predict(x_scaled, verbose=0)
        recovery_scaled = self.recovery_model.predict(x_scaled, verbose=0)

        purity = self.purity_y_scaler.inverse_transform(np.asarray(purity_scaled).reshape(-1, 1)).ravel()
        recovery = self.recovery_y_scaler.inverse_transform(np.asarray(recovery_scaled).reshape(-1, 1)).ravel()

        pred = pd.DataFrame(index=feature_df.index)
        pred["pred_CO2_purity"] = np.clip(purity, 1e-4, 0.999)
        pred["pred_CO2_recovery"] = np.clip(recovery, 1e-4, 0.999)
        pred["pred_CO2_purity_std"] = 0.0
        pred["pred_CO2_recovery_std"] = 0.0
        return pred

    def evaluate_candidates(self, candidates: pd.DataFrame, fixed_input: dict[str, float]) -> pd.DataFrame:
        feature_rows = [
            build_feature_row(
                row._asdict() if hasattr(row, "_asdict") else row.to_dict(),
                fixed_input,
            )
            for _, row in candidates.iterrows()
        ]
        feature_df = pd.DataFrame(feature_rows)[FEATURE_COLS]
        predictions = self.predict_performance(feature_df)
        metrics = self._compute_economic_metrics(candidates.reset_index(drop=True), fixed_input, predictions)
        factor_df = feature_df[FACTOR_COLS].reset_index(drop=True)
        return pd.concat([candidates.reset_index(drop=True), factor_df, predictions, metrics], axis=1)

    def _compute_economic_metrics(
        self,
        candidates: pd.DataFrame,
        fixed_input: dict[str, float],
        predictions: pd.DataFrame,
    ) -> pd.DataFrame:
        fixed = normalize_fixed_input(fixed_input)
        y_co2, y_n2 = compute_effective_feed_fractions(
            fixed["co2_mol_frac"],
            fixed["h2o_ppmv"],
            fixed["sox_ppmv"],
            fixed["nox_ppmv"],
        )

        feed = self._feed_moles(candidates, y_co2, y_n2)
        cycle = self._cycle_times(candidates)
        product = self._product_moles(feed, predictions)
        energy = self._energy_cost(candidates, fixed, product)
        bed_volume_m3 = self.constants.bed_length_m * self.constants.area_m2

        co2_product_kg = product["n_co2_product_mol"] * CO2_MW_KG_PER_MOL
        co2_product_t = np.maximum(co2_product_kg / 1000.0, 1e-9)
        annual_capture_t_per_bed = np.maximum(
            co2_product_t * (self.constants.operating_hours_per_year * 3600.0 / cycle["cycle_time_s"]),
            self.cost.min_capture_tpy_per_bed,
        )

        capex_cost = self.cost.annualized_capex_per_m3_bed * bed_volume_m3 / annual_capture_t_per_bed
        fixed_om_cost = self.cost.annual_fixed_om_per_m3_bed * bed_volume_m3 / annual_capture_t_per_bed
        adsorbent_cost = (
            self.cost.annual_adsorbent_budget_per_m3_bed
            * bed_volume_m3
            * (0.35 + 1.30 * fixed["deactivation_index"])
            / annual_capture_t_per_bed
        )

        unrecovered_t = np.maximum(
            (feed["n_co2_feed_mol"] - product["n_co2_product_mol"]) * CO2_MW_KG_PER_MOL / 1000.0,
            0.0,
        )
        unrecovered_cost = self.cost.carbon_price_per_tco2 * unrecovered_t / co2_product_t
        credit = self.cost.co2_credit_per_tco2

        purity_shortfall = np.maximum(self.cost.purity_target - predictions["pred_CO2_purity"].to_numpy(), 0.0)
        recovery_shortfall = np.maximum(self.cost.recovery_target - predictions["pred_CO2_recovery"].to_numpy(), 0.0)
        purity_penalty = self.cost.purity_penalty_weight * purity_shortfall ** 2
        recovery_penalty = self.cost.recovery_penalty_weight * recovery_shortfall ** 2
        uncertainty_penalty = self.cost.uncertainty_penalty_weight * (
            predictions["pred_CO2_purity_std"].to_numpy() + predictions["pred_CO2_recovery_std"].to_numpy()
        )
        domain_penalty = 50_000.0 * self._domain_distance(candidates)

        total_cost = (
            energy["specific_energy_cost_per_tco2"]
            + capex_cost
            + fixed_om_cost
            + adsorbent_cost
            + unrecovered_cost
            + purity_penalty
            + recovery_penalty
            + uncertainty_penalty
            + domain_penalty
            - credit
        )

        return pd.DataFrame({
            "cycle_time_s": cycle["cycle_time_s"],
            "n_CO2_feed_calc": feed["n_co2_feed_mol"],
            "n_N2_feed_calc": feed["n_n2_feed_mol"],
            "n_CO2_product_calc": product["n_co2_product_mol"],
            "n_N2_product_calc": product["n_n2_product_mol"],
            "co2_product_kg_per_cycle": co2_product_kg,
            "annual_capture_t_per_bed": annual_capture_t_per_bed,
            "vacuum_energy_kwh_per_cycle": energy["vacuum_energy_kwh_per_cycle"],
            "blower_energy_kwh_per_cycle": energy["blower_energy_kwh_per_cycle"],
            "compression_energy_kwh_per_cycle": energy["compression_energy_kwh_per_cycle"],
            "specific_energy_kwh_per_tco2": energy["specific_energy_kwh_per_tco2"],
            "energy_cost_per_tco2": energy["specific_energy_cost_per_tco2"],
            "capex_proxy_cost_per_tco2": capex_cost,
            "fixed_om_cost_per_tco2": fixed_om_cost,
            "adsorbent_cost_per_tco2": adsorbent_cost,
            "unrecovered_cost_per_tco2": unrecovered_cost,
            "purity_penalty_per_tco2": purity_penalty,
            "recovery_penalty_per_tco2": recovery_penalty,
            "uncertainty_penalty_per_tco2": uncertainty_penalty,
            "domain_penalty_per_tco2": domain_penalty,
            "total_cost_per_tco2": total_cost,
        })

    def _domain_distance(self, candidates: pd.DataFrame) -> np.ndarray:
        feature_rows = candidates[DECISION_COLS].copy()
        distances = np.zeros(len(feature_rows))
        for col in DECISION_BOUNDS:
            train_low, train_high = self.feature_ranges[col]
            values = feature_rows[col].to_numpy()
            low_gap = np.maximum(train_low - values, 0.0)
            high_gap = np.maximum(values - train_high, 0.0)
            distances += (low_gap + high_gap) / max(train_high - train_low, 1e-9)
        return distances

    def _feed_moles(self, candidates: pd.DataFrame, y_co2: float, y_n2: float) -> dict[str, np.ndarray]:
        c_feed = self.constants.high_pressure_bar * 1e5 / (R * self.constants.temperature_k)
        q_feed = candidates["v0"].to_numpy() * self.constants.area_m2 * self.constants.bed_void_fraction
        t_ads = candidates["tADS"].to_numpy()
        return {
            "n_co2_feed_mol": c_feed * q_feed * y_co2 * t_ads,
            "n_n2_feed_mol": c_feed * q_feed * y_n2 * t_ads,
        }

    def _cycle_times(self, candidates: pd.DataFrame) -> dict[str, np.ndarray]:
        t_ads = candidates["tADS"].to_numpy()
        t_desorb = np.maximum(t_ads * self.constants.desorb_time_factor, self.constants.min_desorb_time_s)
        cycle_time = candidates["t_press"].to_numpy() + t_ads + candidates["t_depress"].to_numpy() + t_desorb
        return {
            "t_desorb_s": t_desorb,
            "cycle_time_s": cycle_time,
        }

    @staticmethod
    def _product_moles(feed: dict[str, np.ndarray], predictions: pd.DataFrame) -> dict[str, np.ndarray]:
        purity = predictions["pred_CO2_purity"].to_numpy()
        recovery = predictions["pred_CO2_recovery"].to_numpy()
        n_co2_product = np.maximum(recovery * feed["n_co2_feed_mol"], 1e-9)
        n_n2_product = np.maximum(n_co2_product * (1.0 - purity) / np.maximum(purity, 1e-6), 1e-9)
        return {
            "n_co2_product_mol": n_co2_product,
            "n_n2_product_mol": n_n2_product,
        }

    def _energy_cost(
        self,
        candidates: pd.DataFrame,
        fixed_input: dict[str, float],
        product: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        total_product_mol = product["n_co2_product_mol"] + product["n_n2_product_mol"]
        vacuum_energy = self._isentropic_energy_kwh(
            n_mol=total_product_mol,
            p_in_bar=np.maximum(candidates["PL"].to_numpy(), 1e-6),
            p_out_bar=self.cost.vacuum_discharge_bar,
            efficiency=self.cost.vacuum_efficiency,
        )
        blower_energy = self._blower_energy_kwh(candidates, fixed_input)
        compression_energy = self._isentropic_energy_kwh(
            n_mol=total_product_mol,
            p_in_bar=np.full(len(candidates), self.cost.vacuum_discharge_bar),
            p_out_bar=self.cost.product_compression_outlet_bar,
            efficiency=self.cost.compressor_efficiency,
        )

        co2_product_t = np.maximum(product["n_co2_product_mol"] * CO2_MW_KG_PER_MOL / 1000.0, 1e-9)
        specific_energy = (vacuum_energy + blower_energy + compression_energy) / co2_product_t
        return {
            "vacuum_energy_kwh_per_cycle": vacuum_energy,
            "blower_energy_kwh_per_cycle": blower_energy,
            "compression_energy_kwh_per_cycle": compression_energy,
            "specific_energy_kwh_per_tco2": specific_energy,
            "specific_energy_cost_per_tco2": specific_energy * self.cost.electricity_cost_per_kwh,
        }

    def _blower_energy_kwh(self, candidates: pd.DataFrame, fixed_input: dict[str, float]) -> np.ndarray:
        fixed = normalize_fixed_input(fixed_input)
        y_co2, y_n2 = compute_effective_feed_fractions(
            fixed["co2_mol_frac"],
            fixed["h2o_ppmv"],
            fixed["sox_ppmv"],
            fixed["nox_ppmv"],
        )
        mw_mix = y_co2 * CO2_MW_KG_PER_MOL + y_n2 * N2_MW_KG_PER_MOL
        rho = self.constants.high_pressure_bar * 1e5 * mw_mix / (R * self.constants.temperature_k)
        mu_mix = y_co2 * 1.48e-5 + y_n2 * 1.76e-5
        eps = self.constants.bed_void_fraction
        dp = self.constants.particle_diameter_m
        v0 = candidates["v0"].to_numpy()
        delta_p = self.constants.bed_length_m * (
            (150.0 * mu_mix * (1.0 - eps) ** 2 / (dp ** 2 * eps ** 3)) * v0
            + (1.75 * rho * (1.0 - eps) / (dp * eps ** 3)) * v0 ** 2
        )
        q_feed = v0 * self.constants.area_m2 * self.constants.bed_void_fraction
        work_j = q_feed * delta_p * candidates["tADS"].to_numpy() / self.cost.blower_efficiency
        return work_j / 3.6e6

    def _isentropic_energy_kwh(
        self,
        n_mol: np.ndarray,
        p_in_bar: np.ndarray,
        p_out_bar: float,
        efficiency: float,
    ) -> np.ndarray:
        gamma = self.cost.gas_gamma
        pressure_ratio = np.maximum(p_out_bar / np.maximum(p_in_bar, 1e-8), 1.0)
        factor = gamma / (gamma - 1.0) * (pressure_ratio ** ((gamma - 1.0) / gamma) - 1.0)
        work_j = n_mol * R * self.constants.temperature_k * factor / max(efficiency, 1e-6)
        return work_j / 3.6e6

    def optimize_for_fixed_input(
        self,
        fixed_input: dict[str, float],
        n_samples: int = 4000,
        local_steps: int = 6,
        elite_size: int = 12,
        seed: int = 42,
    ) -> tuple[pd.Series, pd.DataFrame]:
        normalized_fixed = normalize_fixed_input(fixed_input)
        candidates = lhs_sampling(DECISION_BOUNDS, n_samples=n_samples, seed=seed)
        evaluated = self.evaluate_candidates(candidates, normalized_fixed)

        bounds_span = {k: v[1] - v[0] for k, v in DECISION_BOUNDS.items()}
        rng = np.random.default_rng(seed + 1000)
        for step in range(local_steps):
            elite = evaluated.nsmallest(elite_size, "total_cost_per_tco2")[DECISION_COLS]
            radius_scale = 0.25 / (step + 1.0)
            local_candidates: list[dict[str, float]] = []
            for _, row in elite.iterrows():
                for _ in range(max(40, n_samples // max(elite_size, 1) // 4)):
                    trial = {}
                    for key, (low, high) in DECISION_BOUNDS.items():
                        noise = rng.normal(0.0, bounds_span[key] * radius_scale)
                        trial[key] = clip(float(row[key] + noise), low, high)
                    local_candidates.append(trial)
            local_df = pd.DataFrame(local_candidates)
            local_eval = self.evaluate_candidates(local_df, normalized_fixed)
            evaluated = pd.concat([evaluated, local_eval], ignore_index=True)

        best = evaluated.nsmallest(1, "total_cost_per_tco2").iloc[0]
        return best, evaluated.sort_values("total_cost_per_tco2").reset_index(drop=True)


def load_training_data(train_csv: Path) -> pd.DataFrame:
    return pd.read_csv(train_csv)


def load_fixed_inputs(fixed_inputs_csv: Path | None, fallback_df: pd.DataFrame) -> pd.DataFrame:
    required_or_derivable = GAS_STATE_COLS + FACTOR_COLS

    if fixed_inputs_csv is not None:
        fixed_inputs = pd.read_csv(fixed_inputs_csv)
        missing_gas = [col for col in GAS_STATE_COLS if col not in fixed_inputs.columns]
        if missing_gas:
            raise ValueError(f"Missing fixed-input gas-state columns: {missing_gas}")
        for col in FACTOR_COLS:
            if col not in fixed_inputs.columns:
                fixed_inputs[col] = np.nan
        records = [normalize_fixed_input(row.to_dict()) for _, row in fixed_inputs.iterrows()]
        return pd.DataFrame(records)[required_or_derivable]

    dedup = fallback_df[required_or_derivable].drop_duplicates().head(5).reset_index(drop=True)
    records = [normalize_fixed_input(row.to_dict()) for _, row in dedup.iterrows()]
    return pd.DataFrame(records)[required_or_derivable]


def summarize_result(case_id: int, fixed_input: pd.Series, best: pd.Series) -> dict[str, float | int]:
    summary = {
        "case_id": case_id,
        **fixed_input.to_dict(),
        "opt_tADS": best["tADS"],
        "opt_PL": best["PL"],
        "opt_v0": best["v0"],
        "opt_t_press": best["t_press"],
        "opt_t_depress": best["t_depress"],
        "pred_CO2_purity": best["pred_CO2_purity"],
        "pred_CO2_recovery": best["pred_CO2_recovery"],
        "n_CO2_product_calc": best["n_CO2_product_calc"],
        "n_N2_product_calc": best["n_N2_product_calc"],
        "specific_energy_kwh_per_tco2": best["specific_energy_kwh_per_tco2"],
        "energy_cost_per_tco2": best["energy_cost_per_tco2"],
        "capex_proxy_cost_per_tco2": best["capex_proxy_cost_per_tco2"],
        "fixed_om_cost_per_tco2": best["fixed_om_cost_per_tco2"],
        "adsorbent_cost_per_tco2": best["adsorbent_cost_per_tco2"],
        "unrecovered_cost_per_tco2": best["unrecovered_cost_per_tco2"],
        "purity_penalty_per_tco2": best["purity_penalty_per_tco2"],
        "recovery_penalty_per_tco2": best["recovery_penalty_per_tco2"],
        "uncertainty_penalty_per_tco2": best["uncertainty_penalty_per_tco2"],
        "domain_penalty_per_tco2": best["domain_penalty_per_tco2"],
        "total_cost_per_tco2": best["total_cost_per_tco2"],
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DNN-surrogate techno-economic optimization for PSA/VSA operation.")
    parser.add_argument("--train-csv", type=Path, default=Path("../model/output.csv"))
    parser.add_argument("--fixed-inputs-csv", type=Path, default=None)
    parser.add_argument("--purity-model", type=Path, default=Path("../model/CO2_purity_dnn.keras"))
    parser.add_argument("--recovery-model", type=Path, default=Path("../model/CO2_recovery_dnn.keras"))
    parser.add_argument("--results-csv", type=Path, default=Path("../model/optimization_results.csv"))
    parser.add_argument("--all-candidates-csv", type=Path, default=Path("../model/optimization_candidates.csv"))
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--local-steps", type=int, default=6)
    parser.add_argument("--elite-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df = load_training_data(args.train_csv)
    optimizer = PSAEconomicsOptimizer(
        train_df=train_df,
        purity_model_path=args.purity_model,
        recovery_model_path=args.recovery_model,
    )
    fixed_inputs = load_fixed_inputs(args.fixed_inputs_csv, optimizer.train_df)

    result_rows: list[dict[str, float | int]] = []
    candidate_frames: list[pd.DataFrame] = []
    for case_id, fixed_input in fixed_inputs.iterrows():
        best, evaluated = optimizer.optimize_for_fixed_input(
            fixed_input=fixed_input.to_dict(),
            n_samples=args.n_samples,
            local_steps=args.local_steps,
            elite_size=args.elite_size,
            seed=args.seed + case_id,
        )
        result_rows.append(summarize_result(case_id + 1, fixed_input, best))
        evaluated = evaluated.copy()
        evaluated.insert(0, "case_id", case_id + 1)
        for col in GAS_STATE_COLS + FACTOR_COLS:
            evaluated[col] = fixed_input[col]
        candidate_frames.append(evaluated)

    result_df = pd.DataFrame(result_rows)
    all_candidates_df = pd.concat(candidate_frames, ignore_index=True)

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.results_csv, index=False)
    all_candidates_df.to_csv(args.all_candidates_csv, index=False)

    print(f"Saved optimization summary: {args.results_csv}")
    print(f"Saved evaluated candidates: {args.all_candidates_csv}")
    print(result_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
