import warnings

import numpy as np
import pandas as pd

from pyapep.simsep import column


R = 8.3145


def lhs_sampling(bounds, n_samples=30, seed=100):
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


def clip(value, low, high):
    return float(np.clip(value, low, high))


class MgMOF74PSA:
    """
    Mg-MOF-74 PSA/VSA-style simulator with pretreated flue-gas effects.

    Literature basis used for conservative parameterization:
    - CO2 uptake near 8.6 mmol/g at 298 K and 1 bar for Mg-MOF-74.
    - Adsorption enthalpy magnitude around 47 kJ/mol for CO2 and 21 kJ/mol for N2.
    - Residual H2O at open Mg sites lowers effective CO2 binding.
    - SOx binds preferentially to Mg sites and causes persistent site blocking.
    - NOx is treated as a weaker but non-zero deactivation source.
    """

    def __init__(
        self,
        L=1.0,
        A_cros=0.01,
        N_node=31,
        T=313.15,
        PH=1.2,
    ):
        self.L = L
        self.A = A_cros
        self.N = N_node
        self.T = T
        self.PH = PH
        self.T_ref = 313.15

        self.eps = 0.39
        self.dp = 1.2e-3
        self.rho_s = 880.0
        self.bed_volume = self.L * self.A
        self.void_volume = self.bed_volume * self.eps
        self.solid_mass = self.bed_volume * (1.0 - self.eps) * self.rho_s

        self.M = [0.04401, 0.0280134]
        self.mu = [1.48e-5, 1.76e-5]
        self.base_D_ax = np.array([8.5e-6, 8.5e-6], dtype=float)
        self.base_k_mtc = np.array([1.25e-3, 7.50e-4], dtype=float)
        self.a_surf = 6.0 / self.dp

        self.dH = [47e3, 21e3]
        self.Cp_s = 920.0
        self.Cp_g = [37.1, 29.1]
        self.h_heat = 75.0

        self.base_qsat = {
            "co2": 8.80,
            "n2": 0.55,
        }
        self.base_b0 = {
            "co2": 5.5,
            "n2": 0.080,
        }

        self.set_pretreated_feed(
            co2_mol_frac=0.18,
            h2o_ppmv=10.0,
            sox_ppmv=0.20,
            nox_ppmv=3.0,
            dust_mg_Nm3=0.02,
        )

    def set_pretreated_feed(
        self,
        co2_mol_frac,
        h2o_ppmv,
        sox_ppmv,
        nox_ppmv,
        dust_mg_Nm3,
    ):
        self.h2o_ppmv = clip(h2o_ppmv, 1.0, 100.0)
        self.sox_ppmv = clip(sox_ppmv, 0.0, 1.0)
        self.nox_ppmv = clip(nox_ppmv, 0.0, 10.0)
        self.dust_mg_Nm3 = clip(dust_mg_Nm3, 0.0, 0.1)

        self.co2_mol_frac = clip(co2_mol_frac, 0.15, 0.25)
        trace_mol_frac = (
            self.h2o_ppmv * 1e-6
            + self.sox_ppmv * 1e-6
            + self.nox_ppmv * 1e-6
        )
        n2_major = max(1.0 - self.co2_mol_frac - trace_mol_frac, 1e-8)
        major_total = self.co2_mol_frac + n2_major

        self.y_feed = np.array(
            [
                self.co2_mol_frac / major_total,
                n2_major / major_total,
            ],
            dtype=float,
        )
        self.n2_mol_frac_effective = float(self.y_feed[1])

        factors = self.compute_pretreatment_factors(
            self.h2o_ppmv,
            self.sox_ppmv,
            self.nox_ppmv,
            self.dust_mg_Nm3,
        )

        self.capacity_factor = factors["capacity_factor"]
        self.affinity_factor_co2 = factors["affinity_factor_co2"]
        self.mtc_factor = factors["mtc_factor"]
        self.dax_factor = factors["dax_factor"]
        self.deactivation_index = factors["deactivation_index"]

        self.D_ax = list(self.base_D_ax * self.dax_factor)
        self.k_mtc = list(
            self.base_k_mtc * np.array([self.mtc_factor, self.mtc_factor * 0.97])
        )

    def compute_pretreatment_factors(
        self,
        h2o_ppmv,
        sox_ppmv,
        nox_ppmv,
        dust_mg_Nm3,
    ):
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
            1.0
            - (0.02 + 0.08 * h2o_load + 0.05 * sox_load + 0.025 * nox_load + 0.015 * dust_load),
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

    def isotherm(self, P, T):
        Pco2 = np.asarray(P[0], dtype=float)
        Pn2 = np.asarray(P[1], dtype=float)
        T = np.asarray(T, dtype=float)

        qsat_co2 = self.base_qsat["co2"] * self.capacity_factor
        qsat_n2 = self.base_qsat["n2"] * (0.94 + 0.03 * self.capacity_factor)

        b_co2 = (
            self.base_b0["co2"]
            * self.affinity_factor_co2
            * np.exp(self.dH[0] / R * (1.0 / T - 1.0 / self.T_ref))
        )
        b_n2 = self.base_b0["n2"] * np.exp(
            self.dH[1] / R * (1.0 / T - 1.0 / self.T_ref)
        )

        denom = 1.0 + b_co2 * Pco2 + b_n2 * Pn2

        q_co2 = qsat_co2 * b_co2 * Pco2 / denom
        q_n2 = qsat_n2 * b_n2 * Pn2 / denom

        return [
            np.clip(q_co2, 0.0, qsat_co2),
            np.clip(q_n2, 0.0, qsat_n2),
        ]

    def make_column(self, P_init, y_init):
        c = column(
            self.L,
            self.A,
            n_component=2,
            N_node=self.N,
        )

        c.adsorbent_info(
            self.isotherm,
            self.eps,
            self.dp,
            self.rho_s,
        )
        c.gas_prop_info(self.M, self.mu)
        c.mass_trans_info(self.k_mtc, self.a_surf, self.D_ax)
        c.thermal_info(self.dH, self.Cp_s, self.Cp_g, self.h_heat)

        P0 = np.ones(self.N) * P_init
        Tg0 = np.ones(self.N) * self.T
        Ts0 = np.ones(self.N) * self.T

        y0 = [
            np.ones(self.N) * y_init[0],
            np.ones(self.N) * y_init[1],
        ]
        q0 = self.isotherm([y0[0] * P0, y0[1] * P0], Tg0)

        c.initialC_info(P0, Tg0, Ts0, y0, q0)
        return c

    def set_boundary(self, c, Pin, Pout, y_in, v_superficial, forward=True):
        Q_in = max(v_superficial, 1e-9) * self.A * self.eps

        c.boundaryC_info(
            Pout,
            Pin,
            self.T,
            list(y_in),
            Cv_in=8e-4,
            Cv_out=8e-4,
            Q_inlet=Q_in,
            assigned_v_option=True,
            foward_flow_direction=forward,
        )

    def unpack_result(self, y_result):
        C_co2 = y_result[:, 0:self.N]
        C_n2 = y_result[:, self.N:2 * self.N]
        q_co2 = y_result[:, 2 * self.N:3 * self.N]
        q_n2 = y_result[:, 3 * self.N:4 * self.N]
        return C_co2, C_n2, q_co2, q_n2

    def update_initial_from_final(self, c, y_result, P_new):
        C_co2, C_n2, q_co2, q_n2 = self.unpack_result(y_result)

        C1 = np.maximum(C_co2[-1, :], 1e-12)
        C2 = np.maximum(C_n2[-1, :], 1e-12)
        Ctot = C1 + C2

        y1 = C1 / Ctot
        y2 = C2 / Ctot

        P0 = np.ones(self.N) * P_new
        Tg0 = np.ones(self.N) * self.T
        Ts0 = np.ones(self.N) * self.T
        q0 = [
            np.maximum(q_co2[-1, :], 0.0),
            np.maximum(q_n2[-1, :], 0.0),
        ]

        c.initialC_info(P0, Tg0, Ts0, [y1, y2], q0)

    def run_step(self, c, step_time, n_sec=10):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_result, z, t = c.run_ma(
                step_time,
                n_sec=n_sec,
                CPUtime_print=False,
            )

        return {
            "y_result": y_result,
            "z": z,
            "t": t,
        }

    def integrate_outlet_moles(self, step, v_superficial, outlet="left"):
        y_result = step["y_result"]
        t = step["t"]
        C_co2, C_n2, _, _ = self.unpack_result(y_result)

        idx = 0 if outlet == "left" else -1
        Q = max(v_superficial, 1e-9) * self.A * self.eps

        F_co2 = np.maximum(C_co2[:, idx], 0.0) * Q
        F_n2 = np.maximum(C_n2[:, idx], 0.0) * Q

        n_co2 = np.trapezoid(F_co2, t)
        n_n2 = np.trapezoid(F_n2, t)
        return n_co2, n_n2

    def inventory_from_step(self, step):
        C_co2, C_n2, q_co2, q_n2 = self.unpack_result(step["y_result"])

        gas_co2 = float(np.mean(np.maximum(C_co2[-1, :], 0.0)) * self.void_volume)
        gas_n2 = float(np.mean(np.maximum(C_n2[-1, :], 0.0)) * self.void_volume)
        ads_co2 = float(np.mean(np.maximum(q_co2[-1, :], 0.0)) * self.solid_mass)
        ads_n2 = float(np.mean(np.maximum(q_n2[-1, :], 0.0)) * self.solid_mass)

        total_gas = gas_co2 + gas_n2
        if total_gas > 0.0:
            y_co2 = gas_co2 / total_gas
        else:
            y_co2 = 0.0

        return {
            "gas_co2": gas_co2,
            "gas_n2": gas_n2,
            "ads_co2": ads_co2,
            "ads_n2": ads_n2,
            "y_co2": y_co2,
            "y_n2": 1.0 - y_co2,
        }

    def evaluate_cycle_metrics(self, history, v0, PL, tADS):
        C_feed = self.PH * 1e5 / (R * self.T)
        Q_feed = v0 * self.A * self.eps

        n_co2_feed = C_feed * Q_feed * self.y_feed[0] * tADS
        n_n2_feed = C_feed * Q_feed * self.y_feed[1] * tADS

        v_vac = max(v0 * 0.12, 1e-4)
        n_co2_dep, n_n2_dep = self.integrate_outlet_moles(
            history["depress"],
            v_superficial=v_vac,
            outlet="left",
        )
        n_co2_des, n_n2_des = self.integrate_outlet_moles(
            history["desorb"],
            v_superficial=v_vac,
            outlet="left",
        )

        state_ads = self.inventory_from_step(history["ads"])
        state_des = self.inventory_from_step(history["desorb"])

        n_co2_dynamic = max(
            (state_ads["ads_co2"] + state_ads["gas_co2"])
            - (state_des["ads_co2"] + state_des["gas_co2"]),
            0.0,
        )
        n_n2_dynamic = max(
            (state_ads["ads_n2"] + state_ads["gas_n2"])
            - (state_des["ads_n2"] + state_des["gas_n2"]),
            0.0,
        )

        tail_fraction = clip(0.32 + 0.20 * (PL / self.PH), 0.34, 0.46)
        n_co2_void_tail = state_ads["gas_co2"] * tail_fraction
        n_n2_void_tail = state_ads["gas_n2"] * tail_fraction

        outlet_co2 = max(n_co2_dep + n_co2_des, 0.0)
        outlet_n2 = max(n_n2_dep + n_n2_des, 0.0)

        co2_release_target = n_co2_dynamic + n_co2_void_tail
        n2_release_target = n_n2_dynamic + n_n2_void_tail
        pressure_swing_factor = clip(1.0 - PL / self.PH, 0.0, 1.0)
        feed_strength = clip((self.y_feed[0] - 0.15) / 0.10, 0.0, 1.0)
        recovery_cap = clip(
            0.48
            + 0.22 * pressure_swing_factor
            + 0.10 * feed_strength
            + 0.05 * (self.capacity_factor - 0.82) / 0.16
            - 0.06 * self.deactivation_index / 0.55,
            0.45,
            0.85,
        )

        n_co2_product = min(
            0.35 * outlet_co2 + 0.65 * co2_release_target,
            recovery_cap * n_co2_feed,
        )
        n_n2_product = max(
            0.25 * outlet_n2 + 0.75 * n2_release_target,
            1e-10,
        )

        n_co2_product = max(n_co2_product, 1e-10)
        total_product = n_co2_product + n_n2_product
        CO2_purity = n_co2_product / total_product
        CO2_recovery = n_co2_product / n_co2_feed if n_co2_feed > 0 else 0.0

        valid_physics, status_detail = self.validate_metrics(
            CO2_purity=CO2_purity,
            CO2_recovery=CO2_recovery,
            n_co2_feed=n_co2_feed,
            n_co2_product=n_co2_product,
            n_n2_product=n_n2_product,
            n_co2_dynamic=n_co2_dynamic,
            y_co2_ads=state_ads["y_co2"],
        )

        return {
            "CO2_purity": CO2_purity,
            "CO2_recovery": CO2_recovery,
            "n_CO2_feed": n_co2_feed,
            "n_N2_feed": n_n2_feed,
            "n_CO2_product": n_co2_product,
            "n_N2_product": n_n2_product,
            "n_CO2_dynamic_product": n_co2_dynamic,
            "n_N2_dynamic_product": n_n2_dynamic,
            "n_CO2_available": co2_release_target,
            "bed_y_CO2_before_blowdown": state_ads["y_co2"],
            "bed_y_N2_before_blowdown": state_ads["y_n2"],
            "n_CO2_void_tail": n_co2_void_tail,
            "n_N2_void_tail": n_n2_void_tail,
            "valid_physics": valid_physics,
            "status_detail": status_detail,
        }

    def validate_metrics(
        self,
        CO2_purity,
        CO2_recovery,
        n_co2_feed,
        n_co2_product,
        n_n2_product,
        n_co2_dynamic,
        y_co2_ads,
    ):
        values = [
            CO2_purity,
            CO2_recovery,
            n_co2_feed,
            n_co2_product,
            n_n2_product,
            n_co2_dynamic,
            y_co2_ads,
        ]
        if not np.all(np.isfinite(values)):
            return False, "non-finite metrics"

        if n_co2_product <= 0.0 or n_n2_product < 0.0:
            return False, "non-positive product flow"
        if n_co2_product > 0.88 * n_co2_feed:
            return False, "co2 product exceeds practical recovery cap"
        if not (0.10 <= CO2_purity <= 0.95):
            return False, "co2 purity outside stable range"
        if not (0.02 <= CO2_recovery <= 0.86):
            return False, "co2 recovery outside stable range"
        if CO2_purity * CO2_recovery < 0.05:
            return False, "poor/failed separation under sampled condition"
        if not (0.01 <= y_co2_ads <= 0.80):
            return False, "bed composition before blowdown is unrealistic"
        if n_co2_dynamic <= 1e-6:
            return False, "near-zero dynamic working capacity"

        return True, "ok"

    def simulate_one_cycle_with_column(
        self,
        c,
        tADS,
        PL,
        v0,
        t_press,
        t_depress,
        t_desorb=None,
        n_sec=10,
        calculate_metrics=True,
    ):
        if t_desorb is None:
            t_desorb = tADS

        history = {}

        self.set_boundary(
            c,
            Pin=self.PH,
            Pout=self.PH,
            y_in=self.y_feed,
            v_superficial=max(v0 * 0.25, 1e-4),
            forward=True,
        )
        history["press"] = self.run_step(c, t_press, n_sec)
        self.update_initial_from_final(c, history["press"]["y_result"], self.PH)

        self.set_boundary(
            c,
            Pin=self.PH,
            Pout=self.PH,
            y_in=self.y_feed,
            v_superficial=v0,
            forward=True,
        )
        history["ads"] = self.run_step(c, tADS, n_sec)
        self.update_initial_from_final(c, history["ads"]["y_result"], self.PH)

        self.set_boundary(
            c,
            Pin=PL,
            Pout=PL,
            y_in=[0.01, 0.99],
            v_superficial=max(v0 * 0.12, 1e-4),
            forward=False,
        )
        history["depress"] = self.run_step(c, t_depress, n_sec)
        self.update_initial_from_final(c, history["depress"]["y_result"], PL)

        self.set_boundary(
            c,
            Pin=PL,
            Pout=PL,
            y_in=[0.01, 0.99],
            v_superficial=max(v0 * 0.12, 1e-4),
            forward=False,
        )
        history["desorb"] = self.run_step(c, t_desorb, n_sec)
        self.update_initial_from_final(c, history["desorb"]["y_result"], PL)

        if not calculate_metrics:
            return None, history

        metrics = self.evaluate_cycle_metrics(history, v0=v0, PL=PL, tADS=tADS)
        return metrics, history

    def simulate_cycles(
        self,
        tADS,
        PL,
        v0,
        t_press,
        t_depress,
        n_cycles=10,
        n_sec=10,
    ):
        c = self.make_column(
            P_init=PL,
            y_init=[0.01, 0.99],
        )
        final_metrics = None

        for cycle_idx in range(1, n_cycles + 1):
            calculate_metrics = cycle_idx == n_cycles
            final_metrics, _ = self.simulate_one_cycle_with_column(
                c=c,
                tADS=tADS,
                PL=PL,
                v0=v0,
                t_press=t_press,
                t_depress=t_depress,
                t_desorb=max(tADS * 0.90, 45.0),
                n_sec=n_sec,
                calculate_metrics=calculate_metrics,
            )

        return final_metrics


def generate_synthetic_dataset(
    n_samples=30,
    n_cycles=10,
    output_csv="mg_mof74_pretreated_psa_lhs.csv",
    seed=100,
    include_invalid=False,
):
    bounds = {
        "tADS": (70.0, 180.0),
        "PL": (0.12, 0.35),
        "v0": (0.010, 0.032),
        "t_press": (20.0, 70.0),
        "t_depress": (25.0, 75.0),
        "h2o_ppmv": (1.0, 100.0),
        "sox_ppmv": (0.0, 1.0),
        "nox_ppmv": (0.0, 10.0),
        "dust_mg_Nm3": (0.0, 0.1),
        "co2_mol_frac": (0.15, 0.25),
    }

    model = MgMOF74PSA(
        L=1.0,
        A_cros=0.01,
        N_node=31,
        T=313.15,
        PH=1.2,
    )

    rows = []
    accepted = 0
    batch_id = 0
    max_batches = max(4, n_samples * 3)

    while accepted < n_samples and batch_id < max_batches:
        lhs_df = lhs_sampling(bounds, n_samples=n_samples, seed=seed + 97 * batch_id)

        for _, row in lhs_df.iterrows():
            params = row.to_dict()

            model.set_pretreated_feed(
                co2_mol_frac=float(params["co2_mol_frac"]),
                h2o_ppmv=float(params["h2o_ppmv"]),
                sox_ppmv=float(params["sox_ppmv"]),
                nox_ppmv=float(params["nox_ppmv"]),
                dust_mg_Nm3=float(params["dust_mg_Nm3"]),
            )

            try:
                metrics = model.simulate_cycles(
                    tADS=float(params["tADS"]),
                    PL=float(params["PL"]),
                    v0=float(params["v0"]),
                    t_press=float(params["t_press"]),
                    t_depress=float(params["t_depress"]),
                    n_cycles=n_cycles,
                    n_sec=10,
                )

                result = {
                    "sample_id": accepted + 1 if metrics["valid_physics"] else np.nan,
                    **params,
                    "n2_mol_frac_effective": model.n2_mol_frac_effective,
                    "capacity_factor": model.capacity_factor,
                    "affinity_factor_co2": model.affinity_factor_co2,
                    "mtc_factor": model.mtc_factor,
                    "dax_factor": model.dax_factor,
                    "deactivation_index": model.deactivation_index,
                    **metrics,
                    "n_cycles": n_cycles,
                    "status": "ok" if metrics["valid_physics"] else f"rejected: {metrics['status_detail']}",
                }

            except Exception as exc:
                result = {
                    "sample_id": np.nan,
                    **params,
                    "n2_mol_frac_effective": model.n2_mol_frac_effective,
                    "capacity_factor": model.capacity_factor,
                    "affinity_factor_co2": model.affinity_factor_co2,
                    "mtc_factor": model.mtc_factor,
                    "dax_factor": model.dax_factor,
                    "deactivation_index": model.deactivation_index,
                    "CO2_purity": np.nan,
                    "CO2_recovery": np.nan,
                    "n_CO2_feed": np.nan,
                    "n_N2_feed": np.nan,
                    "n_CO2_product": np.nan,
                    "n_N2_product": np.nan,
                    "n_CO2_dynamic_product": np.nan,
                    "n_N2_dynamic_product": np.nan,
                    "n_CO2_available": np.nan,
                    "bed_y_CO2_before_blowdown": np.nan,
                    "bed_y_N2_before_blowdown": np.nan,
                    "n_CO2_void_tail": np.nan,
                    "n_N2_void_tail": np.nan,
                    "valid_physics": False,
                    "status_detail": str(exc),
                    "n_cycles": n_cycles,
                    "status": f"error: {exc}",
                }

            if result["valid_physics"]:
                result["sample_id"] = accepted + 1
                rows.append(result)
                accepted += 1
                print(f"accepted {accepted}/{n_samples}")
            elif include_invalid:
                rows.append(result)

            if accepted >= n_samples:
                break

        batch_id += 1

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"Saved: {output_csv}")
    if accepted < n_samples:
        print(f"Accepted {accepted} valid samples out of requested {n_samples}.")
    return result_df


if __name__ == "__main__":
    df = generate_synthetic_dataset(
        n_samples=30,
        n_cycles=10,
        output_csv="test.csv",
        seed=100,
        include_invalid=False,
    )
    print(df.head())
