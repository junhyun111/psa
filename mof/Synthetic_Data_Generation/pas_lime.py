import warnings

import numpy as np
import pandas as pd

from pyapep.simsep import column


R = 8.3145


def lhs_sampling(bounds, n_samples=10, seed=20):
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


class LimeKilnPSA:
    """
    PSA/VSA-style surrogate for lime-kiln flue gas CO2 capture with Mg-MOF-74.

    Basis for the lime-kiln feed envelope:
    - Typical wet kiln exhaust contains about 24.3% CO2, 15.3% H2O, 0.7% O2,
      and the balance mostly N2 in EPA lime-industry references.
    - CO2 capture studies for lime kilns commonly report exhaust CO2 around
      22-29 vol%, so the sampling bounds are widened slightly around that range.
    - Raw kiln exhaust is hot, but PSA feed must be cooled and pretreated.

    Modeling assumptions:
    - The PSA model remains binary (CO2 vs. lumped inert) for compatibility
      with the original simulator.
    - O2, residual H2O, SOx, NOx, and dust are treated as deactivation or
      mass-transfer penalties rather than explicit adsorbing species.
    - Both raw flue-gas state and conditioned PSA-feed state are emitted as
      features so that transfer-learning experiments can absorb feature shifts.
    """

    def __init__(
        self,
        L=1.0,
        A_cros=0.01,
        N_node=31,
        T=323.15,
        PH=1.20,
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
        self.mu = [1.52e-5, 1.80e-5]
        self.base_D_ax = np.array([8.8e-6, 9.0e-6], dtype=float)
        self.base_k_mtc = np.array([1.30e-3, 7.20e-4], dtype=float)
        self.a_surf = 6.0 / self.dp

        self.dH = [47e3, 21e3]
        self.Cp_s = 920.0
        self.Cp_g = [37.1, 29.1]
        self.h_heat = 78.0

        self.base_qsat = {
            "co2": 8.80,
            "n2": 0.55,
        }
        self.base_b0 = {
            "co2": 5.8,
            "n2": 0.075,
        }

        self.set_lime_kiln_feed(
            co2_mol_frac_wet=0.255,
            o2_mol_frac_wet=0.015,
            h2o_mol_frac_wet=0.135,
            sox_ppmv_raw=45.0,
            nox_ppmv_raw=180.0,
            dust_mg_Nm3_raw=22.0,
            kiln_exit_temp_c=620.0,
            feed_temp_c=45.0,
            dehumidification_efficiency=0.995,
            desulfurization_efficiency=0.985,
            denox_efficiency=0.70,
            dedusting_efficiency=0.995,
        )

    def set_lime_kiln_feed(
        self,
        co2_mol_frac_wet,
        o2_mol_frac_wet,
        h2o_mol_frac_wet,
        sox_ppmv_raw,
        nox_ppmv_raw,
        dust_mg_Nm3_raw,
        kiln_exit_temp_c,
        feed_temp_c,
        dehumidification_efficiency,
        desulfurization_efficiency,
        denox_efficiency,
        dedusting_efficiency,
    ):
        self.co2_mol_frac_wet = clip(co2_mol_frac_wet, 0.22, 0.30)
        self.o2_mol_frac_wet = clip(o2_mol_frac_wet, 0.003, 0.050)
        self.h2o_mol_frac_wet = clip(h2o_mol_frac_wet, 0.08, 0.18)

        inert_balance = 1.0 - (
            self.co2_mol_frac_wet + self.o2_mol_frac_wet + self.h2o_mol_frac_wet
        )
        self.n2_mol_frac_wet = max(inert_balance, 0.50)

        self.sox_ppmv_raw = clip(sox_ppmv_raw, 0.0, 250.0)
        self.nox_ppmv_raw = clip(nox_ppmv_raw, 20.0, 500.0)
        self.dust_mg_Nm3_raw = clip(dust_mg_Nm3_raw, 2.0, 120.0)
        self.kiln_exit_temp_c = clip(kiln_exit_temp_c, 420.0, 980.0)
        self.feed_temp_c = clip(feed_temp_c, 30.0, 80.0)
        self.feed_temp_k = self.feed_temp_c + 273.15
        self.T = self.feed_temp_k

        self.dehumidification_efficiency = clip(dehumidification_efficiency, 0.95, 0.9995)
        self.desulfurization_efficiency = clip(desulfurization_efficiency, 0.85, 0.999)
        self.denox_efficiency = clip(denox_efficiency, 0.20, 0.95)
        self.dedusting_efficiency = clip(dedusting_efficiency, 0.90, 0.9995)

        self.h2o_ppmv_residual = clip(
            self.h2o_mol_frac_wet * 1e6 * (1.0 - self.dehumidification_efficiency),
            10.0,
            4000.0,
        )
        self.sox_ppmv_residual = clip(
            self.sox_ppmv_raw * (1.0 - self.desulfurization_efficiency),
            0.0,
            40.0,
        )
        self.nox_ppmv_residual = clip(
            self.nox_ppmv_raw * (1.0 - self.denox_efficiency),
            0.0,
            200.0,
        )
        self.dust_mg_Nm3_residual = clip(
            self.dust_mg_Nm3_raw * (1.0 - self.dedusting_efficiency),
            0.0,
            10.0,
        )

        dry_major_total = max(
            self.co2_mol_frac_wet + self.o2_mol_frac_wet + self.n2_mol_frac_wet,
            1e-8,
        )
        self.co2_mol_frac_dry = self.co2_mol_frac_wet / dry_major_total
        self.o2_mol_frac_dry = self.o2_mol_frac_wet / dry_major_total
        self.inert_mol_frac_effective = 1.0 - self.co2_mol_frac_dry

        self.y_feed = np.array(
            [self.co2_mol_frac_dry, self.inert_mol_frac_effective],
            dtype=float,
        )

        factors = self.compute_pretreatment_factors()
        self.capacity_factor = factors["capacity_factor"]
        self.affinity_factor_co2 = factors["affinity_factor_co2"]
        self.mtc_factor = factors["mtc_factor"]
        self.dax_factor = factors["dax_factor"]
        self.deactivation_index = factors["deactivation_index"]
        self.oxidative_stress_index = factors["oxidative_stress_index"]
        self.cooling_load_index = factors["cooling_load_index"]

        self.D_ax = list(self.base_D_ax * self.dax_factor)
        self.k_mtc = list(
            self.base_k_mtc * np.array([self.mtc_factor, self.mtc_factor * 0.97])
        )

    def compute_pretreatment_factors(self):
        h2o_load = clip(self.h2o_ppmv_residual / 4000.0, 0.0, 1.0)
        sox_load = clip(self.sox_ppmv_residual / 40.0, 0.0, 1.0)
        nox_load = clip(self.nox_ppmv_residual / 200.0, 0.0, 1.0)
        dust_load = clip(self.dust_mg_Nm3_residual / 10.0, 0.0, 1.0)
        o2_load = clip((self.o2_mol_frac_dry - 0.003) / 0.047, 0.0, 1.0)
        thermal_load = clip((self.kiln_exit_temp_c - 420.0) / 560.0, 0.0, 1.0)
        cooling_load = clip((self.feed_temp_c - 30.0) / 50.0, 0.0, 1.0)

        oxidative_stress_index = clip(0.65 * o2_load + 0.35 * nox_load, 0.0, 1.0)
        deactivation_index = clip(
            0.28 * h2o_load
            + 0.18 * sox_load
            + 0.11 * nox_load
            + 0.05 * dust_load
            + 0.05 * o2_load
            + 0.03 * thermal_load,
            0.0,
            0.72,
        )

        capacity_factor = clip(
            1.0
            - (
                0.03
                + 0.10 * h2o_load
                + 0.07 * sox_load
                + 0.03 * nox_load
                + 0.02 * dust_load
                + 0.01 * thermal_load
            ),
            0.76,
            0.98,
        )
        affinity_factor_co2 = clip(
            1.0
            - (
                0.02
                + 0.06 * h2o_load
                + 0.05 * sox_load
                + 0.02 * nox_load
                + 0.015 * o2_load
            ),
            0.80,
            0.99,
        )
        mtc_factor = clip(
            1.0
            - (
                0.05 * h2o_load
                + 0.03 * dust_load
                + 0.025 * cooling_load
                + 0.02 * sox_load
            ),
            0.82,
            0.99,
        )
        dax_factor = clip(
            1.0 + 0.05 * dust_load + 0.03 * nox_load + 0.02 * cooling_load,
            1.0,
            1.10,
        )

        return {
            "capacity_factor": capacity_factor,
            "affinity_factor_co2": affinity_factor_co2,
            "mtc_factor": mtc_factor,
            "dax_factor": dax_factor,
            "deactivation_index": deactivation_index,
            "oxidative_stress_index": oxidative_stress_index,
            "cooling_load_index": cooling_load,
        }

    def isotherm(self, P, T):
        Pco2 = np.asarray(P[0], dtype=float)
        Pinert = np.asarray(P[1], dtype=float)
        T = np.asarray(T, dtype=float)

        qsat_co2 = self.base_qsat["co2"] * self.capacity_factor
        qsat_n2 = self.base_qsat["n2"] * (0.93 + 0.04 * self.capacity_factor)

        b_co2 = (
            self.base_b0["co2"]
            * self.affinity_factor_co2
            * np.exp(self.dH[0] / R * (1.0 / T - 1.0 / self.T_ref))
        )
        b_n2 = self.base_b0["n2"] * np.exp(
            self.dH[1] / R * (1.0 / T - 1.0 / self.T_ref)
        )

        denom = 1.0 + b_co2 * Pco2 + b_n2 * Pinert

        q_co2 = qsat_co2 * b_co2 * Pco2 / denom
        q_n2 = qsat_n2 * b_n2 * Pinert / denom

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

        c.adsorbent_info(self.isotherm, self.eps, self.dp, self.rho_s)
        c.gas_prop_info(self.M, self.mu)
        c.mass_trans_info(self.k_mtc, self.a_surf, self.D_ax)
        c.thermal_info(self.dH, self.Cp_s, self.Cp_g, self.h_heat)

        P0 = np.ones(self.N) * P_init
        Tg0 = np.ones(self.N) * self.T
        Ts0 = np.ones(self.N) * self.T

        y0 = [np.ones(self.N) * y_init[0], np.ones(self.N) * y_init[1]]
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
        c_co2 = y_result[:, 0:self.N]
        c_n2 = y_result[:, self.N:2 * self.N]
        q_co2 = y_result[:, 2 * self.N:3 * self.N]
        q_n2 = y_result[:, 3 * self.N:4 * self.N]
        return c_co2, c_n2, q_co2, q_n2

    def update_initial_from_final(self, c, y_result, P_new):
        c_co2, c_n2, q_co2, q_n2 = self.unpack_result(y_result)

        c1 = np.maximum(c_co2[-1, :], 1e-12)
        c2 = np.maximum(c_n2[-1, :], 1e-12)
        ctot = c1 + c2

        y1 = c1 / ctot
        y2 = c2 / ctot

        P0 = np.ones(self.N) * P_new
        Tg0 = np.ones(self.N) * self.T
        Ts0 = np.ones(self.N) * self.T
        q0 = [np.maximum(q_co2[-1, :], 0.0), np.maximum(q_n2[-1, :], 0.0)]

        c.initialC_info(P0, Tg0, Ts0, [y1, y2], q0)

    def run_step(self, c, step_time, n_sec=10):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_result, z, t = c.run_ma(step_time, n_sec=n_sec, CPUtime_print=False)

        return {
            "y_result": y_result,
            "z": z,
            "t": t,
        }

    def integrate_outlet_moles(self, step, v_superficial, outlet="left"):
        y_result = step["y_result"]
        t = step["t"]
        c_co2, c_n2, _, _ = self.unpack_result(y_result)

        idx = 0 if outlet == "left" else -1
        q_vol = max(v_superficial, 1e-9) * self.A * self.eps

        f_co2 = np.maximum(c_co2[:, idx], 0.0) * q_vol
        f_n2 = np.maximum(c_n2[:, idx], 0.0) * q_vol

        n_co2 = np.trapezoid(f_co2, t)
        n_n2 = np.trapezoid(f_n2, t)
        return n_co2, n_n2

    def inventory_from_step(self, step):
        c_co2, c_n2, q_co2, q_n2 = self.unpack_result(step["y_result"])

        gas_co2 = float(np.mean(np.maximum(c_co2[-1, :], 0.0)) * self.void_volume)
        gas_n2 = float(np.mean(np.maximum(c_n2[-1, :], 0.0)) * self.void_volume)
        ads_co2 = float(np.mean(np.maximum(q_co2[-1, :], 0.0)) * self.solid_mass)
        ads_n2 = float(np.mean(np.maximum(q_n2[-1, :], 0.0)) * self.solid_mass)

        total_gas = gas_co2 + gas_n2
        y_co2 = gas_co2 / total_gas if total_gas > 0.0 else 0.0

        return {
            "gas_co2": gas_co2,
            "gas_n2": gas_n2,
            "ads_co2": ads_co2,
            "ads_n2": ads_n2,
            "y_co2": y_co2,
            "y_n2": 1.0 - y_co2,
        }

    def evaluate_cycle_metrics(self, history, v0, PL, tADS):
        c_feed = self.PH * 1e5 / (R * self.T)
        q_feed = v0 * self.A * self.eps

        n_co2_feed = c_feed * q_feed * self.y_feed[0] * tADS
        n_n2_feed = c_feed * q_feed * self.y_feed[1] * tADS

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

        tail_fraction = clip(0.30 + 0.16 * (PL / self.PH), 0.30, 0.44)
        n_co2_void_tail = state_ads["gas_co2"] * tail_fraction
        n_n2_void_tail = state_ads["gas_n2"] * tail_fraction

        outlet_co2 = max(n_co2_dep + n_co2_des, 0.0)
        outlet_n2 = max(n_n2_dep + n_n2_des, 0.0)

        co2_release_target = n_co2_dynamic + n_co2_void_tail
        n2_release_target = n_n2_dynamic + n_n2_void_tail
        pressure_swing_factor = clip(1.0 - PL / self.PH, 0.0, 1.0)
        feed_strength = clip((self.y_feed[0] - 0.22) / 0.12, 0.0, 1.0)

        recovery_cap = clip(
            0.56
            + 0.18 * pressure_swing_factor
            + 0.12 * feed_strength
            + 0.05 * (self.capacity_factor - 0.76) / 0.22
            - 0.07 * self.deactivation_index / 0.72
            - 0.03 * self.cooling_load_index,
            0.50,
            0.90,
        )

        n_co2_product = min(
            0.35 * outlet_co2 + 0.65 * co2_release_target,
            recovery_cap * n_co2_feed,
        )
        n_n2_product = max(0.22 * outlet_n2 + 0.78 * n2_release_target, 1e-10)

        n_co2_product = max(n_co2_product, 1e-10)
        total_product = n_co2_product + n_n2_product
        co2_purity = n_co2_product / total_product
        co2_recovery = n_co2_product / n_co2_feed if n_co2_feed > 0 else 0.0

        valid_physics, status_detail = self.validate_metrics(
            co2_purity=co2_purity,
            co2_recovery=co2_recovery,
            n_co2_feed=n_co2_feed,
            n_co2_product=n_co2_product,
            n_n2_product=n_n2_product,
            n_co2_dynamic=n_co2_dynamic,
            y_co2_ads=state_ads["y_co2"],
        )

        return {
            "CO2_purity": co2_purity,
            "CO2_recovery": co2_recovery,
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
        co2_purity,
        co2_recovery,
        n_co2_feed,
        n_co2_product,
        n_n2_product,
        n_co2_dynamic,
        y_co2_ads,
    ):
        values = [
            co2_purity,
            co2_recovery,
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
        if n_co2_product > 0.92 * n_co2_feed:
            return False, "co2 product exceeds practical recovery cap"
        if not (0.20 <= co2_purity <= 0.98):
            return False, "co2 purity outside stable range"
        if not (0.05 <= co2_recovery <= 0.91):
            return False, "co2 recovery outside stable range"
        if co2_purity * co2_recovery < 0.12:
            return False, "poor/failed separation under sampled condition"
        if not (0.05 <= y_co2_ads <= 0.92):
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
            y_in=[0.02, 0.98],
            v_superficial=max(v0 * 0.12, 1e-4),
            forward=False,
        )
        history["depress"] = self.run_step(c, t_depress, n_sec)
        self.update_initial_from_final(c, history["depress"]["y_result"], PL)

        self.set_boundary(
            c,
            Pin=PL,
            Pout=PL,
            y_in=[0.02, 0.98],
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
        c = self.make_column(P_init=PL, y_init=[0.02, 0.98])
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
                t_desorb=max(tADS * 0.90, 50.0),
                n_sec=n_sec,
                calculate_metrics=calculate_metrics,
            )

        return final_metrics


def generate_synthetic_dataset(
    n_samples=10,
    n_cycles=10,
    output_csv="lime_kiln_psa_lhs.csv",
    seed=20,
    include_invalid=False,
):
    bounds = {
        "tADS": (80.0, 220.0),
        "PL": (0.08, 0.28),
        "v0": (0.010, 0.036),
        "t_press": (20.0, 75.0),
        "t_depress": (30.0, 90.0),
        "co2_mol_frac_wet": (0.22, 0.30),
        "o2_mol_frac_wet": (0.003, 0.050),
        "h2o_mol_frac_wet": (0.08, 0.18),
        "sox_ppmv_raw": (0.0, 250.0),
        "nox_ppmv_raw": (20.0, 500.0),
        "dust_mg_Nm3_raw": (2.0, 120.0),
        "kiln_exit_temp_c": (420.0, 980.0),
        "feed_temp_c": (30.0, 80.0),
        "dehumidification_efficiency": (0.95, 0.9995),
        "desulfurization_efficiency": (0.85, 0.999),
        "denox_efficiency": (0.20, 0.95),
        "dedusting_efficiency": (0.90, 0.9995),
    }

    model = LimeKilnPSA(
        L=1.0,
        A_cros=0.01,
        N_node=31,
        T=323.15,
        PH=1.20,
    )

    rows = []
    accepted = 0
    batch_id = 0
    max_batches = max(4, n_samples * 3)

    while accepted < n_samples and batch_id < max_batches:
        lhs_df = lhs_sampling(bounds, n_samples=n_samples, seed=seed + 97 * batch_id)

        for _, row in lhs_df.iterrows():
            params = row.to_dict()

            model.set_lime_kiln_feed(
                co2_mol_frac_wet=float(params["co2_mol_frac_wet"]),
                o2_mol_frac_wet=float(params["o2_mol_frac_wet"]),
                h2o_mol_frac_wet=float(params["h2o_mol_frac_wet"]),
                sox_ppmv_raw=float(params["sox_ppmv_raw"]),
                nox_ppmv_raw=float(params["nox_ppmv_raw"]),
                dust_mg_Nm3_raw=float(params["dust_mg_Nm3_raw"]),
                kiln_exit_temp_c=float(params["kiln_exit_temp_c"]),
                feed_temp_c=float(params["feed_temp_c"]),
                dehumidification_efficiency=float(params["dehumidification_efficiency"]),
                desulfurization_efficiency=float(params["desulfurization_efficiency"]),
                denox_efficiency=float(params["denox_efficiency"]),
                dedusting_efficiency=float(params["dedusting_efficiency"]),
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
                    "co2_mol_frac_dry": model.co2_mol_frac_dry,
                    "o2_mol_frac_dry": model.o2_mol_frac_dry,
                    "n2_mol_frac_wet": model.n2_mol_frac_wet,
                    "inert_mol_frac_effective": model.inert_mol_frac_effective,
                    "h2o_ppmv_residual": model.h2o_ppmv_residual,
                    "sox_ppmv_residual": model.sox_ppmv_residual,
                    "nox_ppmv_residual": model.nox_ppmv_residual,
                    "dust_mg_Nm3_residual": model.dust_mg_Nm3_residual,
                    "capacity_factor": model.capacity_factor,
                    "affinity_factor_co2": model.affinity_factor_co2,
                    "mtc_factor": model.mtc_factor,
                    "dax_factor": model.dax_factor,
                    "deactivation_index": model.deactivation_index,
                    "oxidative_stress_index": model.oxidative_stress_index,
                    "cooling_load_index": model.cooling_load_index,
                    **metrics,
                    "n_cycles": n_cycles,
                    "status": "ok" if metrics["valid_physics"] else f"rejected: {metrics['status_detail']}",
                }
            except Exception as exc:
                result = {
                    "sample_id": np.nan,
                    **params,
                    "co2_mol_frac_dry": model.co2_mol_frac_dry,
                    "o2_mol_frac_dry": model.o2_mol_frac_dry,
                    "n2_mol_frac_wet": model.n2_mol_frac_wet,
                    "inert_mol_frac_effective": model.inert_mol_frac_effective,
                    "h2o_ppmv_residual": model.h2o_ppmv_residual,
                    "sox_ppmv_residual": model.sox_ppmv_residual,
                    "nox_ppmv_residual": model.nox_ppmv_residual,
                    "dust_mg_Nm3_residual": model.dust_mg_Nm3_residual,
                    "capacity_factor": model.capacity_factor,
                    "affinity_factor_co2": model.affinity_factor_co2,
                    "mtc_factor": model.mtc_factor,
                    "dax_factor": model.dax_factor,
                    "deactivation_index": model.deactivation_index,
                    "oxidative_stress_index": model.oxidative_stress_index,
                    "cooling_load_index": model.cooling_load_index,
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
        n_samples=10,
        n_cycles=10,
        output_csv="lime_seed42.csv",
        seed=20,
        include_invalid=False,
    )
    print(df.head())
