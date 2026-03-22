"""The base class for a 6-bus and 7-device :code:`gym-anm` environment."""

import datetime as dt
import numpy as np
from typing import Optional

from gym_anm import ANMEnv
from .rendering.py import rendering
from .network import network as network_default
from .utils import random_date
from .fix_reward import build_constraint_costs


class BCP_base(ANMEnv):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        observation,
        K,
        delta_t,
        gamma,
        lamb,
        aux_bounds=None,
        costs_clipping=(None, None),
        seed=None,
        network_spec=network_default,
    ):
        super().__init__(network_spec, observation, K, delta_t, gamma, lamb, aux_bounds, costs_clipping, seed)
        # If False, branch overloads are not included in the Lagrangian constraint channel.
        # They are still logged via constraint/branch_overload_pu_max and can still be
        # penalized in the objective (see W_BRANCH_OVERLOAD below).
        self.include_branch_overload_in_constraints = True
        self.network_specs = self.simulator.get_rendering_specs()
        self.timestep_length = dt.timedelta(minutes=int(60 * delta_t))
        self.date = None
        self.date_init = None
        self.year_count = 0
        self.skipped_frames = None
        self.render_mode = None
        self.is_rendering = False

    def render(self, mode="human", skip_frames=0):
        if self.render_mode is None:
            if mode not in ["human"]:
                raise NotImplementedError()
            self.render_mode = mode
            self.skipped_frames = 0
            rendered_network_specs = ["dev_type", "dev_p", "dev_q", "branch_s", "bus_v", "des_soc"]
            specs = {s: self.network_specs[s] for s in rendered_network_specs}
            self._init_render(specs)
            self.render(mode=mode, skip_frames=skip_frames)
            self.is_rendering = True
        else:
            self.skipped_frames = (self.skipped_frames + 1) % (skip_frames + 1)
            if self.skipped_frames:
                return
            full_state = self.simulator.state
            dev_p = [full_state["dev_p"]["MW"].get(i, 0) for i in sorted(full_state["dev_p"]["MW"].keys())]
            dev_q = [full_state["dev_q"]["MVAr"].get(i, 0) for i in sorted(full_state["dev_q"]["MVAr"].keys())]
            branch_s = [full_state["branch_s"]["MVA"].get(i, 0) for i in sorted(full_state["branch_s"]["MVA"].keys())]
            des_soc = [full_state["des_soc"]["MWh"].get(i, 0) for i in sorted(full_state["des_soc"]["MWh"].keys())]
            gen_p_max = [full_state["gen_p_max"]["MW"].get(i, 0) for i in sorted(full_state["gen_p_max"]["MW"].keys())]
            bus_v_magn = [full_state["bus_v_magn"]["pu"].get(i, 0) for i in sorted(full_state["bus_v_magn"]["pu"].keys())]
            costs = [self.e_loss, self.penalty]
            network_collapsed = not self.simulator.pfe_converged
            self._update_render(dev_p, dev_q, branch_s, des_soc, gen_p_max, bus_v_magn, costs, network_collapsed)

    def safe_array(self, x):
        return np.array([np.nan if v is None else float(v) for v in x])

    def step(self, action):
        obs, r, terminated, truncated, info = super().step(action)

        r = 0.0  # Completely replaces ANMEnv reward with custom economic objective.
                 # Constraint penalties are handled externally by LagrangianWrapper.
                 # lamb=0 in BCP disables ANMEnv's own penalty term.

        self.date += self.timestep_length

        full_state = self.simulator.state or {}

        device_data = self.simulator.devices
        base_mva = self.simulator.baseMVA

        dev_p_dict = full_state.get("dev_p", {}).get("MW", {})
        dev_q_dict = full_state.get("dev_q", {}).get("MVAr", {})
        gen_p_max_dict = full_state.get("gen_p_max", {}).get("MW", {})
        bus_v_dict = full_state.get("bus_v_magn", {}).get("pu", {})

        device_ids = sorted(device_data.keys())
        dev_p = np.array([dev_p_dict.get(i, 0.0) for i in device_ids], dtype=np.float32)
        dev_q = np.array([dev_q_dict.get(i, 0.0) for i in device_ids], dtype=np.float32)

        device_type_names = [type(device_data[i]).__name__ for i in device_ids]
        # Map back to integer for violation loops: 0=Load, 1=Classical, 2=Renewable, 3=Storage
        type_mapping = {"Load": 0, "ClassicalGen": 1, "RenewableGen": 2, "StorageUnit": 3}
        device_type = np.array([type_mapping.get(t, 1) for t in device_type_names], dtype=np.int32)

        p_max = np.array([device_data[i].p_max for i in device_ids], dtype=np.float32) * base_mva
        p_min = np.array([device_data[i].p_min for i in device_ids], dtype=np.float32) * base_mva
        q_max = np.array([device_data[i].q_max for i in device_ids], dtype=np.float32) * base_mva
        q_min = np.array([device_data[i].q_min for i in device_ids], dtype=np.float32) * base_mva

        mask = (device_type != 0)
        p_violation = mask * (
            np.maximum(0, dev_p - np.nan_to_num(p_max, nan=np.inf))
            + np.maximum(0, np.nan_to_num(p_min, nan=-np.inf) - dev_p)
        )
        q_violation = mask * (
            np.maximum(0, dev_q - np.nan_to_num(q_max, nan=np.inf))
            + np.maximum(0, np.nan_to_num(q_min, nan=-np.inf) - dev_q)
        )

        bus_ids = sorted(self.simulator.buses.keys())
        bus_v = np.array([bus_v_dict.get(i, 1.0) for i in bus_ids], dtype=float)
        bus_objects = [self.simulator.buses[i] for i in bus_ids]
        v_max = np.array([getattr(b, 'v_max', 1.1) for b in bus_objects], dtype=float)
        v_min = np.array([getattr(b, 'v_min', 0.9) for b in bus_objects], dtype=float)
        v_violation = np.maximum(0, bus_v - v_max) + np.maximum(0, v_min - bus_v)

        delta_t_h = float(getattr(self, "delta_t", 0.25))

        branch_s_dict = full_state.get("branch_s", {}).get("MVA", {})
        branch_ids = sorted(self.simulator.branches.keys())

        branch_overload_mva_max = 0.0
        branch_overload_mwh = 0.0
        branch_soft_penalty_mwh = 0.0
        branch_loading = {}

        for br_id in branch_ids:
            s_mva = abs(float(branch_s_dict.get(br_id, 0.0)))
            s_max = self.simulator.branches[br_id].rate * base_mva
            overload_mva = max(0.0, s_mva - s_max)

            branch_overload_mva_max = max(branch_overload_mva_max, overload_mva)
            branch_overload_mwh += overload_mva * delta_t_h
            branch_soft_penalty_mwh += max(0.0, s_mva - 0.8 * s_max) * delta_t_h
            branch_loading[br_id] = s_mva / max(s_max, 1e-6)

        # ── Constraint costs (used by LagrangianWrapper) ──────────────────────────
        # P-channel combines device active power violations AND branch thermal overloads
        # so that existing (3,) lambda checkpoints remain compatible.
        p_violation_pu_max = float(np.max(p_violation) / base_mva) if len(p_violation) > 0 else 0.0
        branch_overload_pu_max = float(branch_overload_mva_max / base_mva) if base_mva else 0.0
        if getattr(self, "include_branch_overload_in_constraints", True):
            combined_p_constraint = max(p_violation_pu_max, branch_overload_pu_max)
        else:
            combined_p_constraint = p_violation_pu_max

        q_violation_pu_max = float(np.max(q_violation) / base_mva) if len(q_violation) > 0 else 0.0

        info["constraint_costs"] = build_constraint_costs(
            v_violation_pu_max=float(np.max(v_violation)) if len(v_violation) > 0 else 0.0,
            combined_p_constraint_pu=combined_p_constraint,
            q_violation_pu_max=q_violation_pu_max,
        )
        info["constraint/p_violation_pu_max"] = p_violation_pu_max
        info["constraint/branch_overload_pu_max"] = branch_overload_pu_max
        info["pfe_converged"] = bool(getattr(self.simulator, "pfe_converged", True))

        # ── Economic objective ────────────────────────────────────────────────────
        #
        # Goals:
        #   1. Minimize renewable curtailment (use free local generation)
        #   2. Maximize self-sufficiency (reduce slack/grid interaction)
        #   3. Encourage battery cycling (charge when surplus, discharge when needed)
        #   4. Discourage unnecessary reactive power (wastes apparent power capacity)
        #   5. Soft branch loading penalty (pro-active congestion management)
        #
        # NOTE: Hard branch overload is intentionally NOT included here.
        # It is fully handled by the Lagrangian multiplier (lambda_p) via
        # info["constraint_costs"]. Double-penalizing it would prevent lambda
        # from converging to a meaningful value.

        slack_p = float(dev_p_dict.get(0, 0.0))

        # Slack interaction (asymmetric):
        #   import (slack_p > 0): drawing from grid  → penalized at full weight
        #   export (slack_p < 0): sending surplus out → penalized at reduced weight
        slack_import_mw = max(0.0, slack_p)
        slack_export_mw = max(0.0, -slack_p)

        # Curtailment: potential renewable generation vs actual injection
        renewable_ids = [i for i in device_ids if type(device_data[i]).__name__ == "RenewableGen"]
        curtailment_mw = 0.0
        for dev_id in renewable_ids:
            if dev_id not in gen_p_max_dict:
                continue
            p_potential = float(gen_p_max_dict.get(dev_id, 0.0))
            p_actual = float(dev_p_dict.get(dev_id, 0.0))
            curtailment_mw += max(0.0, p_potential - p_actual)

        # Battery state
        battery_p = float(dev_p_dict.get(3, 0.0))
        battery_soc_mwh = float(full_state.get("des_soc", {}).get("MWh", {}).get(3, 0.0))

        # Battery cycling reward (asymmetric):
        #   Discharging is rewarded more than charging because it directly
        #   serves loads and reduces slack import.
        #   W_CYCLE has been removed — battery use reward replaces it entirely,
        #   with a net positive incentive to cycle the battery.
        discharge_mwh = max(0.0,  battery_p) * delta_t_h   # battery_p > 0 → discharging
        charge_mwh    = max(0.0, -battery_p) * delta_t_h   # battery_p < 0 → charging

        # SoC management:
        #   Reserve penalty  → keep some charge available at all times
        #   Ceiling penalty  → discourage staying near full (blocks future charging)
        #   Threshold lowered to 60 MWh (60% of 100 MWh capacity) so the agent
        #   is pushed to discharge well before the battery is completely full.
        SOC_RESERVE_MWH  = 10.0
        SOC_CEILING_MWH  = 60.0
        soc_reserve_shortfall_mwh = max(0.0, SOC_RESERVE_MWH - battery_soc_mwh) * delta_t_h
        soc_ceiling_excess_mwh    = max(0.0, battery_soc_mwh - SOC_CEILING_MWH) * delta_t_h

        # Reactive power penalty:
        #   Applied only when all bus voltages are within bounds.
        #   If voltages are violated, Q injection may be legitimately needed for
        #   support, so we relax the penalty to avoid conflicting signals.
        controllable_q_ids = [
            i for i in device_ids
            if type(device_data[i]).__name__ in ("RenewableGen", "StorageUnit")
        ]
        total_q_mvar = sum(abs(float(dev_q_dict.get(i, 0.0))) for i in controllable_q_ids)
        voltages_ok = bool(np.all((bus_v >= v_min) & (bus_v <= v_max)))
        q_penalty_factor = 1.0 if voltages_ok else 0.1
        q_magnitude_mvarh = total_q_mvar * delta_t_h * q_penalty_factor

        # ── Weights ───────────────────────────────────────────────────────────────
        W_CURTAIL       = 1.0   # per MWh curtailed
        W_SLACK_IMPORT  = 1.0   # per MWh imported from grid
        W_SLACK_EXPORT  = 0.1   # per MWh exported to grid (less bad than curtailment)
        W_SOC_RESERVE   = 0.5   # per MWh below reserve threshold
        W_SOC_CEILING   = 15.0  # per MWh above ceiling threshold (raised from 5 → 15)
        W_Q_MAGNITUDE   = 2.0   # per MVArh of unnecessary reactive power
        W_BRANCH_SOFT   = 3.0   # per MVAh above 80% loading (soft congestion signal)
        W_BRANCH_HARD   = 0.0   # intentionally 0: Lagrangian handles hard overloads
        W_DISCHARGE     = 0.1   # reward per MWh discharged
        W_CHARGE        = 0.02  # reward per MWh charged (smaller: charging is less urgent)

        curtailment_mwh   = curtailment_mw   * delta_t_h
        slack_import_mwh  = slack_import_mw  * delta_t_h
        slack_export_mwh  = slack_export_mw  * delta_t_h

        battery_use_reward = W_DISCHARGE * discharge_mwh + W_CHARGE * charge_mwh

        r = r - (
            W_CURTAIL      * curtailment_mwh
            + W_SLACK_IMPORT * slack_import_mwh
            + W_SLACK_EXPORT * slack_export_mwh
            + W_SOC_RESERVE  * soc_reserve_shortfall_mwh
            + W_SOC_CEILING  * soc_ceiling_excess_mwh
            + W_Q_MAGNITUDE  * q_magnitude_mvarh
            + W_BRANCH_SOFT  * branch_soft_penalty_mwh
            + W_BRANCH_HARD  * branch_overload_mwh   # always 0, kept for clarity
        ) + battery_use_reward

        # ── Info dict ─────────────────────────────────────────────────────────────
        info["objective_costs"] = np.array(
            [
                curtailment_mwh,
                slack_import_mwh,
                slack_export_mwh,
                discharge_mwh,           # replaces cycle_mwh (unsigned discharge)
                charge_mwh,              # separate charge tracking
                soc_reserve_shortfall_mwh,
                soc_ceiling_excess_mwh,
                q_magnitude_mvarh,
                branch_overload_mwh,
            ],
            dtype=np.float32,
        )
        info["objective/cost_curtailment_mwh"]          = float(curtailment_mwh)
        info["objective/cost_slack_import_mwh"]         = float(slack_import_mwh)
        info["objective/cost_slack_export_mwh"]         = float(slack_export_mwh)
        info["objective/reward_discharge_mwh"]          = float(discharge_mwh)
        info["objective/reward_charge_mwh"]             = float(charge_mwh)
        info["objective/cost_soc_reserve_shortfall_mwh"] = float(soc_reserve_shortfall_mwh)
        info["objective/cost_soc_ceiling_excess_mwh"]   = float(soc_ceiling_excess_mwh)
        info["objective/slack_p_mw"]                    = float(slack_p)
        info["objective/curtailment_mw"]                = float(curtailment_mw)
        info["objective/battery_soc_mwh"]               = float(battery_soc_mwh)
        info["objective/cost_q_magnitude_mvarh"]        = float(q_magnitude_mvarh)
        info["objective/q_penalty_factor"]              = float(q_penalty_factor)
        info["objective/cost_branch_soft_mvah"]         = float(branch_soft_penalty_mwh)
        info["objective/cost_branch_overload_mvah"]     = float(branch_overload_mwh)
        info["branch_loading_pu"]                       = branch_loading

        self.year_count = (self.date - self.date_init).days // 365

        return obs, r, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        render_mode = self.render_mode
        obs, info = super().reset(seed=seed, options=options)
        self.render_mode = render_mode
        self.year_count = 0
        if options is not None and "date_init" in options:
            self.date_init = options["date_init"]
        else:
            self.date_init = random_date(self.np_random, 2020)
        self.date = self.date_init
        return obs, info

    def reset_date(self, date_init):
        self.date_init = date_init
        self.date = date_init

    def _init_render(self, network_specs):
        title = type(self).__name__
        dev_type = list(network_specs["dev_type"].values())
        ps, qs = [], []
        for i in network_specs["dev_p"].keys():
            p_min_max = [network_specs["dev_p"][i]["MW"][j] for j in [0, 1]]
            ps.append(np.max(np.abs(p_min_max)))
            q_min_max = [network_specs["dev_q"][i]["MVAr"][j] for j in [0, 1]]
            qs.append(np.max(np.abs(q_min_max)))
        branch_rate = []
        for br in network_specs["branch_s"].keys():
            branch_rate.append(network_specs["branch_s"][br]["MVA"][1])
        bus_v_min, bus_v_max = [], []
        for i in network_specs["bus_v"].keys():
            bus_v_min.append(network_specs["bus_v"][i]["pu"][0])
            bus_v_max.append(network_specs["bus_v"][i]["pu"][1])
        soc_max = []
        for i in network_specs["des_soc"].keys():
            soc_max.append(network_specs["des_soc"][i]["MWh"][1])
        c1 = 100 if self.costs_clipping[0] is None else self.costs_clipping[0]
        c2 = 10000 if self.costs_clipping[1] is None else self.costs_clipping[1]
        self.http_server, self.ws_server = rendering.start(
            title, dev_type, ps, qs, branch_rate, bus_v_min, bus_v_max, soc_max, (c1, c2)
        )

    def _update_render(self, dev_p, dev_q, branch_s, des_soc, gen_p_max, bus_v_magn, costs, network_collapsed):
        rendering.update(
            self.ws_server.address, self.date, self.year_count,
            dev_p, dev_q, branch_s, des_soc, gen_p_max, bus_v_magn, costs, network_collapsed,
        )

    def close(self):
        if self.is_rendering:
            try:
                rendering.close(self.http_server, self.ws_server)
            except AttributeError:
                pass
        self.render_mode = None