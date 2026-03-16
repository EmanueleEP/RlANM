"""The base class for a 6-bus and 7-device :code:`gym-anm` environment."""

import datetime as dt
import numpy as np
from typing import Optional

from gym_anm import ANMEnv
from .rendering.py import rendering
from .network import network as network_default
from .utils import random_date


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

        device_type = np.array(
            [getattr(device_data[i], "dev_type", 1) for i in device_ids], dtype=np.int32
        )

        p_max = np.array([device_data[i].p_max for i in device_ids], dtype=np.float32) * base_mva
        p_min = np.array([device_data[i].p_min for i in device_ids], dtype=np.float32) * base_mva
        q_max = np.array([device_data[i].q_max for i in device_ids], dtype=np.float32) * base_mva
        q_min = np.array([device_data[i].q_min for i in device_ids], dtype=np.float32) * base_mva

        p_violation = np.zeros_like(dev_p)
        q_violation = np.zeros_like(dev_q)

        for i in range(len(dev_p)):
            if device_type[i] == 0:
                continue
            if not np.isnan(p_max[i]):
                p_violation[i] += max(0, dev_p[i] - p_max[i])
            if not np.isnan(p_min[i]):
                p_violation[i] += max(0, p_min[i] - dev_p[i])

        for i in range(len(dev_q)):
            if device_type[i] == 0:
                continue
            if not np.isnan(q_max[i]):
                q_violation[i] += max(0, dev_q[i] - q_max[i])
            if not np.isnan(q_min[i]):
                q_violation[i] += max(0, q_min[i] - dev_q[i])

        bus_ids = sorted(self.simulator.buses.keys())
        bus_v = np.array([bus_v_dict.get(i, 1.0) for i in bus_ids], dtype=float)
        bus_objects = [self.simulator.buses[i] for i in bus_ids]
        v_max = np.array([getattr(b, 'v_max', 1.1) for b in bus_objects], dtype=float)
        v_min = np.array([getattr(b, 'v_min', 0.9) for b in bus_objects], dtype=float)
        v_violation = np.maximum(0, bus_v - v_max) + np.maximum(0, v_min - bus_v)

        info["constraint_costs"] = np.array([
            np.max(v_violation) if len(v_violation) > 0 else 0.0,
            np.max(p_violation) / base_mva if len(p_violation) > 0 else 0.0,
            np.max(q_violation) / base_mva if len(q_violation) > 0 else 0.0,
        ], dtype=np.float32)
        info["pfe_converged"] = bool(getattr(self.simulator, "pfe_converged", True))


        # objective:
        #   - minimize renewable curtailment (true energy waste)
        #   - maximize self-sufficiency (reduce reliance on slack/grid interaction)

        delta_t_h = float(getattr(self, "delta_t", 0.25))
        slack_p = float(dev_p_dict.get(0, 0.0))
        
        # Slack interaction (asymmetric):
        # - import (slack_p > 0): drawing from the grid (hurts self-sufficiency)
        # - export (slack_p < 0): sending surplus to the grid (less bad than curtailment)
        slack_import_mw = max(0.0, slack_p)
        slack_export_mw = max(0.0, -slack_p)

        # Curtailment derived from potential renewable generation (gen_p_max) vs actual injection (dev_p).
        # Only sum devices with DEV_TYPE == 2 (wind/solar).
        renewable_ids = [i for i in device_ids if getattr(device_data[i], "dev_type", None) == 2]
        curtailment_mw = 0.0
        for dev_id in renewable_ids:
            if dev_id not in gen_p_max_dict:
                continue
            p_potential = float(gen_p_max_dict.get(dev_id, 0.0))
            p_actual = float(dev_p_dict.get(dev_id, 0.0))
            curtailment_mw += max(0.0, p_potential - p_actual)

        # Optional: tiny cycling penalty to prevent pointless oscillations.
        battery_p = float(dev_p_dict.get(3, 0.0))
        battery_soc_mwh = float(full_state.get("des_soc", {}).get("MWh", {}).get(3, 0.0))

        # Weights are in "reward per MWh".
        W_CURTAIL = 1.0
        W_SLACK_IMPORT = 2.0
        W_SLACK_EXPORT = 0.2
        W_CYCLE = 0.01
        W_SOC_RESERVE = 2.0
        # start penalizing when battery SoC is below the following MWh to encourage maintaining a reserve for reliability
        SOC_RESERVE_MWH = 5.0
        W_SOC_CEILING = 2.0
        # start penalizing when battery SoC is above the following MWh to discourage hoarding / near-full saturation
        SOC_CEILING_MWH = 90.0

        curtailment_mwh = curtailment_mw * delta_t_h
        slack_import_mwh = slack_import_mw * delta_t_h
        slack_export_mwh = slack_export_mw * delta_t_h
        cycle_mwh = abs(battery_p) * delta_t_h
        soc_reserve_shortfall_mwh = max(0.0, SOC_RESERVE_MWH - battery_soc_mwh)
        soc_ceiling_excess_mwh = max(0.0, battery_soc_mwh - SOC_CEILING_MWH)

        r = r - (
            W_CURTAIL * curtailment_mwh
            + W_SLACK_IMPORT * slack_import_mwh
            + W_SLACK_EXPORT * slack_export_mwh
            + W_CYCLE * cycle_mwh
            + W_SOC_RESERVE * soc_reserve_shortfall_mwh
            + W_SOC_CEILING * soc_ceiling_excess_mwh
        )

        info["objective_costs"] = np.array(
            [curtailment_mwh, slack_import_mwh, slack_export_mwh, cycle_mwh, soc_reserve_shortfall_mwh, soc_ceiling_excess_mwh],
            dtype=np.float32,
        )
        info["objective/cost_curtailment_mwh"] = float(curtailment_mwh)
        info["objective/cost_slack_import_mwh"] = float(slack_import_mwh)
        info["objective/cost_slack_export_mwh"] = float(slack_export_mwh)
        info["objective/cost_cycle_mwh"] = float(cycle_mwh)
        info["objective/cost_soc_reserve_shortfall_mwh"] = float(soc_reserve_shortfall_mwh)
        info["objective/cost_soc_ceiling_excess_mwh"] = float(soc_ceiling_excess_mwh)
        info["objective/slack_p_mw"] = float(slack_p)
        info["objective/curtailment_mw"] = float(curtailment_mw)
        info["objective/battery_soc_mwh"] = float(battery_soc_mwh)

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
