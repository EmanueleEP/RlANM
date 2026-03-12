"""The base class for a 6-bus and 7-device :code:`gym-anm` environment."""

import datetime as dt
import numpy as np
from typing import Optional

from gym_anm import ANMEnv
from .rendering.py import rendering
from .network import network
from .utils import random_date


class BCP_base(ANMEnv):
    """

    The structure of the electricity distribution network used for this
    environment is shown below:

    Slack ---
            |
         -----------------
         |         |     | 
        battery   Load   PV

    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, observation, K, delta_t, gamma, lamb, aux_bounds=None, costs_clipping=(None, None), seed=None):

        super().__init__(network, observation, K, delta_t, gamma, lamb, aux_bounds, costs_clipping, seed)

        # Rendering variables.
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
            # Build lists ordered by IDs so renderer maps values to the correct devices/branches
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

        # Increment the date for rendering
        self.date += self.timestep_length

        # Extract state
        full_state = self.simulator.state
        gen_p_max = np.array(list(full_state["gen_p_max"]["MW"].values()))
        dev_p = np.array(list(full_state["dev_p"]["MW"].values()))
        dev_q = np.array(list(full_state["dev_q"]["MVAr"].values()))

        device_data = self.simulator.devices
        base_mva = self.simulator.baseMVA

        device_type = np.array(
            [getattr(dev, 'dev_type', 1) for dev in device_data.values()], dtype=np.int32
        )

        p_max = np.array([dev.p_max for dev in device_data.values()], dtype=np.float32) * base_mva
        p_min = np.array([dev.p_min for dev in device_data.values()], dtype=np.float32) * base_mva
        q_max = np.array([dev.q_max for dev in device_data.values()], dtype=np.float32) * base_mva
        q_min = np.array([dev.q_min for dev in device_data.values()], dtype=np.float32) * base_mva

        # Constraint violations for Lagrangian
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

        bus_v = np.array(list(full_state["bus_v_magn"]["pu"].values()))
        bus_objects = list(self.simulator.buses.values())
        v_max = np.array([getattr(b, 'v_max', 1.1) for b in bus_objects], dtype=float)
        v_min = np.array([getattr(b, 'v_min', 0.9) for b in bus_objects], dtype=float)
        v_violation = np.maximum(0, bus_v - v_max) + np.maximum(0, v_min - bus_v)

        info["constraint_costs"] = np.array([
            np.mean(v_violation),
            np.mean(p_violation) / base_mva,
            np.mean(q_violation) / base_mva,
        ], dtype=np.float32)

        # General reward shaping

        # Penalise curtailment of ANY local generator (wasted local energy).
        #    gen_p_max has one entry per generator; dev_p is indexed over all devices.
        gen_dev_ids = [
            i for i, dev in enumerate(device_data.values())
            if getattr(dev, 'dev_type', 0) == 2  # dev_type=2 means non-controllable generator
        ]
        if len(gen_dev_ids) > 0:
            curtailment = np.maximum(0, gen_p_max - dev_p[gen_dev_ids])
        else:
            curtailment = np.zeros(1)
        curtailment_penalty = 0.5 * np.sum(curtailment)

        # Penalise importing power from the slack bus (grid dependency).
        #    Slack is always device 0. Positive dev_p[0] means importing from grid.
        slack_import = max(0.0, float(dev_p[0]))
        import_penalty = 0.3 * slack_import

        # Get storage device IDs (needed for both battery reward and SOC penalty)
        des_soc = list(full_state["des_soc"]["MWh"].values())
        storage_dev_ids = [
            i for i, dev in enumerate(device_data.values())
            if getattr(dev, 'dev_type', 0) == 3  # dev_type=3 means storage
        ]

        # Get renewable generation (device 2 is solar/renewable)
        renewable_power = float(dev_p[2]) if len(dev_p) > 2 else 0.0

        # Reward battery usage: prioritize charging renewable, incentivize discharge
        battery_reward = 0.0
        for dev_id in storage_dev_ids:
            battery_power = float(dev_p[dev_id])
            if battery_power < 0:  # Charging (negative P = storing energy)
                if renewable_power > 0:
                    # High reward for charging with renewable energy (store for later use)
                    renewable_ratio = min(1.0, abs(battery_power) / max(renewable_power, 0.1))
                    battery_reward += 0.8 * abs(battery_power) * renewable_ratio
                else:
                    # Lower reward for charging from grid (but allow it for flexibility)
                    battery_reward += 0.2 * abs(battery_power)
            else:  # Discharging (positive P = grid injection)
                # Discharge is valuable: reduces import and provides energy
                battery_reward += 0.6 * abs(battery_power)

        # SoC readiness: penalise storage being near-empty or near-full.
        soc_penalty = 0.0
        for idx, dev_id in enumerate(storage_dev_ids):
            soc = des_soc[idx]
            soc_max_mwh = self.simulator.devices[list(device_data.keys())[dev_id]].soc_max * base_mva
            soc_ratio = soc / soc_max_mwh if soc_max_mwh > 0 else 0.0
            # Parabola centered at 50%: penalises extremes (0% and 100%)
            soc_penalty += -0.3 * (soc_ratio - 0.5) ** 2

        # Increment the year count.
        self.year_count = (self.date - self.date_init).days // 365

        r = r - curtailment_penalty - import_penalty + battery_reward + soc_penalty

        return obs, r, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Save rendering setup to restore after the reset().
        render_mode = self.render_mode

        obs, info = super().reset(seed=seed, options=options)

        # Restore the rendering setup.
        self.render_mode = render_mode

        # Reset the date (for rendering).
        self.year_count = 0
        if options is not None and "date_init" in options:
            self.date_init = options["date_init"]
        else:
            self.date_init = random_date(self.np_random, 2020)
        self.date = self.date_init

        return obs, info

    def reset_date(self, date_init):
        """Reset the date displayed in the visualization (and the year count)."""
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
        costs_range = (c1, c2)

        self.http_server, self.ws_server = rendering.start(
            title, dev_type, ps, qs, branch_rate, bus_v_min, bus_v_max, soc_max, costs_range
        )

    def _update_render(self, dev_p, dev_q, branch_s, des_soc, gen_p_max, bus_v_magn, costs, network_collapsed):
        rendering.update(
            self.ws_server.address,
            self.date,
            self.year_count,
            dev_p,
            dev_q,
            branch_s,
            des_soc,
            gen_p_max,
            bus_v_magn,
            costs,
            network_collapsed,
        )

    def close(self):
        if self.is_rendering:
            try:
                rendering.close(self.http_server, self.ws_server)
            except AttributeError:
                pass
        self.render_mode = None