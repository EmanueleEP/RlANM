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
        """
        Render the current state of the environment.

        Visualizing the agent-environment interactions in real-time (e.g.,
        during training) is hard to follow and not very useful, as the state of
        the distribution network changes too quickly (you can try with
        :code:`mode='human'` and :code`skip_frames=0`). Instead, setting :code:`skip_frames>0`
        will only update the rendering of the environment every `skip_frames`+1
        steps (assuming :code:`render(skip_frames)` is called after every step),
        which will make it much easier to follow for the human eye.

        Parameters
        ----------
        mode : {'human'}, optional
            The mode of rendering. If 'human', the environment is rendered while
            the agent interacts with it.
        skip_frames : int, optional
            The number of frames (steps) to skip when rendering the environment.
            For example, :code:`skip_frames=3` will update the rendering of the
            environment every 4 calls to :py:func:`render()`.

        Raises
        ------
        NotImplementedError
            If a non-valid mode is specified.

        Notes
        -----
        1. The use of :code:`mode='human'` and :code:`skip_frames>0` assumes that :py:func`render()`
           is called after each step the agent takes in the environment.
           The same behavior can be achieved with :code:`skip_frames=0` and calling
           :py:func`render()` less frequently.
        """

        if self.render_mode is None:
            if mode not in ["human"]:
                raise NotImplementedError()

            # Render the initial image of the distribution network.
            self.render_mode = mode
            self.skipped_frames = 0
            rendered_network_specs = ["dev_type", "dev_p", "dev_q", "branch_s", "bus_v", "des_soc"]
            specs = {s: self.network_specs[s] for s in rendered_network_specs}
            self._init_render(specs)

            # Render the initial state.
            self.render(mode=mode, skip_frames=skip_frames)
            self.is_rendering = True

        else:
            self.skipped_frames = (self.skipped_frames + 1) % (skip_frames + 1)
            if self.skipped_frames:
                return

            full_state = self.simulator.state
            dev_p = list(full_state["dev_p"]["MW"].values())
            dev_q = list(full_state["dev_q"]["MVAr"].values())
            branch_s = list(full_state["branch_s"]["MVA"].values())
            des_soc = list(full_state["des_soc"]["MWh"].values())
            gen_p_max = list(full_state["gen_p_max"]["MW"].values())
            bus_v_magn = list(full_state["bus_v_magn"]["pu"].values())
            costs = [self.e_loss, self.penalty]
            network_collapsed = not self.simulator.pfe_converged

            self._update_render(dev_p, dev_q, branch_s, des_soc, gen_p_max, bus_v_magn, costs, network_collapsed)

    def safe_array(self, x):
        return np.array([np.nan if v is None else float(v) for v in x])

    def step(self, action):
        obs, r, terminated, truncated, info = super().step(action)

        #print(dir(self.simulator))

        # Increment the date (for rendering).
        self.date += self.timestep_length

        # extraction info from state
        full_state = self.simulator.state

        gen_p_max = np.array(list(full_state["gen_p_max"]["MW"].values()))
        dev_p = np.array(list(full_state["dev_p"]["MW"].values()))
        dev_q = np.array(list(full_state["dev_q"]["MVAr"].values()))

        storage_device_ids = [3] # battery index
        curtailment = np.maximum(0, gen_p_max - dev_p[[2]])

        # curtailment penalty
        curtailment_penalty = 0.5 * np.sum(curtailment)

        # MW currently being wasted
        pv_surplus = float(np.sum(curtailment))  

        device_data = self.simulator.devices
        base_mva = self.simulator.baseMVA  
        
        device_type = np.array([getattr(dev, 'dev_type', 1) for dev in device_data.values()],dtype=np.int32)

        p_max = np.array([dev.p_max for dev in device_data.values()], dtype=np.float32) * base_mva
        p_min = np.array([dev.p_min for dev in device_data.values()], dtype=np.float32) * base_mva

        q_max = np.array([dev.q_max for dev in device_data.values()], dtype=np.float32) * base_mva
        q_min = np.array([dev.q_min for dev in device_data.values()], dtype=np.float32) * base_mva
        
        p_violation = np.zeros_like(dev_p)
        q_violation = np.zeros_like(dev_q)

        # active violations
        for i in range(len(dev_p)):
            if device_type[i] == 0:
                continue
            if not np.isnan(p_max[i]):
                p_violation[i] += max(0, dev_p[i] - p_max[i])
            if not np.isnan(p_min[i]):
                p_violation[i] += max(0, p_min[i] - dev_p[i])

        # reactive violations
        for i in range(len(dev_q)):
            if device_type[i] == 0:
                continue
            if not np.isnan(q_max[i]):
                q_violation[i] += max(0, dev_q[i] - q_max[i])
            if not np.isnan(q_min[i]):
                q_violation[i] += max(0, q_min[i] - dev_q[i])

        # voltage violations
        bus_v = np.array(list(full_state["bus_v_magn"]["pu"].values()))

        bus_objects = list(self.simulator.buses.values())
        v_max = np.array([getattr(b, 'v_max', 1.1) for b in bus_objects], dtype=float)
        v_min = np.array([getattr(b, 'v_min', 0.9) for b in bus_objects], dtype=float)

        v_violation = np.maximum(0, bus_v - v_max) + \
                    np.maximum(0, v_min - bus_v)

        info["constraint_costs"] = np.array([
            np.mean(v_violation),
            np.mean(p_violation) / base_mva,
            np.mean(q_violation) / base_mva,
        ], dtype=np.float32)

        # battery absorption
        battery_charging = 0.0
        for dev_id in storage_device_ids:
            p_battery = dev_p[dev_id]
            if p_battery < 0:
                battery_charging += abs(p_battery)
        
        absorbed = min(battery_charging, pv_surplus)
        self_consumption_bonus = 0.4 * absorbed

        # Increment the year count.
        self.year_count = (self.date - self.date_init).days // 365

        # make battery more active
        des_soc = list(full_state["des_soc"]["MWh"].values())
        for dev_id in storage_device_ids:
            soc = des_soc[0]
            soc_max = self.simulator.devices[dev_id].soc_max * self.simulator.baseMVA
            soc_ratio = soc / soc_max if soc_max > 0 else 0.0
            soc_readiness = -0.3 * (soc_ratio - 0.5) ** 2
            r += soc_readiness

        # setting penalty for wasting resources
        r = r - curtailment_penalty + self_consumption_bonus

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
        """
        Initialize the rendering of the environment state.

        Parameters
        ----------
        network_specs : dict of {str : list}
            The operating characteristics of the electricity distribution network.
        """

        # Set visualization title to class name.
        title = type(self).__name__

        # Convert dict of network specs into lists.
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

        # Set default costs range if not specified.
        c1 = 100 if self.costs_clipping[0] is None else self.costs_clipping[0]
        c2 = 10000 if self.costs_clipping[1] is None else self.costs_clipping[1]
        costs_range = (c1, c2)

        self.http_server, self.ws_server = rendering.start(
            title, dev_type, ps, qs, branch_rate, bus_v_min, bus_v_max, soc_max, costs_range
        )

    def _update_render(self, dev_p, dev_q, branch_s, des_soc, gen_p_max, bus_v_magn, costs, network_collapsed):
        """
        Update the rendering of the environment state.

        Parameters
        ----------
        dev_p  : list of float
            The real power injection from each device (MW).
        dev_q : list of float
            The reactive power injection from each device (MW).
        branch_s : list of float
            The apparent power flow in each branch (MVA).
        des_soc : list of float
            The state of charge of each storage unit (MWh).
        gen_p_max : list of float
            The potential real power generation of each RER generator before
            curtailment (MW).
        bus_v_magn : list of float
            The voltage magnitude of each bus (pu).
        costs : list of float
            The total energy loss and the total penalty associated with operating
            constraints violation.
        network_collapsed : bool
            True if no load flow solution is found (possibly infeasible); False
            otherwise.
        """
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
        """
        Close the rendering.
        """
        # Close the rendering if it is currently active
        if self.is_rendering:
            try:
                rendering.close(self.http_server, self.ws_server)
            except AttributeError:
                pass
        self.render_mode = None
