"""The :code:`BCP-v0` task."""

import numpy as np

from .bcp import BCP_base


class BCP(BCP_base):
    """The :code:`BCP-v0` task."""

    def __init__(self):
        observation = "state"  # fully observable environment
        K = 1
        delta_t = 0.25  # 15 minutes between timesteps
        gamma = 0.995
        lamb = 100
        aux_bounds = np.array([[0, 24 / delta_t - 1]])
        costs_clipping = (1, 100)
        super().__init__(observation, K, delta_t, gamma, lamb, aux_bounds, costs_clipping)

        # Consumption and maximum generation 24-hour time series.
        self.P_loads = _get_load_time_series()
        self.P_maxs = _get_gen_time_series()

    def init_state(self):
        n_dev, n_gen, n_des = 4, 1, 1

        state = np.zeros(2 * n_dev + n_des + n_gen + self.K)

        t_0 = self.np_random.integers(0, int(24 / self.delta_t))
        state[-1] = t_0

        # --- Load (P, Q) injections ---
        load_device_ids = [1]  # IDs in state vector
        for idx, dev_id in enumerate(load_device_ids):
            p_load = self.P_loads[idx]  # row = device
            state[dev_id] = p_load[t_0]
            state[n_dev + dev_id] = p_load[t_0] * self.simulator.devices[dev_id].qp_ratio

        # --- Non-slack generator (P, Q) injections ---
        gen_device_ids = [2]
        for idx, dev_id in enumerate(gen_device_ids):
            p_max = self.P_maxs[idx]  # row = generator device
            state[2 * n_dev + n_des + idx] = p_max[t_0]
            state[dev_id] = p_max[t_0]
            state[n_dev + dev_id] = self.np_random.uniform(
                self.simulator.devices[dev_id].q_min, self.simulator.devices[dev_id].q_max
            )

        # --- Energy storage unit ---
        storage_device_ids = [3]
        for idx, dev_id in enumerate(storage_device_ids):
            state[2 * n_dev + idx] = self.np_random.uniform(
                self.simulator.devices[dev_id].soc_min, self.simulator.devices[dev_id].soc_max
            )

        return state

    def next_vars(self, s_t):
        aux = int((s_t[-1] + 1) % (24 / self.delta_t))

        vars = []
        for p_load in self.P_loads:
            vars.append(p_load[aux])
        for p_max in self.P_maxs:
            vars.append(p_max[aux])

        vars.append(aux)

        return np.array(vars)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)

        # Reset the time of the day based on the auxiliary variable.
        new_date = self.date + self.state[-1] * self.timestep_length
        super().reset_date(new_date)

        return obs, info


def _get_load_time_series():
    s1 = -np.ones(25); s12 = np.linspace(-1.5, -4.5, 7)
    s2 = -5 * np.ones(13); s23 = np.linspace(-4.625, -2.375, 7)
    s3 = -2 * np.ones(13)
    P1 = np.concatenate((s1, s12, s2, s23, s3, s23[::-1], s2, s12[::-1], s1[:4]))
    P_loads = P1.reshape(1, -1)  # shape = (1, 96)
    assert P_loads.shape[1] == 96
    return P_loads



def _get_gen_time_series():
    s1 = np.zeros(25); s12 = np.linspace(0.5, 3.5, 7)
    s2 = 4 * np.ones(13); s23 = np.linspace(7.25, 36.75, 7)
    s3 = 30 * np.ones(13)
    P2 = np.concatenate((s1, s12, s2, s23, s3, s23[::-1], s2, s12[::-1], s1[:4]))
    P_maxs = P2.reshape(1, -1)  # shape = (1, 96)
    assert P_maxs.shape[1] == 96
    return P_maxs



if __name__ == "__main__":
    import time

    env = BCP()
    env.reset()
    print("Environment reset and ready.")

    T = 50
    start = time.time()
    for i in range(T):
        print(i)
        a = env.action_space.sample()
        o, r, _, _, _ = env.step(a)
        env.render()
        time.sleep(0.5)

    print("Done with {} steps in {} seconds!".format(T, time.time() - start))
