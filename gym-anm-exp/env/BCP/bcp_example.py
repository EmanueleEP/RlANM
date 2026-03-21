"""The :code:`BCP-v0` task."""

import numpy as np
import pandas as pd

from .bcp import BCP_base

profiles_path = "env/BCP/daysHoursPQ.txt"


try:
    from rl_agents.hyperparameters import USE_LAGRANGIAN as _USE_LAGRANGIAN
except ImportError:
    _USE_LAGRANGIAN = False
 
_LAMB = 0 if _USE_LAGRANGIAN else 100


class BCP(BCP_base):
    """The :code:`BCP-v0` task."""

    def __init__(self):
        observation = "state"  # fully observable environment
        K = 1
        delta_t = 0.25  # 15 minutes between timesteps
        gamma = 0.995
        lamb = _LAMB # penalty weight 
        aux_bounds = np.array([[0, 24 / delta_t - 1]])
        costs_clipping = (1, 100)
        super().__init__(observation, K, delta_t, gamma, lamb, aux_bounds, costs_clipping)

        # Consumption and maximum generation 24-hour time series.
        self.P_loads = _get_load_time_series(0.1, 123)
        self.P_maxs = _get_gen_time_series(0.5, 123)
        
        # # load from file
        # P_loads, P_maxs, days = _load_profiles_from_csv(profiles_path, delta_t=self.delta_t)
        # self.P_loads = P_loads
        # self.P_maxs  = P_maxs
        # self._n_days = len(days)
        # # Total timesteps available across all days
        # self._total_steps = self.P_loads.shape[1]

    def init_state(self):
        n_dev, n_gen, n_des = 4, 1, 1

        state = np.zeros(2 * n_dev + n_des + n_gen + self.K)

        # old way
        t_0 = self.np_random.integers(0, int(24 / self.delta_t))
        
        # from file 
        # t_0 = self.np_random.integers(0, self._total_steps)
        state[-1] = t_0

        

        # Load (P, Q) injections
        load_device_ids = [1]  # IDs in state vector
        for idx, dev_id in enumerate(load_device_ids):
            p_load = self.P_loads[idx]  # row = device
            state[dev_id] = p_load[t_0]
            state[n_dev + dev_id] = p_load[t_0] * self.simulator.devices[dev_id].qp_ratio

        # Non-slack generator (P, Q) injections
        gen_device_ids = [2]
        for idx, dev_id in enumerate(gen_device_ids):
            p_max = self.P_maxs[idx]  # row = generator device
            state[2 * n_dev + n_des + idx] = p_max[t_0]
            state[dev_id] = p_max[t_0]
            state[n_dev + dev_id] = self.np_random.uniform(
                self.simulator.devices[dev_id].q_min, self.simulator.devices[dev_id].q_max
            )

        # Energy storage unit
        storage_device_ids = [3]
        for idx, dev_id in enumerate(storage_device_ids):
            # IMPORTANT: `gym-anm` expects SoC in the init_state vector in **MWh**
            # (Simulator.reset() divides by baseMVA to convert to p.u.).
            soc_min_mwh = float(self.simulator.devices[dev_id].soc_min) * float(self.simulator.baseMVA)
            soc_max_mwh = float(self.simulator.devices[dev_id].soc_max) * float(self.simulator.baseMVA)

            # Randomize initial SoC between 1% and 60% of capacity during training.
            # This ensures the agent experiences diverse battery states and learns
            # both charging and discharging strategies.
            initial_soc_mwh = float(self.np_random.uniform(
                0.01 * soc_max_mwh, 0.6 * soc_max_mwh
            ))
            initial_soc_mwh = float(np.clip(initial_soc_mwh, soc_min_mwh, soc_max_mwh))

            state[2 * n_dev + idx] = initial_soc_mwh  # SoC in MWh

        return state

    def next_vars(self, s_t):
        # old way
        aux = int((s_t[-1] + 1) % (24 / self.delta_t))

        # DEBUG
        #print(f"DEBUG t={aux} ({aux * self.delta_t:.1f}h), p_max={self.P_maxs[0][aux]:.2f} MW")
        
        # from file
        # aux = int((s_t[-1] + 1) % self._total_steps)  

        vars = []
        for p_load in self.P_loads:
            vars.append(p_load[aux])
        for p_max in self.P_maxs:
            vars.append(p_max[aux])

        vars.append(aux)

        return np.array(vars)

    def reset(self, **kwargs):
        # Call parent reset first to initialize simulator state
        obs, info = super().reset(**kwargs)

        # Reset the time of the day based on the auxiliary variable.
        new_date = self.date + self.state[-1] * self.timestep_length
        super().reset_date(new_date)

        return obs, info

# 1 - added gaussian noise
def _get_load_time_series(noise_std=0.1, seed=None):
    rng = np.random.default_rng(seed)
    s1 = -np.ones(25); s12 = np.linspace(-1.5, -4.5, 7)
    s2 = -5 * np.ones(13); s23 = np.linspace(-4.625, -2.375, 7)
    s3 = -2 * np.ones(13)
    P1 = np.concatenate((s1, s12, s2, s23, s3, s23[::-1], s2, s12[::-1], s1[:4]))
    P1 += rng.normal(0, noise_std, size=P1.shape) # add gaussian noise
    P_loads = P1.reshape(1, -1)  # shape = (1, 96)
    assert P_loads.shape[1] == 96
    return P_loads


# 1 - added gaussian noise
def _get_gen_time_series(noise_std=0.1, seed=None):
    rng = np.random.default_rng(seed)
    s1 = np.zeros(25); s12 = np.linspace(0.5, 3.5, 7)
    s2 = 4 * np.ones(13); s23 = np.linspace(7.25, 36.75, 7)
    s3 = 30 * np.ones(13)
    P2 = np.concatenate((s1, s12, s2, s23, s3, s23[::-1], s2, s12[::-1], s1[:4]))
    P2 += rng.normal(0, noise_std, size=P2.shape) # add gaussian noise
    P_maxs = P2.reshape(1, -1)  # shape = (1, 96)
    assert P_maxs.shape[1] == 96
    return P_maxs
    #return _get_profile_solar_pv()

def _get_profile_solar_pv(delta_t=0.25, peak_mw=30.0, peak_hour=14.0, width=3.5):
    # width      : std dev of the Gaussian in hours (controls ramp steepness)
    
    steps = int(24 / delta_t)
    t = np.linspace(0, 24, steps, endpoint=False)
    P_max = peak_mw * np.exp(-0.5 * ((t - peak_hour) / width) ** 2)
    P_max = np.clip(P_max, 0, None)
    return P_max.reshape(1, -1)


def _load_profiles_from_csv(filepath: str, delta_t: float = 0.25):

    df = pd.read_csv(filepath)

    _validate_columns(df)

    steps_per_hour = int(1 / delta_t)   
    steps_per_day  = int(24 / delta_t)

    days = sorted(df["day"].unique())
    all_loads, all_maxs = [], []

    for day in days:
        day_df = df[df["day"] == day].sort_values("hour").reset_index(drop=True)

        _validate_day(day_df, day)

        # hours arrays
        p_load_hourly = day_df["p_load"].values.astype(float)
        p_max_hourly  = day_df["p_max"].values.astype(float)

        # Interpolate to sub-hourly resolution
        p_load_fine = _interpolate_to_steps(p_load_hourly, steps_per_hour, steps_per_day)
        p_max_fine  = _interpolate_to_steps(p_max_hourly,  steps_per_hour, steps_per_day)

        all_loads.append(p_load_fine)
        all_maxs.append(p_max_fine)

    # Stack days into one long series
    P_loads = np.concatenate(all_loads).reshape(1, -1)
    P_maxs  = np.concatenate(all_maxs).reshape(1, -1)

    return P_loads, P_maxs, days


def _interpolate_to_steps(hourly: np.ndarray, steps_per_hour: int, steps_per_day: int) -> np.ndarray:
    """Linear interpolation from 24 hourly values to steps_per_day sub-hourly values."""
    # Append first value at end to allow interpolation through hour 23
    hourly_ext = np.append(hourly, hourly[0])
    x_hourly   = np.arange(25)                          # 0..24 (inclusive)
    x_fine     = np.linspace(0, 24, steps_per_day, endpoint=False)
    return np.interp(x_fine, x_hourly, hourly_ext)


def _validate_columns(df: pd.DataFrame):
    required = {"day", "hour", "p_load", "p_max"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"profiles.csv is missing columns: {missing}")


def _validate_day(day_df: pd.DataFrame, day: int):
    hours = set(day_df["hour"].values)
    expected = set(range(24))
    if hours != expected:
        missing = expected - hours
        raise ValueError(f"Day {day} is missing hours: {sorted(missing)}")
    if (day_df["p_load"] > 0).any():
        raise ValueError(f"Day {day} has positive p_load values (loads must be <= 0).")
    if (day_df["p_max"] < 0).any():
        raise ValueError(f"Day {day} has negative p_max values (generation must be >= 0).")



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
