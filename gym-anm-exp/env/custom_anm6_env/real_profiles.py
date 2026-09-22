"""ANM6 environment driven by split-specific real-data profile banks."""

from pathlib import Path

import numpy as np

from .anm6_custom import ANM6_Custom


SPLITS = {"train", "validation", "test"}


def _default_profile_path(split):
    repository_parent = Path(__file__).resolve().parents[4]
    return repository_parent / "Datasets" / "processed" / "real_profiles_{}.npz".format(split)


def _resolve_profile_path(profile_path, split):
    if profile_path is None:
        return _default_profile_path(split)
    path = Path(profile_path).expanduser()
    if path.is_dir():
        return path / "real_profiles_{}.npz".format(split)
    if "{split}" in str(path):
        return Path(str(path).format(split=split))
    return path


class RealProfileANM6(ANM6_Custom):
    """ANM6 using DEI loads, Padova PVGIS PV and T1 wind profiles."""

    def __init__(
        self,
        profile_path=None,
        split="train",
        profile_seed=None,
        solar_scale=1.0,
        load_scale=1.0,
    ):
        if split not in SPLITS:
            raise ValueError("split must be one of {}".format(sorted(SPLITS)))
        self.profile_split = split
        self.profile_seed = profile_seed
        self.solar_scale = float(solar_scale)
        self.load_scale = float(load_scale)
        if self.solar_scale <= 0.0 or self.load_scale <= 0.0:
            raise ValueError("solar_scale and load_scale must be positive")
        self.profile_path = _resolve_profile_path(profile_path, split).resolve()
        if not self.profile_path.is_file():
            raise FileNotFoundError(
                "Real profile bank not found: {}. Run data_pipeline.build_profile_bank "
                "or pass --profile-path.".format(self.profile_path)
            )
        with np.load(self.profile_path, allow_pickle=False) as archive:
            loads = np.asarray(archive["load_p_mw"], dtype=np.float64)
            generation = np.asarray(archive["gen_p_max_mw"], dtype=np.float64)
            qp_ratios = np.asarray(archive["load_qp_ratios"], dtype=np.float64)
        if loads.ndim != 3 or loads.shape[1:] != (3, 96):
            raise ValueError("load_p_mw must have shape (days, 3, 96)")
        if generation.shape != (loads.shape[0], 2, 96):
            raise ValueError("gen_p_max_mw must have shape (days, 2, 96)")
        if qp_ratios.shape != (3,) or not np.all(np.isfinite(qp_ratios)):
            raise ValueError("load_qp_ratios must contain three finite values")
        if np.any(loads > 1e-9) or np.any(generation < -1e-9):
            raise ValueError("Invalid injection signs in profile bank")

        loads = loads * self.load_scale
        generation[:, 0, :] *= self.solar_scale
        self.P_loads = loads.transpose(1, 0, 2).reshape(3, -1)
        self.P_maxs = generation.transpose(1, 0, 2).reshape(2, -1)
        self.profile_length = self.P_loads.shape[1]
        self.profile_days = loads.shape[0]

        # Same task parameters as ANM6Easy-v0 (see anm6_easy_custom.py for why
        # they are not in hyperparameters.py).
        observation = "state"
        K = 1
        delta_t = 0.25
        gamma = 0.995
        lamb = 100
        aux_bounds = np.array([[0, self.profile_length - 1]])
        costs_clipping = (1, 100)
        super().__init__(observation, K, delta_t, gamma, lamb, aux_bounds, costs_clipping)

        # Gym-ANM maps each load's Q from P with a fixed ratio. Update the
        # associated feasible Q interval consistently with the measured ratio.
        for dev_id, ratio in zip((1, 3, 5), qp_ratios):
            device = self.simulator.devices[dev_id]
            device.qp_ratio = float(ratio)
            device.q_min = device.p_min * device.qp_ratio
            device.q_max = device.p_max * device.qp_ratio

    def init_state(self):
        n_dev, n_gen, n_des = 7, 2, 1
        state = np.zeros(2 * n_dev + n_des + n_gen + self.K)
        day = int(self.np_random.integers(0, self.profile_days))
        quarter = int(self.np_random.integers(0, 96))
        t_0 = day * 96 + quarter
        state[-1] = t_0

        for dev_id, p_load in zip((1, 3, 5), self.P_loads):
            state[dev_id] = p_load[t_0]
            state[n_dev + dev_id] = p_load[t_0] * self.simulator.devices[dev_id].qp_ratio
        for idx, (dev_id, p_max) in enumerate(zip((2, 4), self.P_maxs)):
            state[2 * n_dev + n_des + idx] = p_max[t_0]
            state[dev_id] = p_max[t_0]
            state[n_dev + dev_id] = self.np_random.uniform(
                self.simulator.devices[dev_id].q_min,
                self.simulator.devices[dev_id].q_max,
            )
        storage = self.simulator.devices[6]
        state[2 * n_dev] = self.np_random.uniform(
            storage.soc_min * self.simulator.baseMVA,
            storage.soc_max * self.simulator.baseMVA,
        )
        return state

    def next_vars(self, s_t):
        index = int((s_t[-1] + 1) % self.profile_length)
        return np.asarray(
            [*(profile[index] for profile in self.P_loads),
             *(profile[index] for profile in self.P_maxs), index],
            dtype=np.float64,
        )

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        new_date = self.date + self.state[-1] * self.timestep_length
        super().reset_date(new_date)
        info.update(
            {"profile_split": self.profile_split, "profile_path": str(self.profile_path)}
        )
        return observation, info
