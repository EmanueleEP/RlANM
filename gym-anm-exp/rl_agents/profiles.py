"""Environment-independent load and generation profile sources for Gym-ANM."""

from dataclasses import dataclass
import csv
import importlib
from pathlib import Path
from types import MethodType
from typing import Mapping

import numpy as np
from gym_anm.simulator.components import Generator, Load


@dataclass(frozen=True)
class ProfileBank:
    """Profile scenarios shaped as (scenario, timestep, device)."""

    load_p_mw: np.ndarray
    gen_p_max_mw: np.ndarray
    load_device_ids: tuple
    gen_device_ids: tuple
    source: str = "provider"


def _resolve_file(path, split):
    value = str(Path(path).expanduser())
    if "{split}" in value:
        value = value.format(split=split)
    resolved = Path(value)
    if resolved.is_dir():
        candidates = [
            resolved / f"profiles_{split}.npz",
            resolved / f"{split}.npz",
            resolved / f"profiles_{split}.csv",
            resolved / f"{split}.csv",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(
            f"No profile file for split {split!r} in {resolved}; expected "
            f"profiles_{split}.npz/.csv or {split}.npz/.csv."
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"Profile file not found: {resolved}")
    return resolved.resolve()


def _as_scenarios(values, n_devices, name):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2:
        if array.shape[1] == n_devices:
            array = array[None, :, :]
        elif array.shape[0] == n_devices:
            array = array.T[None, :, :]
    elif array.ndim == 3 and array.shape[2] != n_devices and array.shape[1] == n_devices:
        array = array.transpose(0, 2, 1)
    if array.ndim != 3 or array.shape[2] != n_devices or array.shape[1] == 0:
        raise ValueError(
            f"{name} must have shape (steps, {n_devices}) or "
            f"(scenarios, steps, {n_devices}); got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _from_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        required = {"load_p_mw", "gen_p_max_mw"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing NPZ fields: {sorted(missing)}")
        return {
            "load_p_mw": archive["load_p_mw"],
            "gen_p_max_mw": archive["gen_p_max_mw"],
            "load_device_ids": archive["load_device_ids"] if "load_device_ids" in archive else None,
            "gen_device_ids": archive["gen_device_ids"] if "gen_device_ids" in archive else None,
        }


def _from_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        load_columns = [field for field in fields if field.startswith("load:")]
        gen_columns = [field for field in fields if field.startswith("generation:")]
        if not load_columns or not gen_columns:
            raise ValueError(
                "CSV profiles require load:<device_id> and generation:<device_id> columns."
            )
        load_ids = [int(field.split(":", 1)[1]) for field in load_columns]
        gen_ids = [int(field.split(":", 1)[1]) for field in gen_columns]
        scenarios = {}
        for row_number, row in enumerate(reader, start=2):
            scenario = row.get("scenario") or "0"
            try:
                loads = [float(row[field]) for field in load_columns]
                generation = [float(row[field]) for field in gen_columns]
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid numeric profile value at {path}:{row_number}") from error
            scenarios.setdefault(scenario, ([], []))[0].append(loads)
            scenarios[scenario][1].append(generation)
    if not scenarios:
        raise ValueError(f"Profile CSV is empty: {path}")
    lengths = {len(values[0]) for values in scenarios.values()}
    if len(lengths) != 1:
        raise ValueError("All CSV profile scenarios must contain the same number of rows.")
    return {
        "load_p_mw": np.asarray([values[0] for values in scenarios.values()]),
        "gen_p_max_mw": np.asarray([values[1] for values in scenarios.values()]),
        "load_device_ids": load_ids,
        "gen_device_ids": gen_ids,
    }


def _load_provider(spec, env, split, seed):
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Profile provider must use the form package.module:callable.")
    provider = getattr(importlib.import_module(module_name), attribute)
    if not callable(provider):
        raise TypeError(f"Profile provider {spec!r} is not callable.")
    result = provider(env=env, split=split, seed=seed)
    if isinstance(result, ProfileBank):
        return result
    if not isinstance(result, Mapping):
        raise TypeError("Profile provider must return a mapping or ProfileBank.")
    return dict(result)


def load_profile_bank(env, *, profile_file=None, profile_provider=None,
                      split="train", seed=None):
    """Load, validate, and reorder profiles to match an environment topology."""
    if bool(profile_file) == bool(profile_provider):
        raise ValueError("Specify exactly one of profile_file or profile_provider.")
    base = env.unwrapped
    if not hasattr(base, "simulator"):
        raise TypeError("External profiles require a Gym-ANM environment with a simulator.")

    load_ids = tuple(
        dev_id for dev_id, device in base.simulator.devices.items()
        if isinstance(device, Load)
    )
    gen_ids = tuple(
        dev_id for dev_id, device in base.simulator.devices.items()
        if isinstance(device, Generator) and not device.is_slack
    )
    if profile_file:
        path = _resolve_file(profile_file, split)
        if path.suffix.lower() == ".npz":
            raw = _from_npz(path)
        elif path.suffix.lower() == ".csv":
            raw = _from_csv(path)
        else:
            raise ValueError(f"Unsupported profile format {path.suffix!r}; use .npz or .csv.")
        source = str(path)
    else:
        raw = _load_provider(profile_provider, base, split, seed)
        if isinstance(raw, ProfileBank):
            source = raw.source or profile_provider
            raw = {
                "load_p_mw": raw.load_p_mw,
                "gen_p_max_mw": raw.gen_p_max_mw,
                "load_device_ids": raw.load_device_ids,
                "gen_device_ids": raw.gen_device_ids,
            }
        else:
            source = profile_provider

    raw_load_ids = raw.get("load_device_ids")
    raw_gen_ids = raw.get("gen_device_ids")
    supplied_load_ids = tuple(int(x) for x in (load_ids if raw_load_ids is None else raw_load_ids))
    supplied_gen_ids = tuple(int(x) for x in (gen_ids if raw_gen_ids is None else raw_gen_ids))
    if len(supplied_load_ids) != len(load_ids) or set(supplied_load_ids) != set(load_ids):
        raise ValueError(f"Load profile IDs {supplied_load_ids} do not match environment IDs {load_ids}.")
    if len(supplied_gen_ids) != len(gen_ids) or set(supplied_gen_ids) != set(gen_ids):
        raise ValueError(f"Generation profile IDs {supplied_gen_ids} do not match environment IDs {gen_ids}.")

    loads = _as_scenarios(raw["load_p_mw"], len(supplied_load_ids), "load_p_mw")
    generation = _as_scenarios(raw["gen_p_max_mw"], len(supplied_gen_ids), "gen_p_max_mw")
    if loads.shape[:2] != generation.shape[:2]:
        raise ValueError("Load and generation profiles must have matching scenarios and timesteps.")
    loads = loads[:, :, [supplied_load_ids.index(dev_id) for dev_id in load_ids]]
    generation = generation[:, :, [supplied_gen_ids.index(dev_id) for dev_id in gen_ids]]
    if np.any(loads > 1e-9):
        raise ValueError("Gym-ANM load injections must be non-positive MW values.")
    if np.any(generation < -1e-9):
        raise ValueError("Maximum generation profiles must be non-negative MW values.")
    return ProfileBank(loads, generation, load_ids, gen_ids, source)


def apply_external_profiles(env, *, profile_file=None, profile_provider=None,
                            split="train", seed=None):
    """Install a profile bank on any compatible Gym-ANM environment instance."""
    if profile_file is None and profile_provider is None:
        return env
    bank = load_profile_bank(
        env, profile_file=profile_file, profile_provider=profile_provider,
        split=split, seed=seed,
    )
    base = env.unwrapped
    original_init_state = base.init_state
    original_next_vars = base.next_vars
    rng = np.random.default_rng(seed) if seed is not None else None

    def choose_index(self):
        random = rng if rng is not None else self.np_random
        scenario = int(random.integers(0, bank.load_p_mw.shape[0]))
        timestep = int(random.integers(0, bank.load_p_mw.shape[1]))
        self._external_profile_position = [scenario, timestep]
        return scenario, timestep

    def init_state(self):
        state = np.asarray(original_init_state(), dtype=np.float64).copy()
        scenario, timestep = choose_index(self)
        n_devices = self.simulator.N_device
        n_storage = self.simulator.N_des
        potential_offset = 2 * n_devices + n_storage
        for dev_id, value in zip(bank.load_device_ids, bank.load_p_mw[scenario, timestep]):
            if dev_id >= n_devices:
                raise ValueError("External profiles require contiguous Gym-ANM device IDs.")
            device = self.simulator.devices[dev_id]
            state[dev_id] = value
            state[n_devices + dev_id] = value * device.qp_ratio
        for index, (dev_id, value) in enumerate(zip(bank.gen_device_ids, bank.gen_p_max_mw[scenario, timestep])):
            state[dev_id] = value
            state[potential_offset + index] = value
        return state

    def next_vars(self, state):
        original = np.asarray(original_next_vars(state), dtype=np.float64)
        scenario, timestep = self._external_profile_position
        timestep = (timestep + 1) % bank.load_p_mw.shape[1]
        self._external_profile_position[1] = timestep
        aux = original[self.simulator.N_load + self.simulator.N_non_slack_gen:]
        return np.concatenate((bank.load_p_mw[scenario, timestep],
                               bank.gen_p_max_mw[scenario, timestep], aux))

    base.init_state = MethodType(init_state, base)
    base.next_vars = MethodType(next_vars, base)
    base.external_profile_bank = bank
    base.external_profile_source = bank.source
    return env
