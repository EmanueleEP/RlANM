"""Constraint costs and reward parameters shared by the custom environments.

All numeric values live in ``rl_agents/hyperparameters.py``. The environments
read them from there as defaults, and the training/evaluation scripts can
override them with the values saved in a run's ``hyperparameters.txt``
(see ``rl_agents.utils.configure_env``).
"""

import numpy as np


# Hyperparameters consumed by the environments (names as in hyperparameters.py).
ENV_PARAM_KEYS = (
    # Normalization of constraint_costs (pu). LagrangianWrapper.cost_limit
    # (V_TOLERANCE, PQ_LIMIT) is expressed in multiples of these. With the
    # Lagrangian/BCP reward the Q cost is reactive_usage_fraction() instead.
    "V_TOL",
    "P_TOL",
    "Q_TOL",
    # Reward weights.
    "W_CURTAIL",
    "W_SLACK_IMPORT",
    "W_SLACK_EXPORT",
    "W_SOC_BOUNDARY",
    "W_Q_MAGNITUDE",
    "W_BRANCH_SOFT",
    # Reward shaping thresholds.
    "BRANCH_SOFT_LOADING",
    "SOC_LOW",
    "SOC_HIGH",
    "BCP_SOC_LOW",
    "BCP_SOC_HIGH",
)


def default_env_params():
    """Return the environment parameters currently set in hyperparameters.py."""
    from rl_agents import hyperparameters

    return {key: float(getattr(hyperparameters, key)) for key in ENV_PARAM_KEYS}


def validate_env_params(params):
    """Reject unknown keys, so a typo cannot silently leave a default in place."""
    unknown = set(params) - set(ENV_PARAM_KEYS)
    if unknown:
        raise KeyError(f"Unknown environment parameters: {sorted(unknown)}")
    return {key: float(value) for key, value in params.items()}


def reactive_usage_fraction(devices, device_ids, dev_q_mvar, base_mva):
    """Total |Q| of `device_ids` as a fraction of their reactive capability.

    The capability of each device is max(|q_max|, |q_min|) from the network
    data, so the result is in [0, 1] and adapts to any network topology.
    """
    capability_mvar = sum(
        max(abs(float(np.nan_to_num(devices[i].q_max))), abs(float(np.nan_to_num(devices[i].q_min))))
        for i in device_ids
    ) * float(base_mva)
    used_mvar = sum(abs(float(dev_q_mvar.get(i, 0.0))) for i in device_ids)
    return used_mvar / capability_mvar if capability_mvar > 0 else 0.0


def build_constraint_costs(
    v_violation_pu_max: float,
    combined_p_constraint_pu: float,
    q_violation_pu_max: float,
    v_tol: float,
    p_tol: float,
    q_tol: float,
) -> np.ndarray:
    """
    Parameters
    ----------
    v_violation_pu_max : float
        max violation of voltage in pu.
    combined_p_constraint_pu : float
        max violation of active power / branch overload in pu.
    q_violation_pu_max : float
        max violation of reactive power in pu.
    v_tol, p_tol, q_tol : float
        normalization scales (V_TOL, P_TOL, Q_TOL in hyperparameters.py).

    Returns
    -------
    np.ndarray of shape (3,), dtype float32
        [v_cost_normalized, p_cost_normalized, q_cost_normalized]
    """
    # Avoid division by zero with misconfigured scales.
    v_tol = max(v_tol, 1e-8)
    p_tol = max(p_tol, 1e-8)
    q_tol = max(q_tol, 1e-8)

    return np.array(
        [
            v_violation_pu_max / v_tol,
            combined_p_constraint_pu / p_tol,
            q_violation_pu_max / q_tol,
        ],
        dtype=np.float32,
    )
