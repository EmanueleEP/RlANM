"""Fix per il double reward shaping e le scale inconsistenti in constraint_costs.
"""

import numpy as np


V_TOL   = 0.01   # pu  — same as LagrangianWrapper(v_tolerance=0.01)
P_TOL   = 0.01   # pu  — same as LagrangianWrapper(pq_limit=0.01)
Q_TOL   = 0.01   # pu  — same as LagrangianWrapper(pq_limit=0.01)

def build_constraint_costs(
    v_violation_pu_max: float,
    combined_p_constraint_pu: float,
    q_violation_pu_max: float,
    v_tol: float = V_TOL,
    p_tol: float = P_TOL,
    q_tol: float = Q_TOL,
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
        operational thresholds (must correspond to the cost_limit of the LagrangianWrapper).

    Returns
    -------
    np.ndarray of shape (3,), dtype float32
        [v_cost_normalized, p_cost_normalized, q_cost_normalized]
    """
    # Evita divisione per zero in caso di soglie malconfigurate
    v_tol = max(v_tol, 1e-8)
    p_tol = max(p_tol, 1e-8)
    q_tol = max(q_tol, 1e-8)

    return np.array(
        [
            v_violation_pu_max   / v_tol,
            combined_p_constraint_pu / p_tol,
            q_violation_pu_max   / q_tol,
        ],
        dtype=np.float32,
    )