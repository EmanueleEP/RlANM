"""A more complex BCP network with additional buses/loads but same controllable devices."""

import numpy as np

network = {"baseMVA": 100.0}

# 6-bus radial feeder (0 is slack, others are PQ).
network["bus"] = np.array(
    [
        [0, 0, 132, 1.0, 1.0],
        [1, 1, 33, 1.1, 0.9],
        [2, 1, 33, 1.1, 0.9],
        [3, 1, 33, 1.1, 0.9],
        [4, 1, 33, 1.1, 0.9],
        [5, 1, 33, 1.1, 0.9],
    ]
)

# Device rows follow the convention used in this repo:
# [DEV_ID, BUS_ID, DEV_TYPE, Q/P, QMAX, QMIN, DEV_STATUS, PMAX, PMIN, PC1, PC2, QC1MAX, QC1MIN, SOC_MAX, EFF]
_device_rows = np.array(
    [
        [0, 0, 0, None, 20, -20, 20, -20, None, None, None, None, None, None, None],  # slack
        [1, 1, -1, 0.2, 0, -20, None, None, None, None, None, None, None, None, None],  # load 1
        [2, 3, 2, None, 30, 0, 30, -30, 20, None, 15, -15, None, None, None],  # solar
        [3, 5, 3, None, 50, -50, 50, -50, 30, -30, 25, -25, 100, 0.01, 0.9],  # battery
        [4, 4, -1, 0.2, 0, -20, None, None, None, None, None, None, None, None, None],  # load 2 (far)
    ],
    dtype=object,
)

network["device"] = np.empty((len(_device_rows), len(_device_rows[0])), dtype=object)
for i, row in enumerate(_device_rows):
    for j, val in enumerate(row):
        network["device"][i, j] = val

# Branches: simple radial chain with tighter limits downstream.
network["branch"] = np.array(
    [
        [0, 1, 0.0036, 0.1834, 0.0, 32, 1, 1],
        [1, 2, 0.0100, 0.0400, 0.0, 16, 1, 1],
        [2, 3, 0.0120, 0.0450, 0.0, 12, 1, 1],
        [3, 4, 0.0150, 0.0550, 0.0, 10, 1, 1],
        [4, 5, 0.0180, 0.0650, 0.0, 8, 1, 1],
    ]
)

