import os
import numpy as np
import math

from selfdrive.modeld.constants import index_function
import matplotlib.pyplot as plt
from selfdrive.config import Conversions as CV

N = 12
MAX_T = 10.0
T_IDXS_LST = [index_function(idx, max_val=MAX_T, max_idx=N + 1) for idx in range(N + 1)]

T_IDXS = np.array(T_IDXS_LST)
T_DIFFS = np.diff(T_IDXS, prepend=[0.])
MIN_ACCEL = -3.5
T_REACT = 1.8
MAX_BRAKE = 9.81


def get_stopped_equivalence_factor_orig(v_lead):
  return T_REACT * v_lead + (v_lead * v_lead) / (2 * MAX_BRAKE)


def get_safe_obstacle_distance_orig(v_ego):
  return 2 * T_REACT * v_ego + (v_ego * v_ego) / (2 * MAX_BRAKE) + 4.0


def get_stopped_equivalence_factor(v_lead):
  offset = np.minimum(v_lead*v_lead/8, 4)
  print(offset)
  return T_REACT * v_lead + (v_lead * v_lead) / (2 * MAX_BRAKE) + offset


def get_safe_obstacle_distance(v_ego):
  return 2.0 * T_REACT * v_ego + (v_ego * v_ego) / (2 * MAX_BRAKE) + 4.0
  # return T_REACT * v_ego + (v_ego*v_ego*v_ego) / (20 * MAX_BRAKE) + 4.0


def desired_follow_distance_orig(v_ego, v_lead):
  return get_safe_obstacle_distance_orig(v_ego) - get_stopped_equivalence_factor_orig(v_lead)


def desired_follow_distance(v_ego, v_lead):
  return get_safe_obstacle_distance(v_ego) - get_stopped_equivalence_factor(v_lead)


fig, axes = plt.subplots(2)
axes[0].invert_xaxis()
axes[1].invert_xaxis()

X = np.linspace(70 * CV.MPH_TO_MS, 1 * CV.MPH_TO_MS, 100)
Y = desired_follow_distance(X, X)
Y_orig = desired_follow_distance_orig(X, X)
true_TR_orig = desired_follow_distance_orig(X, X) / np.maximum(X, 1e-3)
true_TR = desired_follow_distance(X, X) / np.maximum(X, 1e-3)

axes[0].plot(X * CV.MS_TO_MPH, Y_orig, label='Y_orig')
axes[0].plot(X * CV.MS_TO_MPH, Y, label='Y')
axes[0].legend()
axes[0].set_xlabel('speed (mph)')

# axes[1].plot(X * CV.MS_TO_MPH, true_TR_orig, label='true_TR_orig')
# axes[1].plot(X * CV.MS_TO_MPH, true_TR, label='true_TR')
axes[1].plot(X * CV.MS_TO_MPH, (4) / (X + 10.), label='true_TR')
axes[1].plot(X * CV.MS_TO_MPH, (4) / (X + 12), label='true_TR')
axes[1].legend()
axes[1].set_xlabel('speed (mph)')
plt.show()
