import numpy as np
from selfdrive.config import Conversions as CV

TORQUE_SCALE = 1500


def feedforward(angle, speed):
  steer_feedforward = angle  # offset does not contribute to resistive torque
  _c1, _c2, _c3 = 0.35189607550172824, 7.506201251644202, 69.226826411091
  steer_feedforward *= _c1 * speed ** 2 + _c2 * speed + _c3
  # steer_feedforward *= speed ** 2
  return steer_feedforward


class LatControlPF:
  def __init__(self):
    self.k_f = 0.00006908923778520113
    # self.k_f = 0.00003
    self.speed = 0

  @property
  def k_p(self):
    return np.interp(self.speed, [20 * CV.MPH_TO_MS, 70 * CV.MPH_TO_MS], [.06, .15])

  def update(self, setpoint, measurement, speed):
    self.speed = speed
    f = feedforward(setpoint, speed)

    error = setpoint - measurement

    p = error * self.k_p
    f = f * self.k_f

    return p + f  # multiply by 1500 to get torque units
    # return np.clip(p + steer_feedforward, -1, 1)  # multiply by 1500 to get torque units


def random_chance(percent: int):
  return np.random.randint(0, 100) < percent or percent == 100
