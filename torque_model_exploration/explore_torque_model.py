import matplotlib

matplotlib.use('Qt5Agg')
import math
import numpy as np
import matplotlib.pyplot as plt
import os

from selfdrive.config import Conversions as CV
from cereal import car
from selfdrive.controls.lib.latcontrol_model import LatControlModel
from scipy.optimize import curve_fit

dir_name = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_name)


def fit_func(x, _c2):
  # def fit_func(x, _c2, _c6, _c7, _c8):
  desired_angle, v_ego = x
  # exp = np.interp(v_ego, [12, 22], [1, 1.4])
  # coef = _c6 * v_ego ** 2 + _c7 * v_ego + _c8
  # sqrt = np.where(desired_angle > 0, 1, -1) * (np.sqrt(abs(desired_angle * coef)) ** 1.5)
  # return sqrt * _c2# * (_c3 * v_ego ** 2 + _c4 * v_ego + _c5)

  # # f(angle, speed) = steer
  desired_angle *= _c2
  sigmoid = desired_angle / (1 + abs(desired_angle))
  return 0.05 * sigmoid  # * (v_ego ** 2 * _c3)


# def fit_func(x, _c1, _c2, _c5):
#   desired_angle, v_ego = x
#   v1 = _c1 * v_ego + _c2
#   y = 1 / (1 + np.exp(desired_angle * v1)) + _c5
#   return y

def fit_func(x, _c1, _c2, _c3, _c4, _c5, _c6, _c7):
  desired_angle, v_ego = x
  sign = np.where(desired_angle > 0, -1, 1)
  desired_angle = abs(desired_angle)
  pt1 = _c7 * desired_angle ** 4 + _c5 * desired_angle ** 3 + _c1 * desired_angle ** 2 + _c2 * desired_angle
  pt1 *= _c3 * v_ego ** 2 + _c6 * v_ego + _c4
  return pt1 * sign


def ff_function(desired_angle, v_ego):
  desired_angle *= 0.02904609
  sigmoid = desired_angle / (1 + math.fabs(desired_angle))
  return 0.10006696 * sigmoid * (v_ego + 3.12485927)


def model_ff(x, _c1, _c2, _c3):
  yld = []
  desired_angles, v_egos = x
  lower_band = 0.1 * 0.4
  for desired_angle, v_ego in zip(desired_angles, v_egos):
    upper_band = 0.1 * (_c1 * v_ego ** 2 + _c3 * v_ego + _c2)
    sign = -1 if desired_angle < 0 else 1
    ff_band_split_angle = max(0.4712479121927941 * v_ego - 1.053, 0.)
    ff_lower = desired_angle * lower_band
    if abs(desired_angle) < ff_band_split_angle:
      yld.append(ff_lower)
    else:
      ff_offset = (lower_band - upper_band) * ff_band_split_angle
      ff_upper = desired_angle * upper_band
      yld.append(ff_upper + sign * ff_offset)
  return yld


def mock_model(x, _c3, _c4, _c5):
  actual_angle, desired_angle, v_ego = x
  actual_angle, desired_angle, v_ego = np.array(actual_angle), np.array(desired_angle), np.array(v_ego)
  ff = np.array([get_steer_feedforward_toyota(aa, ve) for aa, ve in zip(actual_angle, v_ego)])
  lower_band = 0.05
  # band_split_error = np.interp(abs(actual_angle), [15, 30, 60, 90, 120], [(3.2+2.7)/2, (6.1+3)/2, (12+3.41)/2, 17, 22])
  # band_split_error = np.interp(abs(actual_angle), [15, 30, 60, 90, 120], [2.43, ])

  # if abs(desired_angle - actual_angle) < band_split_angle
  # p = lower_band  # np.interp(abs(desired_angle - actual_angle), [0, 3], [0.05, 0.2])
  error = (desired_angle - actual_angle)
  p = error * np.where(abs(error) < 5, 0.08, 0.1364)
  # p *= _c3 * v_ego ** 2 + _c4 * v_ego + _c5
  # p = error  # np.interp(abs(desired_angle - actual_angle), [0, 3], [0.05, 0.2])
  # p *= _c1 * np.abs(error) ** 2 + _c3 * np.abs(error) + _c2
  # p = (desired_angle - actual_angle) * np.clip(1/abs(ff)*.01, 0.05, 0.2)
  return ff + p


def get_steer_feedforward_toyota(desired_angle, v_ego):
  lower_band = 0.1 * 0.4
  upper_band = 0.1 * (-0.00013868 * v_ego ** 2 + 0.00905046 * v_ego + 0.0362945)
  sign = -1 if desired_angle < 0 else 1
  ff_band_split_angle = max(0.4712479121927941 * v_ego - 1.053, 0.)
  if abs(desired_angle) < ff_band_split_angle:
    return desired_angle * lower_band
  else:
    ff_offset = (lower_band - upper_band) * ff_band_split_angle
    ff_upper = desired_angle * upper_band
    return ff_upper + sign * ff_offset


def predict(angle_steers, angle_steers_des, v_ego, mirror=False):
  if mirror:
    sign = -1 if angle_steers < 0 else 1
    offset = model.predict([0, 0, 0, 0, v_ego])[0]
    model_input = [abs(angle_steers_des), abs(angle_steers), 0., 0., v_ego]

    return (model.predict(model_input)[0] - offset) * sign
  else:
    model_input = [angle_steers_des, angle_steers, 0., 0., v_ego]

    return model.predict(model_input)[0]


CP = car.CarParams()
tune = CP.lateralTuning
tune.init('model')
tune.model.useRates = False  # TODO: makes model sluggish, see comments in latcontrol_model.py
tune.model.multiplier = 1.
tune.model.name = "camryh_tss2"

model = LatControlModel(CP)

FF_ANGLE = 10.


def plot_ff_by_angle(v_ego=5):
  angles = np.linspace(-45, 45, 1000)
  preds = [predict(angle, angle, v_ego, mirror=True) * 1500. for angle in angles]
  # ff_new = [ff_function(angle, v_ego) * 1500. for angle in angles]
  # ff_new = [fit_func([angle, v_ego], 0.02904609, 0.10006696, 3.12) * 1500. for angle in angles]
  ff_new = [model_ff([[angle], [v_ego]], *params)[0] * 1500. for angle in angles]
  # ff_exp = [model_ff(([angle], [v_ego]))[0] * 1500. for angle in angles]

  # plt.figure()
  plt.clf()
  mph = round(CV.MS_TO_MPH * v_ego)
  plt.title('model at {} mph'.format(mph))
  plt.plot(angles, preds, label='prediction')
  # plt.plot(angles, ff_new, label='qadmus\'s ff')
  plt.plot([-45, 45], [0, 0])
  plt.plot(angles, ff_new, label='experimental ff')
  plt.xlabel('angle')
  plt.ylabel('torque')
  plt.ylim(-2000, 2000)
  plt.legend()
  plt.savefig(f'plots/ff_by_angle_{mph}_mph.png')
  # plt.show()


def plot_response(actual_angle=0, v_ego=5):
  angles = np.linspace(actual_angle - 45, actual_angle + 45, 1000)
  preds = [predict(actual_angle, desired_angle, v_ego) * 1500. for desired_angle in angles]

  ff_exp = [mock_model([[actual_angle], [desired_angle], [v_ego]], *params) * 1500. for desired_angle in angles]

  # plt.figure()
  plt.clf()
  mph = round(CV.MS_TO_MPH * v_ego)
  plt.title('model at {} mph at {} deg'.format(mph, round(actual_angle)))
  plt.plot(angles, preds, label='prediction')
  # plt.plot(angles, ff_new, label='qadmus\'s ff')
  plt.plot([actual_angle - 45, actual_angle + 45], [actual_angle, actual_angle])
  plt.plot([actual_angle, actual_angle], [-3000, 3000])
  plt.plot(angles, ff_exp, label='experimental ff')
  plt.xlabel('right desired <- desired angle -> left desired')
  plt.ylabel('torque')
  # plt.ylim(-6000, 6000)
  plt.legend()
  # plt.savefig(f'plots/ff_by_angle_{mph}_mph.png')
  plt.show()


N_SAMPLES = 10000
MIN_SPEED = 5
MAX_SPEED = 80

data_speeds = np.random.uniform(MIN_SPEED * CV.MPH_TO_MS, MAX_SPEED * CV.MPH_TO_MS, N_SAMPLES)
# data_angles = np.random.uniform(-45, 45, N_SAMPLES)
data_angles = np.random.normal(0, 15, N_SAMPLES)
data_des_angles = np.array([np.random.normal(i, 15) for i in data_angles])
data_torque = np.array([predict(angle, angle_des, speed, mirror=True) for speed, angle_des, angle in zip(data_speeds, data_des_angles, data_angles)])

params, covs = curve_fit(
  mock_model, np.array([data_angles, data_des_angles, data_speeds]), np.array(data_torque),
  # bounds=((0, 0.5), (2, 10)),
  # p0=[0.02904609, 0.10006696, 3.12],
  # maxfev=8000
)

print('Got params!')
print(params)

# for speed in np.linspace(MIN_SPEED, MAX_SPEED, round((MAX_SPEED - MIN_SPEED) / 5) + 1):
#   plot_ff_by_angle(speed * CV.MPH_TO_MS)