#!/usr/bin/env python3
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from tqdm import tqdm   # type: ignore
from scipy.optimize import curve_fit

from opendbc.can.parser import CANParser
from selfdrive.car.toyota.values import STEER_THRESHOLD

from common.realtime import DT_CTRL
from selfdrive.config import Conversions as CV
from selfdrive.car.toyota.values import SteerLimitParams as TOYOTA_PARAMS
from selfdrive.car.subaru.carcontroller import CarControllerParams as SUBARU_PARAMS
from tools.lib.route import Route
import seaborn as sns
from tools.lib.logreader import MultiLogIterator
import pickle
import binascii

old_kf = 0.0000795769068  # for plotting old function compared to new polynomial function
MIN_SAMPLES = 15 / DT_CTRL


def to_signed(n, bits):
  if n >= (1 << max((bits - 1), 0)):
    n = n - (1 << max(bits, 0))
  return n


def hex_to_binary(hexdata):
  return (bin(int(binascii.hexlify(hexdata), 16))[2:]).zfill(len(hexdata) * 8)  # adds leading/trailing zeros so data matches up with 8x8 array on cabana


def old_feedforward(v_ego, angle_steers, angle_offset=0):
  steer_feedforward = (angle_steers - angle_offset)
  steer_feedforward *= v_ego ** 2
  return steer_feedforward



def fit_all(x_input, _c1, _c2, _c3, _c4):
  """
    x_input is array of v_ego and a_ego
    all _params are to be fit by curve_fit
    kf is multiplier from angle to torque
    c1-c3 are poly coefficients
  """
  v_ego, a_ego = x_input.copy()
  # return (_c1 * v_ego + _c3) + (_c2 * a_ego)
  return (a_ego * _c1 + (_c4 * (v_ego * _c2 + 1))) * (v_ego * _c3 + 1)
  # return _c4 * a_ego + np.polyval([_c1, _c2, _c3], v_ego)  # use this if we think there is a non-linear speed relationship


class CustomFeedforward:
  def __init__(self, to_fit):
    """
      fit_all: if True, then it fits kf as well as speed poly
               if False, then it only fits kf using speed poly found from prior fitting and data
    """
    if to_fit == 'kf':
      self.c1, self.c2, self.c3 = 0.35189607550172824, 7.506201251644202, 69.226826411091
      self.fit_func = self._fit_kf
    elif to_fit == 'all':
      self.fit_func = self._fit_all
    elif to_fit == 'poly':
      self.kf = 0.00006908923778520113
      self.fit_func = self._fit_poly

  def get(self, v_ego, angle_steers, *args):  # helper function to easily use fitting ff function):
    x_input = np.array((v_ego, angle_steers)).T
    return self.fit_func(x_input, *args)

  @staticmethod
  def _fit_all(x_input, _kf, _c1, _c2, _c3):
    """
      x_input is array of v_ego and angle_steers
      all _params are to be fit by curve_fit
      kf is multiplier from angle to torque
      c1-c3 are poly coefficients
    """
    v_ego, angle_steers = x_input.copy()
    steer_feedforward = angle_steers * np.polyval([_c1, _c2, _c3], v_ego)
    return steer_feedforward * _kf

  def _fit_kf(self, x_input, _kf):
    """
      Just fits kf using best-so-far speed poly
    """
    v_ego, angle_steers = x_input.copy()
    steer_feedforward = angle_steers * np.polyval([self.c1, self.c2, self.c3], v_ego)
    return steer_feedforward * _kf

  def _fit_poly(self, x_input, _c1, _c2, _c3):
    """
      Just fits poly using current kf
    """
    v_ego, angle_steers = x_input.copy()
    steer_feedforward = angle_steers * np.polyval([_c1, _c2, _c3], v_ego)
    return steer_feedforward * self.kf


CF = CustomFeedforward(to_fit='poly')



def fit_ff_model(use_dir, plot=False):
  files = os.listdir(use_dir)
  files = [f for f in files if '.ini' not in f]

  data = [[]]

  engaged, gas_enable = False, False
  # torque_cmd, angle_steers, angle_steers_des, angle_offset, v_ego = None, None, None, None, None
  v_ego, gas_command, a_ego, user_gas, car_gas = None, None, None, None, None
  last_time = 0
  lr = MultiLogIterator([os.path.join(use_dir, i) for i in files], wraparound=False)

  signals = [
    ("GAS_COMMAND", "GAS_COMMAND", 0),
    ("GAS_COMMAND2", "GAS_COMMAND", 0),
    ("ENABLE", "GAS_COMMAND", 0),
    ("INTERCEPTOR_GAS", "GAS_SENSOR", 0),
    ("INTERCEPTOR_GAS2", "GAS_SENSOR", 0),
    ("GAS_PEDAL", "GAS_PEDAL", 0),
  ]
  cp = CANParser("toyota_corolla_2017_pt_generated", signals)

  all_msgs = sorted(lr, key=lambda msg: msg.logMonoTime)

  for msg in tqdm(all_msgs):
    if msg.which() == 'carState':
      v_ego = msg.carState.vEgo
      a_ego = msg.carState.aEgo
      engaged = msg.carState.cruiseState.enabled

    if msg.which() not in ['can', 'sendcan']:
      continue

    updated = cp.update_string(msg.as_builder().to_bytes())
    for u in updated:
      if u == 0x200:
        gas_enable = bool(cp.vl[u]['ENABLE'])
        gas_command = max(round(cp.vl[u]['GAS_COMMAND'] / 255., 4), 0.0)  # unscale, round, and clip
        assert gas_command <= 1, "Gas command above 100%, look into this"
      elif u == 0x201:
        user_gas = ((cp.vl[u]['INTERCEPTOR_GAS'] + cp.vl[u]['INTERCEPTOR_GAS2']) / 2.) / 255.
      elif u == 0x2c1:
        car_gas = cp.vl[u]['GAS_PEDAL']


    # for m in msg.can:
    #   if m.address == 0x2e4 and m.src == 128:  # STEERING_LKA
    #     engaged = bool(m.dat[0] & 1)
    #     torque_cmd = to_signed((m.dat[1] << 8) | m.dat[2], 16)
    #   elif m.address == 0x260 and m.src == 0:  # STEER_TORQUE_SENSOR
    #     steering_pressed = abs(to_signed((m.dat[1] << 8) | m.dat[2], 16)) > STEER_THRESHOLD
    #   elif m.address == 0x25 and m.src == 0:  # STEER_ANGLE_SENSOR
    #     steer_angle = to_signed(int(bin(m.dat[0])[2:].zfill(8)[4:] + bin(m.dat[1])[2:].zfill(8), 2), 12) * 1.5
    #     steer_fraction = to_signed(int(bin(m.dat[4])[2:].zfill(8)[:4], 2), 4) * 0.1
    #     angle_steers = steer_angle + steer_fraction

    if (not engaged and not gas_enable and None not in [v_ego, gas_command, a_ego, user_gas, car_gas] and  # creates uninterupted sections of engaged data
            abs(msg.logMonoTime - last_time) * 1e-9 < 1 / 20):  # also split if there's a break in time
      data[-1].append({'v_ego': v_ego, 'gas_command': gas_command, 'a_ego': a_ego, 'user_gas': user_gas, 'car_gas': car_gas, 'time': msg.logMonoTime * 1e-9})

    elif len(data[-1]):  # if last list has items in it, append new empty section
      data.append([])

    last_time = msg.logMonoTime

  del all_msgs

  print('Max seq. len: {}'.format(max([len(line) for line in data])))

  data = [sec for sec in data if len(sec) > 1 / DT_CTRL]  # long enough sections

  # TODO: do something like this with a_ego, move it up since acceleration lags behind gas command slightly
  # for i in range(len(data)):  # accounts for steer actuator delay (moves torque cmd up by 12 samples)
  #   torque = [line['torque'] for line in data[i]]
  #   for j in range(len(data[i])):
  #     if j < steer_delay:
  #       continue
  #     data[i][j]['torque'] = torque[j - steer_delay]
  #   data[i] = data[i][steer_delay:]  # removes leading samples
  data = [i for j in data for i in j]  # flatten

  if WRITE_DATA := False:  # todo: temp, for debugging
    with open('auto_feedforward/data', 'wb') as f:
      pickle.dump(data, f)

  print(f'Samples (before filtering): {len(data)}')

  # Data filtering
  data = [line for line in data if abs(line['v_ego']) < 19 * CV.MPH_TO_MS]
  data = [line for line in data if line['a_ego'] >= -0.5]  # sometimes a ego is -0.5 while gas is still being applied (todo: maybe remove going up hills? this should be okay for now)
  # data = [line for line in data if abs(line['user_gas']) < 15 / 255]
  data = [line for line in data if line['user_gas'] > 15 / 255]
  data = [line for line in data if line['car_gas'] > 0]

  # data_a_ego = [line['a_ego'] for line in data]  # this is used to find the minimum deceleration where gas is still used
  # print(min(data_a_ego))
  # sns.distplot(data_a_ego, bins=75)
  # plt.savefig('imgs/a_ego.png')

  print(f'Samples (after filtering):  {len(data)}\n')

  assert len(data) > MIN_SAMPLES, 'too few valid samples found in route'

  # print('Max angle: {}'.format(round(max([i['angle_steers'] for i in data]), 2)))
  # print('Top speed: {} mph'.format(round(max([i['v_ego'] for i in data]) * CV.MS_TO_MPH, 2)))
  # print('Torque: min: {}, max: {}\n'.format(*[func([i['torque'] for i in data]) for func in [min, max]]))

  # Data preprocessing
  # for line in data:
  #   line['angle_steers'] = abs(line['angle_steers'] - line['angle_offset'])  # need to offset angle to properly fit ff
  #   line['torque'] = abs(line['torque'])
  #
  #   del line['time'], line['angle_offset'], line['angle_steers_des']  # delete unused

  data_speeds = np.array([line['v_ego'] for line in data])
  data_accels = np.array([line['a_ego'] for line in data])
  data_gas = np.array([line['car_gas'] for line in data])

  params, covs = curve_fit(fit_all, np.array([data_speeds, data_accels]), np.array(data_gas), maxfev=800)
  print('Params: {}'.format(params.tolist()))

  # if len(params) == 4:
  #   print('FOUND KF: {}'.format(params[0]))
  #   print('FOUND POLY: {}'.format(params[1:].tolist()))
  # elif len(params) == 3:
  #   print('FOUND POLY: {}'.format(params.tolist()))
  # elif len(params) == 1:
  #   print('FOUND KF: {}'.format(params[0]))
  # else:
  #   print('Unsupported number of params')
  #   raise Exception('Unsupported number of params: {}'.format(len(params)))
  # if len(params) > 1 and params[-1] < 0:
  #   print('WARNING: intercept is negative, possibly bad fit! needs more data')
  # print()

  def old_gas_func(speed, accel):
    return (accel * 0.5 + (0.05 * (speed / 20 + 1))) * (speed / 25 + 1)

  def new_gas_func(p):
    return fit_all(p, *params)

  from_function = np.array([new_gas_func([line['v_ego'], line['a_ego']]) for line in data])
  print('Fitted function MAE: {}'.format(np.mean(np.abs(data_gas - from_function))))


  # std_func = []
  # fitted_func = []
  # for line in data:
  #   std_func.append(abs(old_feedforward(line['v_ego'], line['angle_steers']) * old_kf * MAX_TORQUE - line['torque']))
  #   fitted_func.append(abs(CF.get(line['v_ego'], line['angle_steers'], *params) * MAX_TORQUE - line['torque']))
  #
  # print('Torque MAE: {} (standard) - {} (fitted)'.format(np.mean(std_func), np.mean(fitted_func)))
  # print('Torque STD: {} (standard) - {} (fitted)\n'.format(np.std(std_func), np.std(fitted_func)))

  if ANALYZE_SPEED := True:
    plt.clf()
    sns.distplot([line['a_ego'] for line in data], bins=100)
    plt.savefig('imgs/accel dist.png')
    plt.clf()

    accel = 0.25
    X_speed = np.linspace(0, 19 * CV.MPH_TO_MS, 20)
    y_gas_old = [old_gas_func(_x, accel) for _x in X_speed]
    y_gas_new = [new_gas_func([_x, accel]) for _x in X_speed]
    plt.plot(X_speed, y_gas_old, label='guessed gas function')
    plt.plot(X_speed, y_gas_new, label='fitted gas function')
    # print(data)

    X_data, y_data = zip(*[[line['v_ego'], line['car_gas']] for line in data if abs(line['a_ego'] - accel) < 0.1])
    print(len(X_data))
    plt.scatter(X_data, y_data, label='data', s=4)
    plt.xlabel('speed')
    plt.ylabel('gas')

    # data_0_accel

    plt.legend()
    plt.savefig('imgs/speed plot.png')

  if ANALYZE_ACCEL := True:
    plt.clf()
    sns.distplot([line['v_ego'] for line in data], bins=100)
    plt.savefig('imgs/speed dist.png')
    plt.clf()

    speed = 5.5
    X_accel = np.linspace(0, 2.25, 20)
    y_gas_old = [old_gas_func(speed, _x) for _x in X_accel]
    y_gas_new = [new_gas_func([speed, _x]) for _x in X_accel]
    plt.plot(X_accel, y_gas_old, label='guessed gas function')
    plt.plot(X_accel, y_gas_new, label='fitted gas function')
    # print(data)

    X_data, y_data = zip(*[[line['a_ego'], line['car_gas']] for line in data if abs(line['v_ego'] - speed) < 0.22352])
    print(len(X_data))
    plt.scatter(X_data, y_data, label='data', s=4)
    plt.xlabel('accel')
    plt.ylabel('gas')

    # data_0_accel

    plt.legend()
    plt.savefig('imgs/accel plot.png')


  raise Exception
  if SPEED_DATA_ANALYSIS := True:  # analyzes how torque needed changes based on speed
    if PLOT_ANGLE_DIST := False:
      sns.distplot([line['angle_steers'] for line in data if abs(line['angle_steers']) < 30], bins=200)
      raise Exception

    res = 100
    color = 'blue'

    _angles = [
      [5, 10],
      [10, 20],
      [10, 15],
      [15, 20],
      [20, 25],
      [20, 30],
      [30, 45],
    ]

    for idx, angle_range in enumerate(_angles):
      angle_range_str = '{} deg'.format('-'.join(map(str, angle_range)))
      temp_data = [line for line in data if angle_range[0] <= abs(line['angle_steers']) <= angle_range[1]]
      if not len(temp_data):
        continue
      print(f'{angle_range} samples: {len(temp_data)}')
      plt.figure(idx)
      plt.clf()
      speeds, torque = zip(*[[line['v_ego'], line['torque']] for line in temp_data])
      plt.scatter(np.array(speeds) * CV.MS_TO_MPH, torque, label=angle_range_str, color=color, s=0.05)

      _x_ff = np.linspace(0, max(speeds), res)
      _y_ff = [old_feedforward(_i, np.mean(angle_range)) * old_kf * MAX_TORQUE for _i in _x_ff]
      plt.plot(_x_ff * CV.MS_TO_MPH, _y_ff, color='orange', label='standard ff model at {} deg'.format(np.mean(angle_range)))

      _y_ff = [CF.get(_i, np.mean(angle_range), *params) * MAX_TORQUE for _i in _x_ff]
      plt.plot(_x_ff * CV.MS_TO_MPH, _y_ff, color='purple', label='new fitted ff function')

      plt.legend()
      plt.xlabel('speed (mph)')
      plt.ylabel('torque')
      plt.savefig('auto_feedforward/plots/{}.png'.format(angle_range_str))

  if ANGLE_DATA_ANALYSIS := True:  # analyzes how angle changes need of torque (RESULT: seems to be relatively linear, can be tuned by k_f)
    if PLOT_ANGLE_DIST := False:
      sns.distplot([line['angle_steers'] for line in data if abs(line['angle_steers']) < 30], bins=200)
      raise Exception

    res = 100

    _speeds = np.r_[[
      [0, 10],
      [10, 20],
      [20, 30],
      [30, 40],
      [40, 50],
      [50, 60],
      [60, 70],
    ]] * CV.MPH_TO_MS
    color = 'blue'

    for idx, speed_range in enumerate(_speeds):
      speed_range_str = '{} mph'.format('-'.join([str(round(i * CV.MS_TO_MPH, 1)) for i in speed_range]))
      temp_data = [line for line in data if speed_range[0] <= line['v_ego'] <= speed_range[1]]
      if not len(temp_data):
        continue
      print(f'{speed_range_str} samples: {len(temp_data)}')
      plt.figure(idx)
      plt.clf()
      angles, torque, speeds = zip(*[[line['angle_steers'], line['torque'], line['v_ego']] for line in temp_data])
      plt.scatter(angles, torque, label=speed_range_str, color=color, s=0.05)

      _x_ff = np.linspace(0, max(angles), res)
      _y_ff = [old_feedforward(np.mean(speed_range), _i) * old_kf * MAX_TORQUE for _i in _x_ff]
      plt.plot(_x_ff, _y_ff, color='orange', label='standard ff model at {} mph'.format(np.round(np.mean(speed_range) * CV.MS_TO_MPH, 1)))

      _y_ff = [CF.get(np.mean(speed_range), _i, *params) * MAX_TORQUE for _i in _x_ff]
      plt.plot(_x_ff, _y_ff, color='purple', label='new fitted ff function')

      plt.legend()
      plt.xlabel('angle (deg)')
      plt.ylabel('torque')
      plt.savefig('auto_feedforward/plots/{}.png'.format(speed_range_str))

  # if PLOT_3D := False:
  #   X_test = np.linspace(0, max(data_speeds), 20)
  #   Y_test = np.linspace(0, max(data_angles), 20)
  #
  #   Z_test = np.zeros((len(X_test), len(Y_test)))
  #   for i in range(len(X_test)):
  #     for j in range(len(Y_test)):
  #       Z_test[i][j] = CF.get(X_test[i], Y_test[j], *params)
  #
  #   X_test, Y_test = np.meshgrid(X_test, Y_test)
  #
  #   fig = plt.figure()
  #   ax = plt.axes(projection='3d')
  #
  #   surf = ax.plot_surface(X_test * CV.MS_TO_MPH, Y_test, Z_test, cmap=cm.magma,
  #                          linewidth=0, antialiased=False)
  #   fig.colorbar(surf, shrink=0.5, aspect=5)
  #
  #   ax.set_xlabel('speed (mph)')
  #   ax.set_ylabel('angle')
  #   ax.set_zlabel('feedforward')
  #   plt.title('New fitted polynomial feedforward function')


# Compares poly with old ff speed function
# x = np.linspace(0, 30, 100)
# y = x ** 2
# _c1, _c2, _c3 = 0.34365576041121065, 12.845373070976711, 51.63304088261174
# y_poly = _c1 * x ** 2 + _c2 * x + _c3
# plt.plot(x, y_poly, label='poly')
# plt.plot(x, y, label='v_ego**2')
# plt.legend()
# plt.show()


if __name__ == "__main__":
  # r = Route("14431dbeedbf3558%7C2020-11-10--22-24-34")
  # lr = MultiLogIterator(r.log_paths(), wraparound=False)
  use_dir = '/openpilot/pedal-ff/rlogs/use'
  # lr = MultiLogIterator([os.path.join(use_dir, i) for i in os.listdir(use_dir)], wraparound=False)
  n = fit_ff_model(use_dir, plot="--plot" in sys.argv)
