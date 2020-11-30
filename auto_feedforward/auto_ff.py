#!/usr/bin/env python3
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from tqdm import tqdm   # type: ignore
from scipy.optimize import curve_fit

from common.realtime import DT_CTRL
from selfdrive.config import Conversions as CV
from selfdrive.car.toyota.values import SteerLimitParams as TOYOTA_PARAMS
# from selfdrive.car.subaru.carcontroller import CarControllerParams as SUBARU_PARAMS
from tools.lib.route import Route
import seaborn as sns
from tools.lib.logreader import MultiLogIterator
import pickle
import binascii

k_f = 0.0000795769068  # for plotting old function compared to new polynomial function
MIN_SAMPLES = 60 * 100


def to_signed(n, bits):
  if n >= (1 << max((bits - 1), 0)):
    n = n - (1 << max(bits, 0))
  return n


def hex_to_binary(hexdata):
  return (bin(int(binascii.hexlify(hexdata), 16))[2:]).zfill(len(hexdata) * 8)  # adds leading/trailing zeros so data matches up with 8x8 array on cabana


def get_feedforward(v_ego, angle_steers, angle_offset=0):
  steer_feedforward = (angle_steers - angle_offset)
  steer_feedforward *= v_ego ** 2
  return steer_feedforward


def _custom_feedforward(_X, _k_f, _c1, _c2, _c3):  # automatically determines all params after input _X
  v_ego, angle_steers = _X.copy()
  # _c1, _c2, _c3 = 0.34365576041121065, 12.845373070976711, 51.63304088261174
  steer_feedforward = angle_steers * (_c1 * v_ego ** 2 + _c2 * v_ego + _c3)
  return steer_feedforward * _k_f


def custom_feedforward(v_ego, angle_steers, *args):  # helper function to easily use fitting ff function
  _X = np.array((v_ego, angle_steers)).T
  return _custom_feedforward(_X, *args)


def fit_ff_model(lr, plot=False):
  CAR_MAKE = 'toyota'
  data = []
  steer_delay = None
  last_plan = None
  last_can = None

  all_msgs = sorted(lr, key=lambda msg: msg.logMonoTime)
  del lr
  print(len(all_msgs))

  try:
    for msg in tqdm(all_msgs):

      if steer_delay is None:
        if msg.which() == 'carParams':
          steer_delay = round(msg.carParams.steerActuatorDelay / DT_CTRL)

      if msg.which() == 'carState':
        if last_plan is None or last_can is None:  # wait for other messages seen
          continue
        data.append({'angle_steers': msg.carState.steeringAngle, 'v_ego': msg.carState.vEgo, 'rate_steers': msg.carState.steeringRate,
                     'angle_steers_des': last_plan.pathPlan.angleSteers, 'angle_offset': last_plan.pathPlan.angleOffset,
                     'engaged': last_can['engaged'], 'torque': last_can['torque'], 'time': msg.logMonoTime * 1e-9})

      elif msg.which() == 'pathPlan':
        last_plan = msg

      elif msg.which() == 'can':
        for m in msg.can:
          if CAR_MAKE == 'toyota':
            if m.address == 0x2e4 and m.src == 128:
              last_can = {'engaged': bool(m.dat[0] & 1), 'torque': to_signed((m.dat[1] << 8) | m.dat[2], 16)}
              break
          elif CAR_MAKE == 'subaru':
            if m.address == 290 and m.src == 128:
              m = hex_to_binary(m.dat)
              last_can = {'torque': -to_signed(int(m[16 + 8 + 3:16 + 8 + 2 + 6] + m[16:16 + 8], 2), 13)}
              last_can['engaged'] = last_can['torque'] != 0  # bool(m[26]) should be correct, but for some reason it's always 1
              break

  except KeyboardInterrupt:
    print('Ctrl-C pressed, continuing...')

  MAX_TORQUE = TOYOTA_PARAMS.STEER_MAX if CAR_MAKE == 'toyota' else 2047

  print(f'{len(data)=}')
  data = [line for line in data if line['engaged']]  # remove disengaged
  print(f'{len(data)=} (engaged)')

  # Now split data by time
  split = [[]]
  for idx, line in enumerate(data):  # split samples by time
    if idx > 0:  # can't get before first
      if line['time'] - data[idx - 1]['time'] > 1 / 20:  # 1/100 is rate but account for lag
        split.append([])
      split[-1].append(line)
  del data

  print([len(line) for line in split])
  print(max([len(line) for line in split]))

  split = [sec for sec in split if len(sec) > DT_CTRL]  # long enough sections
  for i in range(len(split)):  # accounts for steer actuator delay (moves torque cmd up by 12 samples)
    torque = [line['torque'] for line in split[i]]
    for j in range(len(split[i])):
      if j < steer_delay:
        continue
      split[i][j]['torque'] = torque[j - steer_delay]
    split[i] = split[i][steer_delay:]  # removes leading samples

  data = [i for j in split for i in j]  # flatten
  del split

  print(f'Samples (before filtering): {len(data)}')

  # Data filtering
  data = [line for line in data if 1e-4 <= abs(line['angle_steers']) <= 60]
  # data = [line for line in data if abs(line['torque']) != 0]
  data = [line for line in data if abs(line['v_ego']) > 1 * CV.MPH_TO_MS]
  # data = [line for line in data if np.sign(line['angle_steers']) == np.sign(line['torque'])]
  data = [line for line in data if abs(line['angle_steers'] - line['angle_steers_des']) < 1]  # todo: should angle_steers be offset by angle_offset anywhere?
  print(max([i['torque'] for i in data]))
  print(min([i['torque'] for i in data]))

  print(f'Samples (after filtering): {len(data)}')
  with open('auto_feedforward/data', 'wb') as f:
    pickle.dump(data, f)

  # assert len(data) > MIN_SAMPLES, 'too few valid samples found in route'

  print('Max angle: {}'.format(max([abs(i['angle_steers']) for i in data])))
  print('Top speed: {} mph'.format(max([i['v_ego'] for i in data]) * CV.MS_TO_MPH))
  print('Max torque: {}'.format(max([i['torque'] for i in data])))

  # Data preprocessing
  for line in data:
    line['angle_steers'] = abs(line['angle_steers'])
    line['angle_steers_des'] = abs(line['angle_steers_des'])
    line['torque'] = abs(line['torque'])

    del line['time']

  data_speeds = np.array([line['v_ego'] for line in data])
  data_angles = np.array([line['angle_steers'] for line in data])
  data_torque = np.array([line['torque'] for line in data])

  # Tests
  assert all([i >= 0 for i in data_angles]), 'An angle sample is negative'
  assert all([i >= 0 for i in data_torque]), 'A torque sample is negative'
  assert steer_delay > 0, 'Steer actuator delay is zero'

  params, covs = curve_fit(_custom_feedforward, np.array([data_speeds, data_angles]), np.array(data_torque) / MAX_TORQUE, maxfev=800)
  print('FOUND PARAMS: {}'.format(params.tolist()))
  if params[-1] < 0:
    print('WARNING: intercept is negative, possibly bad fit! needs more data')

  std_func = []
  fitted_func = []
  for line in data:
    std_func.append(abs(get_feedforward(line['v_ego'], line['angle_steers']) * k_f * MAX_TORQUE - line['torque']))
    fitted_func.append(abs(custom_feedforward(line['v_ego'], line['angle_steers'], *params) * MAX_TORQUE - line['torque']))

  print('Torque MAE: {} (standard) - {} (fitted)'.format(np.mean(std_func), np.mean(fitted_func)))
  print('Torque STD: {} (standard) - {} (fitted)\n'.format(np.std(std_func), np.std(fitted_func)))

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
      _y_ff = [get_feedforward(_i, np.mean(angle_range)) * k_f * MAX_TORQUE for _i in _x_ff]
      plt.plot(_x_ff * CV.MS_TO_MPH, _y_ff, color='orange', label='standard ff model at {} deg'.format(np.mean(angle_range)))

      _y_ff = [custom_feedforward(_i, np.mean(angle_range), *params) * MAX_TORQUE for _i in _x_ff]
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
      _y_ff = [get_feedforward(np.mean(speed_range), _i) * k_f * MAX_TORQUE for _i in _x_ff]
      plt.plot(_x_ff, _y_ff, color='orange', label='standard ff model at {} mph'.format(np.round(np.mean(speed_range) * CV.MS_TO_MPH, 1)))

      _y_ff = [custom_feedforward(np.mean(speed_range), _i, *params) * MAX_TORQUE for _i in _x_ff]
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
  #       Z_test[i][j] = custom_feedforward(X_test[i], Y_test[j], *params)
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
  use_dir = '/openpilot/auto_feedforward/rlogs/shane/good'
  lr = MultiLogIterator([os.path.join(use_dir, i) for i in os.listdir(use_dir)], wraparound=False)
  n = fit_ff_model(lr, plot="--plot" in sys.argv)
