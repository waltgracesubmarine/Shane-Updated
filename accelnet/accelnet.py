#!/usr/bin/env python3
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import sys

from opendbc.can.parser import CANParser
from tools.lib.logreader import MultiLogIterator
from tools.lib.route import Route
from cereal import car

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from tqdm import tqdm   # type: ignore
from scipy.optimize import curve_fit

from selfdrive.config import Conversions as CV
import pickle

dir_name = os.path.dirname(os.path.realpath(__file__))
STEER_THRESHOLD = 100
DT_CTRL = 0.01

old_kf = 0.00004  # for plotting old function compared to new polynomial function
MIN_SAMPLES = int(30 / DT_CTRL)
MAX_TORQUE = 1500  # fixme: auto detect, this is toyota


def tokenize(data, seq_length):
  seq = []
  for i in range(len(data) - seq_length + 1):
    token = data[i:i + seq_length]
    if len(token) == seq_length:
      seq.append(token)
  return seq


def feedforward(v_ego, angle_steers, kf=1.):  # outputs unscaled ~lateral accel, with optional kf param to scale
  steer_feedforward = angle_steers * v_ego ** 2
  return steer_feedforward * kf


def _fit_kf(x_input, _kf):
  # Fits a kf gain using speed ** 2 feedforward model
  v_ego, angle_steers = x_input.copy()
  return feedforward(v_ego, angle_steers, _kf)


def load_processed(file_name):
  with open(file_name, 'rb') as f:
    return pickle.load(f)


def load_and_process_rlogs(lr, file_name):
  data = [[]]

  log_msgs = {
    'carState': {
      'vEgo': None,
      'aEgo': None,
      'gasPressed': None
    },
    'controlsState': {
      'enabled': False
    }
  }
  last_time = 0
  can_updated = False

  can_signals = [
    ("ACCEL_CMD", "ACC_CONTROL", 0),
  ]
  cp = CANParser("toyota_nodsu_hybrid_pt_generated", can_signals, enforce_checks=False)  # todo: auto load dbc from logs

  all_msgs = sorted(lr, key=lambda msg: msg.logMonoTime)

  for msg in tqdm(all_msgs):
    for log_msg in log_msgs:
      if msg.which() == log_msg:
        for log_signal in log_msgs[log_msg]:
          log_msgs[log_msg][log_signal] = getattr(getattr(msg, log_msg), log_signal)

    if msg.which() != 'sendcan':
      continue

    cp_updated = cp.update_string(msg.as_builder().to_bytes())
    for u in cp_updated:
      if u == 0x343:  # ACC_CONTROL
        can_updated = True

    accel_cmd = float(cp.vl['ACC_CONTROL']['ACCEL_CMD'])

    sample_ok = None not in [log_msgs['carState']['vEgo'], log_msgs['carState']['aEgo'], accel_cmd] and can_updated
    sample_ok = sample_ok and log_msgs['controlsState']['enabled'] and not log_msgs['carState']['gasPressed']

    # creates uninterupted sections of engaged data
    if sample_ok and abs(msg.logMonoTime - last_time) * 1e-9 < 1 / 20:  # also split if there's a break in time
      to_append = {'accel_cmd': accel_cmd, 'time': msg.logMonoTime * 1e-9}
      for log_signals in log_msgs.values():
        to_append = {**to_append, **log_signals}
      data[-1].append(to_append)
    elif len(data[-1]):  # if last list has items in it, append new empty section
      data.append([])
    last_time = msg.logMonoTime

  del all_msgs
  data = [sec for sec in data if len(sec) > (5 / DT_CTRL)]  # long enough sections

  with open(file_name, 'wb') as f:  # now dump
    pickle.dump(data, f)
  return data


def main(lr):
  if os.path.exists('processed_data'):
    print('exists')
    data = load_processed('processed_data')
  else:
    print('processing')
    data = load_and_process_rlogs(lr, file_name='processed_data')
  print('Max seq. len: {}'.format(max([len(line) for line in data])))
  print('Seq. lens: {}'.format([len(line) for line in data]))

  data_tokenized = []
  for seq in data:
    data_tokenized += tokenize(seq, round(2. / DT_CTRL))
  del data

  print(f'One second sequences: {len(data_tokenized)}')
  r = np.random.randint(25000)
  plt.plot([i['aEgo'] for i in data_tokenized[r]], label='aEgo')
  plt.plot([i['accel_cmd'] for i in data_tokenized[r]], label='acc_command')
  plt.legend()
  plt.pause(5)
  input()

  # Data preprocessing
  for line in data:
    line['angle_steers'] = abs(line['angle_steers'] - line['angle_offset'])  # need to offset angle to properly fit ff
    line['torque'] = abs(line['torque'])

    del line['time'], line['angle_offset']  # delete unused

  data_speeds = np.array([line['v_ego'] for line in data])
  data_angles = np.array([line['angle_steers'] for line in data])
  data_torque = np.array([line['torque'] for line in data])

  params, covs = curve_fit(_fit_kf, np.array([data_speeds, data_angles]), np.array(data_torque) / MAX_TORQUE,
                           # maxfev=800
                           )
  fit_kf = params[0]
  print('FOUND KF: {}'.format(fit_kf))

  print()

  std_func = []
  fitted_func = []
  for line in data:
    std_func.append(abs(feedforward(line['v_ego'], line['angle_steers'], old_kf) * MAX_TORQUE - line['torque']))
    fitted_func.append(abs(feedforward(line['v_ego'], line['angle_steers'], fit_kf) * MAX_TORQUE - line['torque']))
  print('Function comparison on input data')
  print('Torque MAE, current vs. fitted: {}, {}'.format(round(np.mean(std_func), 3), round(np.mean(fitted_func), 3)))
  print('Torque STD, current vs. fitted: {}, {}'.format(round(np.std(std_func), 3), round(np.std(fitted_func), 3)))

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
      plt.figure()
      speeds, torque = zip(*[[line['v_ego'], line['torque']] for line in temp_data])
      plt.scatter(np.array(speeds) * CV.MS_TO_MPH, torque, label=angle_range_str, color=color, s=0.05)

      _x_ff = np.linspace(0, max(speeds), res)
      _y_ff = [feedforward(_i, np.mean(angle_range), old_kf) * MAX_TORQUE for _i in _x_ff]
      plt.plot(_x_ff * CV.MS_TO_MPH, _y_ff, color='orange', label='standard ff model at {} deg'.format(np.mean(angle_range)))

      _y_ff = [feedforward(_i, np.mean(angle_range), fit_kf) * MAX_TORQUE for _i in _x_ff]
      plt.plot(_x_ff * CV.MS_TO_MPH, _y_ff, color='purple', label='new fitted ff function')

      plt.legend()
      plt.xlabel('speed (mph)')
      plt.ylabel('torque')
      plt.savefig('plots/{}.png'.format(angle_range_str))

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
      plt.figure()
      angles, torque, speeds = zip(*[[line['angle_steers'], line['torque'], line['v_ego']] for line in temp_data])
      plt.scatter(angles, torque, label=speed_range_str, color=color, s=0.05)

      _x_ff = np.linspace(0, max(angles), res)
      _y_ff = [feedforward(np.mean(speed_range), _i, old_kf) * MAX_TORQUE for _i in _x_ff]
      plt.plot(_x_ff, _y_ff, color='orange', label='standard ff model at {} mph'.format(np.round(np.mean(speed_range) * CV.MS_TO_MPH, 1)))

      _y_ff = [feedforward(np.mean(speed_range), _i, fit_kf) * MAX_TORQUE for _i in _x_ff]
      plt.plot(_x_ff, _y_ff, color='purple', label='new fitted ff function')

      plt.legend()
      plt.xlabel('angle (deg)')
      plt.ylabel('torque')
      plt.savefig('plots/{}.png'.format(speed_range_str))

  # if PLOT_3D := False:
  #   X_test = np.linspace(0, max(data_speeds), 20)
  #   Y_test = np.linspace(0, max(data_angles), 20)
  #
  #   Z_test = np.zeros((len(X_test), len(Y_test)))
  #   for i in range(len(X_test)):
  #     for j in range(len(Y_test)):
  #       Z_test[i][j] = feedforward(X_test[i], Y_test[j], fit_kf)
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


if __name__ == '__main__':
  # r = Route(sys.argv[1])
  # lr = MultiLogIterator(r.log_paths(), wraparound=False)
  use_dir = os.path.join(dir_name, 'rlogs')
  route_files = [os.path.join(use_dir, f) for f in os.listdir(use_dir) if f != 'exclude' and '.ini' not in f]
  print(route_files)
  lr = MultiLogIterator(route_files)

  n = main(lr)
