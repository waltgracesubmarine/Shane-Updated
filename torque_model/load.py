#!/usr/bin/env python3
import copy
import os

import matplotlib.pyplot as plt
import numpy as np

from selfdrive.config import Conversions as CV
import seaborn as sns
import pickle
from torque_model.helpers import feedforward, random_chance, TORQUE_SCALE

DT_CTRL = 0.01
MIN_SAMPLES = 5 / DT_CTRL  # seconds to frames

os.chdir('C:/Git/openpilot-repos/op-smiskol-torque/torque_model')


def load_processed(file_name):
  with open(file_name, 'rb') as f:
    return pickle.load(f)


def get_steer_delay(speed):
  return int(np.interp(speed, [5 * CV.MPH_TO_MS, 70 * CV.MPH_TO_MS], [0.32, .62]) * 100)


def offset_torque(_data):  # todo: offsetting both speed and accel seem to decrease model loss by a LOT. maybe we should just offset all gas instead of these two todo: maybe not?
  for i in range(len(_data)):  # accounts for steer actuator delay (from torque to change in angle)
    steering_angle = [line['steering_angle'] for line in _data[i]]
    steering_rate = [line['steering_rate'] for line in _data[i]]
    data_len = len(_data[i])
    steer_delay = get_steer_delay(0)
    for j in range(data_len):
      steer_delay = get_steer_delay(_data[i][j]['v_ego'])  # interpolate steer delay from speed
      if j + steer_delay >= data_len:
        break
      _data[i][j]['fut_steering_angle'] = float(steering_angle[j + steer_delay])
      _data[i][j]['fut_steering_rate'] = float(steering_rate[j + steer_delay])
    _data[i] = _data[i][:-steer_delay]  # removes trailing samples (uses last steer delay)
  return _data


def load_data():  # filters and processes raw pickle data from rlogs
  data = load_processed('data')
  # for sec in data:
  #   print('len: {}'.format(len(sec)))
  #   print('num. 0 steering_rate: {}'.format(len([line for line in sec if line['steering_rate'] == 0])))
  #   print()

  # there is a delay between sending torque and reaching the angle
  # this adds future steering angle and rate data to each sample, which we will use to train on as inputs
  # data for model: what current torque (output) gets us to the future (input)
  # this makes more sense than training on desired angle from lateral planner since humans don't always follow what the mpc would predict in any given situation
  data = offset_torque(data)

  # filter data
  KEEP_DATA = 'engaged'  # user, engaged, or all

  def sample_ok(line):
    return 33 * CV.MPH_TO_MS < line['v_ego'] and abs(line['steering_rate']) < 100 and abs(line['torque_eps']) < (TORQUE_SCALE * (4/3))

  filtered_data = []
  for sequence in data:
    filtered_sequence = []
    for line in sequence:
      if not sample_ok(line):
        continue

      if line['engaged'] and KEEP_DATA in ['all', 'engaged']:  # and random_chance(15):
        line['torque'] = line['torque_cmd']
        filtered_sequence.append(line)
      if not line['engaged'] and KEEP_DATA in ['all', 'user']:
        line['torque'] = line['torque_eps']  # i think eps makes more sense than driver
        filtered_sequence.append(line)

    if len(filtered_sequence):
      filtered_data.append(filtered_sequence)

  flattened = [i for j in filtered_data for i in j]  # flatten
  del data
  del filtered_sequence

  # Remove outliers
  stds = 3*2
  angle_cut_off = np.std([abs(line['steering_angle']) for line in flattened]) * stds
  rate_cut_off = np.std([abs(line['steering_rate']) for line in flattened]) * stds
  speed_cut_off = np.std([abs(line['v_ego']) for line in flattened]) * stds
  torque_cut_off = np.std([abs(line['torque']) for line in flattened]) * stds

  new_data = []
  for line in flattened:
    if abs(line['steering_angle']) < angle_cut_off and abs(line['steering_rate']) < rate_cut_off and \
      abs(line['v_ego']) < speed_cut_off and abs(line['torque']) < torque_cut_off:
      new_data.append(line)
  del flattened

  return new_data, filtered_data

  # filtered_data = []  # todo: check for disengagement (or engagement if disengaged) or user override in future
  # for sec in data:  # remove samples where we're braking in the future but not now
  #   new_sec = []
  #   for idx, line in enumerate(sec):
  #     accel_delay = get_accel_delay(line['v_ego'])  # interpolate accel delay from speed
  #     if idx + accel_delay < len(sec):
  #       if line['brake_pressed'] is sec[idx + accel_delay]['brake_pressed']:
  #         new_sec.append(line)
  #   if len(new_sec) > 0:
  #     filtered_data.append(new_sec)
  # data = filtered_data
  # del filtered_data


if __name__ == "__main__":
  speed_range = [10.778, 13.16]
  data, data_sequences = load_data()
  data = data_sequences[4]

  for idx, line in enumerate(data):
    past = 20
    if idx < past:
      line['steering_rate_calculated'] = 0
    else:
      calculated = line['steering_angle'] - data[idx - past]['steering_angle']
      line['steering_rate_calculated'] = calculated * (100 / past) if calculated != 0 else 0

  print(len(data_sequences))
  plt.plot([line['steering_rate_can'] for line in data], label='with fraction')
  plt.plot([line['steering_rate'] for line in data], label='current')
  # plt.plot([line['steering_rate_calculated'] for line in data], label='calculated (1 second)')
  plt.legend()

  raise Exception

  # data = [l for l in data if not l['engaged']]
  data = [l for l in data if speed_range[0] <= l['v_ego'] <= speed_range[1]]

  plt.figure()
  plt.plot([l['torque_eps'] for l in data], label='eps')
  plt.plot([l['torque_driver'] for l in data], label='driver')
  plt.legend()

  plt.figure()
  sns.distplot([l['v_ego'] for l in data], bins=200)

  plt.figure()
  angles = [abs(l['fut_steering_angle']) for l in data]
  torque = [abs(l['torque']) for l in data]
  x = np.linspace(0, max(angles), 100)
  plt.plot(x, [feedforward(_x, np.mean(speed_range)) * 0.00006908923778520113 * 1500 for _x in x])
  plt.scatter(angles, torque, s=0.5)
  plt.legend()
