#!/usr/bin/env python3
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

try:
  from opendbc.can.parser import CANParser
  from tools.lib.logreader import MultiLogIterator
  from tools.lib.route import Route
except:
  pass

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from tqdm import tqdm   # type: ignore
from scipy.optimize import curve_fit

from selfdrive.config import Conversions as CV
import seaborn as sns

import pickle
import binascii


DT_CTRL = 0.01
MIN_SAMPLES = 5 / DT_CTRL  # seconds to frames


def coast_accel(speed):  # given a speed, output coasting deceleration
  if speed < 0.384:  # this relationship is very nonlinear (below 5 mph it accelerates above it decelerates, but this piecewise function should do the trick)
    return (.565 / .324) * speed
  elif speed < 2.003:  # 2.003, .235
    return -0.1965455628350208 * speed + 0.6286807623585466
  elif speed < 2.71:  # 2.71, -.255
    return -0.6506364922206507 * speed + 1.5382248939179632
  elif speed < 6:  # 6, -.177
    return 0.014589665653495445 * speed - 0.26453799392097266
  else:  # 9.811, -.069
    return 0.028339018630280762 * speed - 0.3470341117816846


def compute_gb_old(accel, speed):
  # return (accel * 0.5 + (0.05 * (speed / 20 + 1))) * (speed / 25 + 1)
  return float(accel) / 3.0


def coasting_func(x_input, _c1, _c2, _c3):  # x is speed
  return _c3 * x_input ** 2 + _c1 * x_input + _c2


def fit_all(x_input, _c1, _c2, _c3, _c4):
  """
    x_input is array of a_ego and v_ego
    all _params are to be fit by curve_fit
    kf is multiplier from angle to torque
    c1-c3 are poly coefficients
  """
  a_ego, v_ego = x_input.copy()

  # return (_c1 * v_ego ** 2 + _c2 * v_ego + _c3) + (_c4 * a_ego)
  return (a_ego * _c1 + (_c4 * (v_ego * _c2 + 1))) * (v_ego * _c3 + 1)
  # return _c4 * a_ego + np.polyval([_c1, _c2, _c3], v_ego)  # use this if we think there is a non-linear speed relationship


def load_processed(file_name):
  with open(file_name, 'rb') as f:
    return pickle.load(f)


def load_and_process_rlogs(lrs, file_name):
  data = [[]]

  for lr in lrs:
    engaged, gas_enable, brake_pressed = False, False, False
    # torque_cmd, angle_steers, angle_steers_des, angle_offset, v_ego = None, None, None, None, None
    v_ego, gas_command, a_ego, user_gas, car_gas, pitch, steering_angle = None, None, None, None, None, None, None
    last_time = 0
    can_updated = False

    signals = [
      ("GAS_COMMAND", "GAS_COMMAND", 0),
      ("GAS_COMMAND2", "GAS_COMMAND", 0),
      ("ENABLE", "GAS_COMMAND", 0),
      ("INTERCEPTOR_GAS", "GAS_SENSOR", 0),
      ("INTERCEPTOR_GAS2", "GAS_SENSOR", 0),
      ("GAS_PEDAL", "GAS_PEDAL", 0),
      ("BRAKE_PRESSED", "BRAKE_MODULE", 0),
    ]
    cp = CANParser("toyota_corolla_2017_pt_generated", signals)

    all_msgs = sorted(lr, key=lambda msg: msg.logMonoTime)

    # gyro_counter = 0
    for msg in tqdm(all_msgs):
      if msg.which() == 'carState':
        v_ego = msg.carState.vEgo
        a_ego = msg.carState.aEgo
        steering_angle = msg.carState.steeringAngle
        engaged = msg.carState.cruiseState.enabled
      # elif msg.which() == 'sensorEvents':
      #   for sensor_reading in msg.sensorEvents:
      #     if sensor_reading.sensor == 4 and sensor_reading.type == 4:
      #       gyro_counter += 1
      #       if gyro_counter % 10 == 0:
      #         print(sensor_reading.gyro.v)
      #         pitch = float(np.degrees(sensor_reading.gyro.v[2]))
      # elif msg.which() == 'liveCalibration':
      #   pitch = float(np.degrees(msg.liveCalibration.rpyCalib[1]))

      if msg.which() not in ['can', 'sendcan']:
        continue
      cp_updated = cp.update_string(msg.as_builder().to_bytes())  # usually all can signals are updated so we don't need to iterate through the updated list

      for u in cp_updated:
        if u == 0x200:  # GAS_COMMAND
          can_updated = True

      gas_enable = bool(cp.vl['GAS_COMMAND']['ENABLE'])
      gas_command = max(round(cp.vl['GAS_COMMAND']['GAS_COMMAND'] / 255., 5), 0.0)  # unscale, round, and clip
      assert gas_command <= 1, "Gas command above 100%, look into this"

      user_gas = ((cp.vl['GAS_SENSOR']['INTERCEPTOR_GAS'] + cp.vl['GAS_SENSOR']['INTERCEPTOR_GAS2']) / 2.) / 232.  # only for user todo: is the max 232?
      car_gas = cp.vl['GAS_PEDAL']['GAS_PEDAL']  # for user AND openpilot/car (less noisy than interceptor but need to check we're not engaged)

      brake_pressed = bool(cp.vl['BRAKE_MODULE']['BRAKE_PRESSED'])

      if msg.which() != 'can':  # only store when can is updated
        continue

      if abs(msg.logMonoTime - last_time) * 1e-9 > 1 / 20:  # todo: remove once debugged
        print('TIME BREAK!')
        print(abs(msg.logMonoTime - last_time) * 1e-9)

      if (v_ego is not None and can_updated and  # creates uninterupted sections of engaged data
              abs(msg.logMonoTime - last_time) * 1e-9 < 1 / 20):  # also split if there's a break in time (todo: I don't think we need to check times)
        data[-1].append({'v_ego': v_ego, 'gas_command': gas_command, 'a_ego': a_ego, 'user_gas': user_gas,
                         'car_gas': car_gas, 'brake_pressed': brake_pressed, 'pitch': pitch, 'engaged': engaged, 'gas_enable': gas_enable,
                         'steering_angle': steering_angle,
                         'time': msg.logMonoTime * 1e-9})
      elif len(data[-1]):  # if last list has items in it, append new empty section
        data.append([])

      last_time = msg.logMonoTime

  del all_msgs

  print('Max seq. len: {}'.format(max([len(line) for line in data])))

  data = [sec for sec in data if len(sec) > 2 / DT_CTRL]  # long enough sections

  with open(file_name, 'wb') as f:  # now dump
    pickle.dump(data, f)
  return data


def fit_ff_model(use_dir, plot=False):
  TOP_FIT_SPEED = (19) * CV.MPH_TO_MS

  if os.path.exists('data'):
    data = load_processed('data')
  else:
    route_dirs = [f for f in os.listdir(use_dir) if '.ini' not in f and f != 'exclude']
    route_files = [[os.path.join(use_dir, i, f) for f in os.listdir(os.path.join(use_dir, i)) if f != 'exclude' and '.ini' not in f] for i in route_dirs]
    lrs = [MultiLogIterator(rd, wraparound=False) for rd in route_files]
    data = load_and_process_rlogs(lrs, file_name='data')

  if OFFSET_ACCEL := True:  # todo: play around with this
    accel_delay = int(0.75 / DT_CTRL)  # about .75 seconds from gas to a_ego  # todo: manually calculated from 10 samples on cabana, might need to verify with data
    for i in range(len(data)):  # accounts for delay (moves a_ego up by x samples since it lags behind gas)
      a_ego = [line['a_ego'] for line in data[i]]
      data_len = len(data[i])
      for j in range(data_len):
        if j + accel_delay >= data_len:
          break
        data[i][j]['a_ego'] = a_ego[j + accel_delay]
      data[i] = data[i][:-accel_delay]  # removes trailing samples

  if os.path.exists('data_coasting'):  # for 2nd function that ouputs decel from speed (assuming coasting)
    data_coasting = load_processed('data_coasting')
  else:
    coast_dir = os.path.join(os.path.dirname(use_dir), 'coast')
    data_coasting = load_and_process_rlogs([MultiLogIterator([os.path.join(coast_dir, f) for f in os.listdir(coast_dir) if '.ini' not in f], wraparound=False)], file_name='data_coasting')

  data = [i for j in data for i in j]  # flatten
  data_coasting = [i for j in data_coasting for i in j]  # flatten
  print(f'Samples (before filtering): {len(data)}')

  # Data filtering
  def general_filters(_line):  # general filters
    return _line['v_ego'] <= TOP_FIT_SPEED and not _line['brake_pressed'] and abs(_line['steering_angle']) <= 25  # and not _line['engaged']

  data_coasting = [line for line in data_coasting if general_filters(line) and line['car_gas'] == 0 and not line['engaged']]

  # todo get rid of long periods of stopped ness
  new_data = []
  for line in data:
    line = line.copy()
    if general_filters(line):
      if not line['engaged'] and not line['gas_enable']:  # user is driving
        line['gas'] = line['car_gas']  # user_gas (interceptor) doesn't map 1:1 with gas command so use car_gas which mostly does
      elif line['engaged'] and line['gas_enable']:  # car is driving and giving gas
        line['gas'] = line['gas_command']
      else:  # engaged but not commanding gas
        continue
      if line['car_gas'] > 0 and line['user_gas'] > 15 / 232:
        new_data.append(line)

  data = new_data
  # data = [line for line in data if line['a_ego'] > coast_accel(line['v_ego'])]
  data = [line for line in data if line['a_ego'] >= -0.5]  # sometimes a ego is -0.5 while gas is still being applied (todo: maybe remove going up hills? this should be okay for now)
  print(f'Samples (after filtering):  {len(data)}\n')
  print(f"Coasting samples: {len(data_coasting)}")

  assert len(data) > MIN_SAMPLES, 'too few valid samples found in route'

  # Now prepare for function fitting
  data_speeds = np.array([line['v_ego'] for line in data])
  data_accels = np.array([line['a_ego'] for line in data])
  data_gas = np.array([line['gas'] for line in data])

  params, covs = curve_fit(fit_all, np.array([data_accels, data_speeds]), np.array(data_gas), maxfev=1000)
  print('Params: {}'.format(params.tolist()))

  def compute_gb_new(accel, speed):
    return fit_all([accel, speed], *params)

  from_function = np.array([compute_gb_new(line['a_ego'], line['v_ego']) for line in data])
  print('Fitted function MAE: {}'.format(np.mean(np.abs(data_gas - from_function))))


  if len(data_coasting) > 100:
    print('\nFitting coasting function!')  # (not filtering a_ego gives us more accurate results)
    coast_params, covs = curve_fit(coasting_func, [line['v_ego'] for line in data_coasting], [line['a_ego'] for line in data_coasting])
    print('Coasting params: {}'.format(coast_params.tolist()))

    data_coasting_a_ego = np.array([line['a_ego'] for line in data_coasting])
    from_function = np.array([coasting_func(line['v_ego'], *coast_params) for line in data_coasting])
    print('Fitted coasting function MAE: {}'.format(np.mean(np.abs(data_coasting_a_ego - from_function))))

    plt.clf()
    plt.title('Coasting data')
    plt.scatter(*zip(*[[line['v_ego'], line['a_ego']] for line in data_coasting]), label='coasting data', s=2)
    x = np.linspace(0, TOP_FIT_SPEED, 100)
    plt.plot(x, coasting_func(x, *coast_params))
    plt.plot(x, coasting_func(x, *coast_params), label='function')

    def piece_wise_function(speed):  # this is very non-linear so create a piecewise function for it
      if speed < 0.384:
        return (.565/.324) * speed
      elif speed < 2.003:  # 2.003, .235
        return -0.1965455628350208 * speed + 0.6286807623585466
      elif speed < 2.71:  # 2.71, -.255
        return -0.6506364922206507 * speed + 1.5382248939179632
      elif speed < 6:  # 6, -.177
        return 0.014589665653495445 * speed - 0.26453799392097266
      else:  # 9.811, -.069
        return 0.028339018630280762 * speed - 0.3470341117816846

    plt.plot(x, [piece_wise_function(_x) for _x in x], 'r', label='piecewise function')
    plt.savefig('imgs/coasting plot.png')
  else:
    raise Exception('Not enough coasting samples')

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
    plt.savefig('plots/accel dist.png')
    plt.clf()

    res = 100
    color = 'blue'

    _accels = [
      [0, 0.5],
      [0.5, 1],
      [1, 1.25],
      [1.25, 1.5],
      [1.5, 2],
      [2, 2.5],
    ]

    for idx, accel_range in enumerate(_accels):
      accel_range_str = '{} m/s/s'.format('-'.join(map(str, accel_range)))
      temp_data = [line for line in data if accel_range[0] <= abs(line['a_ego']) <= accel_range[1]]
      if not len(temp_data):
        continue
      print(f'{accel_range} samples: {len(temp_data)}')
      plt.figure(idx)
      plt.clf()
      speeds, gas = zip(*[[line['v_ego'], line['gas']] for line in temp_data])
      plt.scatter(np.array(speeds) * CV.MS_TO_MPH, gas, label=accel_range_str, color=color, s=0.05)

      _x_ff = np.linspace(0, max(speeds), res)
      _y_ff = [compute_gb_old(np.mean(accel_range), _i) for _i in _x_ff]
      plt.plot(_x_ff * CV.MS_TO_MPH, _y_ff, color='orange', label='standard ff model at {} m/s/s'.format(np.mean(accel_range)))

      _y_ff = [compute_gb_new(np.mean(accel_range), _i) for _i in _x_ff]
      plt.plot(_x_ff * CV.MS_TO_MPH, _y_ff, color='purple', label='new fitted ff function')

      plt.legend()
      plt.xlabel('speed (mph)')
      plt.ylabel('gas')
      plt.savefig('plots/s{}.png'.format(accel_range_str.replace('/', '')))


  if ANALYZE_ACCEL := True:
    plt.clf()
    sns.distplot([line['v_ego'] for line in data], bins=100)
    plt.savefig('imgs/speed dist.png')
    plt.clf()

    res = 100

    _speeds = np.r_[[
      [0, 5],
      [5, 10],
      [10, 15],
      [15, 19],
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
      accels, gas, speeds = zip(*[[line['a_ego'], line['gas'], line['v_ego']] for line in temp_data])
      plt.scatter(accels, gas, label=speed_range_str, color=color, s=0.05)

      _x_ff = np.linspace(0, max(accels), res)
      _y_ff = [compute_gb_old(_i, np.mean(speed_range)) for _i in _x_ff]
      plt.plot(_x_ff, _y_ff, color='orange', label='standard ff model at {} mph'.format(np.round(np.mean(speed_range) * CV.MS_TO_MPH, 1)))

      _y_ff = [compute_gb_new(_i, np.mean(speed_range)) for _i in _x_ff]
      plt.plot(_x_ff, _y_ff, color='purple', label='new fitted ff function')

      plt.legend()
      plt.xlabel('accel (m/s/s)')
      plt.ylabel('gas')
      plt.savefig('plots/a{}.png'.format(speed_range_str))

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
