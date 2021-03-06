#!/usr/bin/env python3
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import sys
from models.konverter.accel_to_gas import predict as best_model_predict

from numpy.random import seed
seed(2147483648)

try:
  from opendbc.can.parser import CANParser
  from tools.lib.logreader import MultiLogIterator
  from tools.lib.route import Route
  from cereal import car
  os.chdir('/openpilot/pedal-ff')
except:
  sys.path.insert(0, 'C:/Git/openpilot-repos/op-smiskol')
  os.environ['PYTHONPATH'] = '.'

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from tqdm import tqdm   # type: ignore
from scipy.optimize import curve_fit
from common.numpy_fast import interp

from selfdrive.config import Conversions as CV
import seaborn as sns
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import activations
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
import tensorflow as tf
import pickle

import wandb
from wandb.keras import WandbCallback


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  pass

DT_CTRL = 0.01
MIN_SAMPLES = 5 / DT_CTRL  # seconds to frames
MAX_GAS_INTERCEPTOR = 232


def transform_car_gas(car_gas):
  CAR_GAS_TO_CMD_POLY = [0.86765184, 0.03172896]  # i was pretty close fitting by hand, but this is the most accurate
  return np.polyval(CAR_GAS_TO_CMD_POLY, car_gas)


hyperparameter_defaults = dict(
  dropout_1=0,  # 1/15,
  dropout_2=0,  # 1/20,

  dense_1=2,
  dense_2=2,

  optimizer='adadelta',
  batch_size=16,
  learning_rate=1.,
  epochs=1000,
)
wandb.init(project="pedal-fix", config=hyperparameter_defaults)
config = wandb.config


def coast_accel(speed):  # given a speed, output coasting acceleration
  points = [[0.0, 0.538], [1.697, 0.28],
            [2.853, -0.199], [3.443, -0.249],
            [19.0 * CV.MPH_TO_MS, -0.145]]
  return interp(speed, *zip(*points))


def compute_gb_old(accel, speed):
  # return (accel * 0.5 + (0.05 * (speed / 20 + 1))) * (speed / 25 + 1)
  return float(accel) / 3.0


def coasting_func(x_input, _c1, _c2, _c3):  # x is speed
  return _c3 * x_input ** 2 + _c1 * x_input + _c2


def build_model(shape):
  model = models.Sequential()
  model.add(layers.Input(shape=shape))
  model.add(layers.Dense(config.dense_1, layers.LeakyReLU()))
  model.add(layers.Dropout(config.dropout_1))
  model.add(layers.Dense(config.dense_2, layers.LeakyReLU()))
  model.add(layers.Dropout(config.dropout_2))
  model.add(layers.Dense(1))

  # opt = optimizers.Adam(learning_rate=0.001 / 3, amsgrad=True)
  return model


def fit_all(x_input, _a3, _a4, _a5, _offset, _e1, _e2, _e3, _e4, _e5, _e6, _e7, _e8):
  """
    x_input is array of a_ego and v_ego
    all _params are to be fit by curve_fit
    kf is multiplier from angle to torque
    c1-c3 are poly coefficients
  """
  a_ego, v_ego = x_input.copy()
  speed_part = (_e5 * a_ego + _e6) * v_ego ** 2 + (_e7 * a_ego + _e8) * v_ego
  accel_part = ((_e1 * v_ego + _e2) * a_ego ** 5 + (_e3 * v_ego + _e4) * a_ego ** 4 + _a3 * a_ego ** 3 + _a4 * a_ego ** 2 + _a5 * a_ego)
  return speed_part + accel_part + _offset

  # return (a_ego * _c1 + (_c4 * (v_ego * _c2 + 1))) * (v_ego * _c3 + 1)
  # return _c4 * a_ego + np.polyval([_c1, _c2, _c3], v_ego)  # use this if we think there is a non-linear speed relationship


def known_bad_accel_to_gas(accel, speed):
  poly, accel_coef = [0.00011699240374307696, 0.01634332377590708, -0.0018321108362775451], 0.1166783696247945
  return (poly[0] * speed ** 2 + poly[1] * speed + poly[2]) + (accel_coef * accel)


def known_good_accel_to_gas(desired_accel, speed):
  # _s1, _s2, _a1, _a2, _a3, offset = [0.00011472033606023426, 0.013876213350425473, 0.023210616718880393, -0.09075484756780133, 0.1956935192117445, -0.01095416947337976]
  # speed_part = (_s1 * speed ** 2 + _s2 * speed)  # this can be linear
  # accel_part = (_a1 * desired_accel ** 3 + _a2 * desired_accel ** 2 + _a3 * desired_accel)
  # return accel_part + speed_part + offset

  # _c1, _c2, _c3, _c4 = [0.015332129994618495, -0.013848089187675144, -0.05406226668839383, 0.180209019025656]  # this function is smooth at low speed
  # return (_c1 * speed + _c2) + (_c3 * desired_accel ** 2 + _c4 * desired_accel)  # but didn't give enough gas above 10 mph

  # this function was accurate at high speeds and fairly smooth at low speeds but mashed on the gas sometimes
  _s1, offset = [((0.011 + .02) / 2 + .0155) / 2, 0.011371989131620245 - .02 - (.016 + .0207) / 2]  # these two have been tuned manually since the curve_fit function didn't seem exactly right
  _a1, _a2, _a3 = [0.022130745681601702, -0.09109186615316711, 0.20997207156680778]
  speed_part = (_s1 * speed)
  accel_part = (_a1 * desired_accel ** 3 + _a2 * desired_accel ** 2) * np.interp(speed, [10. * CV.MPH_TO_MS, 19. * CV.MPH_TO_MS], [1, 0.6])  # todo make this a linear function and clip (quicker)
  accel_part += (_a3 * desired_accel)
  accel_part *= np.interp(desired_accel, [0, 2], [0.8, 1])
  # offset -= np.interp(speed, [0 * CV.MPH_TO_MS, 6 * CV.MPH_TO_MS], [.04, 0]) * np.interp(a_ego, [0.5, 2], [1, 0])  # np.clip(1 - a_ego, 0, 1)
  return accel_part + speed_part + offset


def load_processed(file_name):
  with open(file_name, 'rb') as f:
    return pickle.load(f)


def get_accel_delay(speed):
  return int(np.interp(speed, [5 * CV.MPH_TO_MS, 12.5 * CV.MPH_TO_MS], [20, 40]))


def offset_accel(_data):  # todo: offsetting both speed and accel seem to decrease model loss by a LOT. maybe we should just offset all gas instead of these two todo: maybe not?
  for i in range(len(_data)):  # accounts for delay (moves a_ego up by x samples since it lags behind gas)
    # v_ego = [line['v_ego'] for line in _data[i]]
    a_ego = [line['a_ego'] for line in _data[i]]
    data_len = len(_data[i])
    for j in range(data_len):
      accel_delay = get_accel_delay(_data[i][j]['v_ego'])  # interpolate accel delay from speed
      # if j < accel_delay:  # (v_ego)
      #   continue
      # _data[i][j]['v_ego'] = v_ego[j - accel_delay]
      if j + accel_delay >= data_len:  # (a_ego)
        break
      _data[i][j]['a_ego_current'] = float(_data[i][j]['a_ego'])
      _data[i][j]['a_ego'] = float(a_ego[j + accel_delay])
    # _data[i] = _data[i][accel_delay:]  # removes leading samples (v_ego)
    _data[i] = _data[i][:-accel_delay]  # removes trailing samples (a_ego) (uses last accel delay)
  return _data


def load_and_process_rlogs(lrs, file_name):
  data = [[]]

  for lr in lrs:
    engaged, gas_enable, brake_pressed = False, False, False
    v_ego, gas_command, a_ego, pitch, steering_angle, gear_shifter = None, None, None, None, None, None
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
      ("SPORT_ON", "GEAR_PACKET", 0),
      ("GEAR", "GEAR_PACKET", 0),
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
        gear_shifter = msg.carState.gearShifter
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

      user_gas = (cp.vl['GAS_SENSOR']['INTERCEPTOR_GAS'] + cp.vl['GAS_SENSOR']['INTERCEPTOR_GAS2']) / 2.  # only for user todo: is the max 232?
      car_gas = cp.vl['GAS_PEDAL']['GAS_PEDAL']  # for user AND openpilot/car (less noisy than interceptor but need to check we're not engaged)

      brake_pressed = bool(cp.vl['BRAKE_MODULE']['BRAKE_PRESSED'])
      sport_on = bool(cp.vl['GEAR_PACKET']['SPORT_ON'])

      if msg.which() != 'can':  # only store when can is updated
        continue

      if abs(msg.logMonoTime - last_time) * 1e-9 > 1 / 20:
        print('TIME BREAK!')
        print(abs(msg.logMonoTime - last_time) * 1e-9)

      if (v_ego is not None and can_updated and gear_shifter == car.CarState.GearShifter.drive and not sport_on and  # creates uninterupted sections of engaged data
              abs(msg.logMonoTime - last_time) * 1e-9 < 1 / 20):  # also split if there's a break in time
        data[-1].append({'v_ego': v_ego, 'gas_command': gas_command, 'a_ego': a_ego, 'user_gas': user_gas,
                         'car_gas': car_gas, 'brake_pressed': brake_pressed, 'pitch': pitch, 'engaged': engaged, 'gas_enable': gas_enable,
                         'steering_angle': steering_angle,
                         'time': msg.logMonoTime * 1e-9})
      elif len(data[-1]):  # if last list has items in it, append new empty section
        data.append([])

      last_time = msg.logMonoTime

  del all_msgs

  print('Max seq. len: {}'.format(max([len(line) for line in data])))

  data = [sec for sec in data if len(sec) > 5 / DT_CTRL]  # long enough sections

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

  if os.path.exists('data_coasting'):  # for 2nd function that ouputs decel from speed (assuming coasting)
    data_coasting = load_processed('data_coasting')
  else:
    coast_dir = os.path.join(os.path.dirname(use_dir), 'coast')
    data_coasting = load_and_process_rlogs([MultiLogIterator([os.path.join(coast_dir, f) for f in os.listdir(coast_dir) if '.ini' not in f], wraparound=False)], file_name='data_coasting')

  # for data_0 in data:
  #   data_0 = [l for l in data_0 if not l['engaged']]
  #   if len(data_0) == 0:
  #     continue
  #   a_ego = np.array([l['a_ego'] for l in data_0])
  #   v_ego = np.array([l['v_ego'] for l in data_0])
  #   gas = np.array([l['car_gas'] for l in data_0])
  #   a_ego = a_ego / np.max(np.abs(a_ego))
  #   # a_ego = (a_ego - a_ego.mean()) / a_ego.std()
  #   # v_ego = (v_ego - v_ego.mean()) / v_ego.std()
  #   # gas = (gas - gas.mean()) / gas.std()
  #   plt.clf()
  #   plt.plot(a_ego, label='a_ego')
  #   plt.plot(np.roll(a_ego, -int(int(get_accel_delay(np.mean([i['v_ego'] * 2.2369 for i in data_0]))))), label='a_ego rolled')
  #   # plt.plot(v_ego, label='v_ego')
  #   plt.plot(gas, label='gas')
  #   plt.title(np.mean([i['v_ego'] * 2.2369 for i in data_0]))
  #   plt.legend()
  #   plt.pause(0.01)
  #   plt.show()
  #   input()
  # raise Exception

  if OFFSET_ACCEL := True:
    data = offset_accel(data)
  if COAST_OFFSET_ACCEL := True:
    data_coasting = offset_accel(data_coasting)

  # data_tmp = [l for l in [i for j in data for i in j] if l['engaged'] and l['user_gas'] < 14]  # todo: this all is to convert car gas to gas cmd scale. can be removed when done experimenting with
  # print(len(data_tmp))
  #
  # plt.plot([l['car_gas'] for l in data_tmp], 'o', label='og car gas')
  #
  # # def fit_car_gas_to_cmd(car, _c1, _c2):
  # #   return _c1 * car + _c2
  #
  # # params, _ = curve_fit(fit_car_gas_to_cmd, [l['car_gas'] for l in data_tmp], [l['gas_command'] for l in data_tmp])
  # # print(params)
  # plt.plot([transform_car_gas(l['car_gas']) for l in data_tmp], label='car gas (FITTED)')
  # plt.plot([l['gas_command'] for l in data_tmp], label='cmd')
  #
  # plt.legend()
  # plt.show()
  # plt.pause(0.01)
  # raise Exception

  # new_data = []
  # for sec in data:
  #   new_sec = []
  #   for idx, line in enumerate(sec):
  #     if idx + accel_delay < len(sec):
  #       if not sec[idx + accel_delay]['brake_pressed']:
  #         new_sec.append(sec)
  #       if not sec[idx]['brake_pressed'] and sec[idx + accel_delay]['brake_pressed']:
  #         print(line)
  #   new_data.append(new_sec)
  # raise Exception
  #
  # Removes cases where user brakes shortly after giving gas (gas would be positive, accel negative due to accel offsetting)
  # data = [[line for idx, line in enumerate(sec) if (not sec[idx + get_accel_delay(np.mean(i['v_ego'] for i in sec))]['brake_pressed'] if
  #                                                   idx + get_accel_delay(np.mean(i['v_ego'] for i in sec)) < len(sec) else False)] for sec in data]

  data = [i for j in data for i in j]  # flatten
  data_coasting = [i for j in data_coasting for i in j]  # flatten
  print(f'Samples (before filtering): {len(data)}')
  # data += data_coasting

  # Data filtering
  def general_filters(_line):  # general filters
    return 0.01 * CV.MPH_TO_MS < _line['v_ego'] < TOP_FIT_SPEED and not _line['brake_pressed'] and abs(_line['steering_angle']) <= 25

  data_coasting = [line for line in data_coasting if general_filters(line) and line['car_gas'] == 0 and not line['engaged']]

  engaged_samples = 0
  user_samples = 0
  # coast_user = []

  new_data = []
  for line in data:
    line = line.copy()
    if general_filters(line):
      # since car gas doesn't map to gas command perfectly, only use user samples where gas is above certain threshold
      if not line['engaged']:  # and 0.65 >= line['car_gas']:  # verified for sure working up to 0.65, but probably could go further
        if line['car_gas'] >= 0.2:  # if giving gas high enough for transform function (becomes inacurate 0 to ~.08)
          line['gas'] = float(transform_car_gas(line['car_gas']))  # this matches car gas up with gas cmd fairly accurately
        elif line['car_gas'] == 0 and np.random.randint(0, 100) < 25 and line['user_gas'] < 15:  # keep about a fourth coasting samples
          line['gas'] = 0
        else:
          continue
        user_samples += 1
        # else:  # coasting but speed not in range
        #   continue
      elif line['engaged'] and line['gas_enable'] and line['user_gas'] < 15:  # engaged and user not overriding
        # continue  # todo: skip engaged samples for now
        # # todo this is a hacky fix for bad data. i let op accidentally send gas cmd while not engaged and interceptor didn't like that so it wouldn't apply commanded gas WHILE ENGAGED sometimes. this gets rid of those samples
        # if line['gas_command'] == 0. or line['car_gas'] == 0 or abs(line['gas_command'] - transform_car_gas(line['car_gas'])) > 0.05:  # function avgs 0.011 error
        #   print('SHOULDNT BE HERE')
        #   continue
        engaged_samples += 1
        line['gas'] = float(line['gas_command'])
      else:
        continue

      new_data.append(line)

  data = new_data
  del new_data
  print('There are {} engaged samples and {} user samples!'.format(engaged_samples, user_samples))

  # print('There are {} user coast samples!'.format(len(coast_user)))
  # sns.distplot([line['a_ego'] for line in coast_user], bins=75)

  # raise Exception

  # data = [line for line in data if line['a_ego'] > coast_accel(line['v_ego']) - 0.1]  # this is experimental
  # data = [line for line in data if line['a_ego'] >= -0.5]  # sometimes a ego is -0.5 while gas is still being applied (todo: maybe remove going up hills? this should be okay for now)
  print(f'Samples (after filtering):  {len(data)}\n')

  print(f"Coasting samples: {len(data_coasting)}")

  temp_gas = [l['gas'] for l in data]
  print('Gas min: {} max: {}'.format(round(min(temp_gas), 5), round(max(temp_gas), 5)))
  sns.distplot(temp_gas, bins=75)
  plt.savefig('plots/gas dist.png')

  assert len(data) > MIN_SAMPLES, 'too few valid samples found in route'

  # Now prepare for function fitting
  data_speeds = np.array([line['v_ego'] for line in data])
  data_accels = np.array([line['a_ego'] for line in data])
  # data_cur_accels = np.array([line['a_ego_current'] for line in data])
  data_gas = np.array([line['gas'] for line in data])
  print('MIN ACCEL: {}'.format(min(data_accels)))
  print(f'accel: {np.min(data_accels), np.max(data_accels)}')
  print(f'speed: {np.min(data_speeds), np.max(data_speeds)}')

  x_train = np.array([data_accels, data_speeds]).T
  y_train = np.array(data_gas)

  model = build_model(x_train.shape[1:])
  if config.optimizer == 'adam':
    opt = optimizers.Adam(learning_rate=config.learning_rate, amsgrad=True)
  else:
    opt = optimizers.Adadelta(learning_rate=config.learning_rate)

  model.compile(opt, loss='mse', metrics=['mae'])
  # try:
  #   model.fit(x_train, y_train,
  #             batch_size=config.batch_size,
  #             epochs=config.epochs,
  #             validation_split=0.2,
  #             callbacks=[
  #               # tf.keras.callbacks.EarlyStopping('mae', patience=75),
  #               WandbCallback()
  #             ])
  # except KeyboardInterrupt:
  #   print('Training stopped!')
  # exit(0)

  # model = models.load_model('models/model-best.h5')

  params, covs = curve_fit(fit_all, x_train.T, y_train, maxfev=1000)
  # params = np.array([0.003837992717277964, -0.01235990011251591, 0.06510535652024786, 0.06600037259754446, -0.0006187306447074457, 0.000597369586548703, 0.0018908153873958748, -0.0004395380613128306, 0.00015113406209297302, 0.0003499560967296682, 0.002631675718307645, 0.0034227193219598844])
  print('Params: {}'.format(params.tolist()))
  # params = [((0.011+.02)/2 + .02) / 2, 0.022130745681601702, -0.09109186615316711, 0.20997207156680778, 0.011371989131620245 - .02 - (.016+.0207)/2]

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
    # plt.plot(x, coasting_func(x, *coast_params))
    # plt.plot(x, coasting_func(x, *coast_params), label='function')

    plt.plot(x, [coast_accel(_x) for _x in x], 'r', label='piecewise function')
    plt.savefig('imgs/coasting plot.png')

    plt.clf()
    x = np.linspace(0, 19 * CV.MPH_TO_MS, 100)
    y = [compute_gb_new(coast_accel(spd), spd) for spd in x]  # should be near 0
    plt.plot(x, y)
    plt.savefig('imgs/coasting plot-should-be-0.png')
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

  if PLOT_MODEL := True:
    plt.figure()
    plt.clf()
    known_good = [known_good_accel_to_gas(l['a_ego'], l['v_ego']) for l in data]
    # pred = model.predict_on_batch(np.array([[l['a_ego'], l['v_ego']] for l in data])).reshape(-1)
    pred = best_model_predict(np.array([[l['a_ego'], l['v_ego']] for l in data])).reshape(-1)
    fitted_function = [compute_gb_new(l['a_ego'], l['v_ego']) for l in data]

    # print(len(section))
    plt.plot([l['gas'] for l in data], label='gas (ground truth)')
    # plt.plot([l['a_ego'] / 3 for l in data], label='stock output')
    plt.plot(pred, label='model (prediction)')
    # plt.plot(known_good, label='last good')
    plt.plot(fitted_function, label='fitted function')
    plt.legend()
    plt.savefig('plots/model_plot.png')
    # raise Exception


  if ANALYZE_SPEED := True:
    plt.figure()
    sns.distplot([line['a_ego'] for line in data], bins=100)
    plt.savefig('plots/accel dist.png')

    res = 100
    color = 'blue'

    _accels = [
      [0, 0.25],
      [0.25, 0.5],
      [0.4, 0.6],
      [0.5, .75],
      [0.75, 1],
      [1, 1.25],
      [1.25, 1.5],
      [1.5, 1.75],
      [1.75, 2],
      [2, 2.5],
      [2.5, 3],
      [3, 4],
    ]

    for idx, accel_range in enumerate(_accels):
      accel_range_str = '{} m/s/s'.format('-'.join(map(str, accel_range)))
      temp_data = [line for line in data if accel_range[0] <= abs(line['a_ego']) <= accel_range[1]]
      if not len(temp_data):
        continue
      print(f'{accel_range} samples: {len(temp_data)}')
      plt.figure()
      speeds, gas = zip(*[[line['v_ego'], line['gas']] for line in temp_data])
      plt.scatter(np.array(speeds) * CV.MS_TO_MPH, gas, label=accel_range_str, color=color, s=0.05)

      _x_ff = np.linspace(min(speeds), max(speeds), res)

      # _y_ff = [known_bad_accel_to_gas(np.mean(accel_range), _i) for _i in _x_ff]
      # plt.plot(_x_ff * CV.MS_TO_MPH, _y_ff, color='red', label='bad ff function')
      _y_ff = [known_good_accel_to_gas(np.mean(accel_range), _i) for _i in _x_ff]
      plt.plot(_x_ff * CV.MS_TO_MPH, _y_ff, color='green', label='good ff function')

      _y_ff = [compute_gb_old(np.mean(accel_range), _i) for _i in _x_ff]
      # plt.plot(_x_ff * CV.MS_TO_MPH, _y_ff, color='orange', label='standard ff model at {} m/s/s'.format(np.mean(accel_range)))
      _y_ff = [compute_gb_new(np.mean(accel_range), _i) for _i in _x_ff]
      plt.plot(_x_ff * CV.MS_TO_MPH, _y_ff, color='purple', label='new fitted ff function')

      # _y_ff = [model.predict_on_batch(np.array([[np.mean(accel_range), _i]]))[0][0] for _i in _x_ff]
      _y_ff = [best_model_predict(np.array([[np.mean(accel_range), _i]]))[0][0] for _i in _x_ff]
      plt.plot(_x_ff * CV.MS_TO_MPH, _y_ff, color='cyan', label='model ff')

      plt.legend()
      plt.xlabel('speed (mph)')
      plt.ylabel('gas')
      plt.savefig('plots/s{}.png'.format(accel_range_str.replace('/', '')))


  if ANALYZE_ACCEL := True:
    plt.figure()
    sns.distplot([line['v_ego'] for line in data], bins=100)
    plt.savefig('plots/speed dist.png')

    res = 100

    _speeds = np.r_[[
      [0, 3],
      [3, 6],
      [6, 8],
      [8, 11],
      [11, 14],
      [14, 18],
      [18, 19],
    ]] * CV.MPH_TO_MS
    color = 'blue'

    for idx, speed_range in enumerate(_speeds):
      speed_range_str = '{} mph'.format('-'.join([str(round(i * CV.MS_TO_MPH, 1)) for i in speed_range]))
      temp_data = [line for line in data if speed_range[0] <= line['v_ego'] <= speed_range[1]]
      if not len(temp_data):
        continue
      print(f'{speed_range_str} samples: {len(temp_data)}')
      plt.figure()
      accels, gas, speeds = zip(*[[line['a_ego'], line['gas'], line['v_ego']] for line in temp_data])
      plt.scatter(accels, gas, label=speed_range_str, color=color, s=0.05)

      _x_ff = np.linspace(min(accels), max(accels), res)

      # _y_ff = [known_bad_accel_to_gas(_i, np.mean(speed_range)) for _i in _x_ff]
      # plt.plot(_x_ff, _y_ff, color='red', label='bad ff function')
      _y_ff = [known_good_accel_to_gas(_i, np.mean(speed_range)) for _i in _x_ff]
      plt.plot(_x_ff, _y_ff, color='green', label='good ff function')

      _y_ff = [compute_gb_old(_i, np.mean(speed_range)) for _i in _x_ff]
      # plt.plot(_x_ff, _y_ff, color='orange', label='standard ff model at {} mph'.format(np.round(np.mean(speed_range) * CV.MS_TO_MPH, 1)))
      _y_ff = [compute_gb_new(_i, np.mean(speed_range)) for _i in _x_ff]
      plt.plot(_x_ff, _y_ff, color='purple', label='new fitted ff function')

      # _y_ff = [model.predict_on_batch(np.array([[_i, np.mean(speed_range)]]))[0][0] for _i in _x_ff]
      _y_ff = [best_model_predict(np.array([[_i, np.mean(speed_range)]]))[0][0] for _i in _x_ff]
      plt.plot(_x_ff, _y_ff, color='cyan', label='model ff')

      plt.legend()
      plt.xlabel('accel (m/s/s)')
      plt.ylabel('gas')
      plt.savefig('plots/a{}.png'.format(speed_range_str))

  plt.show()

  return model

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
  model = fit_ff_model(use_dir, plot="--plot" in sys.argv)
