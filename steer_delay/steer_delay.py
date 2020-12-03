#!/usr/bin/env python3
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # type: ignore
from selfdrive.car.toyota.values import STEER_THRESHOLD
from scipy.signal import correlate
import seaborn as sns

from common.realtime import DT_CTRL
from tools.lib.logreader import MultiLogIterator

MIN_SAMPLES = int(30 / DT_CTRL)


def to_signed(n, bits):
  if n >= (1 << max((bits - 1), 0)):
    n = n - (1 << max(bits, 0))
  return n


def find_steer_delay(plot=False):
  use_dir = '/openpilot/steer_delay/rlogs/good'

  routes = [[os.listdir(use_dir)[0]]]
  for rt in os.listdir(use_dir)[1:]:
    rt_name = ''.join(rt.split('--')[:2])
    if rt_name != ''.join(routes[-1][-1].split('--')[:2]):
      routes.append([rt])
    else:
      routes[-1].append(rt)

  print(routes)
  lrs = []
  for _routes in routes:
    lrs.append(MultiLogIterator([os.path.join(use_dir, i) for i in _routes], wraparound=False))


  data = [[]]
  for lr in lrs:
    engaged, steering_pressed = False, False
    torque_cmd, steer_angle = None, None
    yaw_rate = None

    all_msgs = sorted(lr, key=lambda msg: msg.logMonoTime)
    del lr

    for msg in tqdm(all_msgs):
      if msg.which() == 'liveLocationKalman':
        yaw_rate = msg.liveLocationKalman.angularVelocityCalibrated.value[2] * -150

      if msg.which() != 'can':
        continue

      for m in msg.can:
        if m.address == 0x2e4 and m.src == 128:  # STEERING_LKA
          engaged = bool(m.dat[0] & 1)
          torque_cmd = to_signed((m.dat[1] << 8) | m.dat[2], 16)
        elif m.address == 0x260 and m.src == 0:  # STEER_TORQUE_SENSOR
          steering_pressed = abs(to_signed((m.dat[1] << 8) | m.dat[2], 16)) > STEER_THRESHOLD
        elif m.address == 0x25 and m.src == 0:  # STEER_ANGLE_SENSOR
          steer_angle = to_signed(int(bin(m.dat[0])[2:].zfill(8)[4:] + bin(m.dat[1])[2:].zfill(8), 2), 12) * 1.5

      if engaged and not steering_pressed and torque_cmd is not None and steer_angle is not None and yaw_rate is not None:  # creates uninterupted sections of engaged data
        # if abs(steer_angle) < 5:
        #   continue
        data[-1].append({'engaged': engaged, 'torque_cmd': torque_cmd, 'steering_pressed': steering_pressed, 'steer_angle': steer_angle, 'time': msg.logMonoTime * 1e-9, 'yaw_rate': yaw_rate})
      elif len(data[-1]):
        data.append([])

  # data = [sec for sec in data if np.mean([abs(line['steer_angle']) for line in sec])]

  del all_msgs

  split = [sec for sec in data if len(sec) > MIN_SAMPLES]  # long enough sections
  del data
  assert len(split) > 0, "Not enough valid sections of samples"

  print('max seq len: {}'.format(max([len(line) for line in split])))
  print([len(line) for line in split])

  delays = []
  for idx, data in enumerate(split):
    torque = np.array([line['torque_cmd'] for line in data])
    angles = np.array([line['steer_angle'] for line in data])
    rates = np.array([line['yaw_rate'] for line in data])
    if angles.std() == 0 or torque.std() == 0:
      continue

    n_samples = angles.size

    # diff = max(torque) / max(angles)  # todo: not sure which regularization method is better. scale
    # angles *= diff

    angles = (angles - angles.mean()) / angles.std()  # todo: or normalization
    torque = (torque - torque.mean()) / torque.std()
    # rates = (rates - rates.mean()) / rates.std()

    plt.clf()
    plt.title('before offset')
    plt.plot(angles, label='angle')
    plt.plot(torque, label='torque')
    # plt.plot(rates, label='rate')
    plt.legend()
    plt.savefig('/openpilot/steer_delay/plots/{}__before.png'.format(idx))

    xcorr = correlate(angles, torque)[n_samples - 1:]  # indexing forces positive offset

    dt = np.arange(n_samples)  # old: np.arange(1 - n_samples, n_samples). new removes possibility of negative offsqet
    time_shift = dt[xcorr.argmax()]
    if time_shift > 0:
      delays.append(time_shift)

    print('len: {}'.format(len(data)))
    print('time shift: {}'.format(time_shift))
    print()

    torque = np.roll(torque, time_shift)
    plt.clf()
    plt.title('after offset ({})'.format(time_shift))
    plt.plot(angles, label='angle')
    plt.plot(torque, label='torque')
    # plt.plot(rates, label='rate')
    plt.legend()
    plt.savefig('/openpilot/steer_delay/plots/{}_after.png'.format(idx))

  plt.clf()
  sns.distplot(delays, bins=40, label='delays')
  plt.legend()
  plt.savefig('/openpilot/steer_delay/dist.png')
  print(delays)
  print('median: {}'.format(np.median(delays)))
  print('mean: {}'.format(np.mean(delays)))


if __name__ == "__main__":
  # r = Route("14431dbeedbf3558%7C2020-11-10--22-24-34")
  # lr = MultiLogIterator(r.log_paths(), wraparound=False)

  find_steer_delay(plot="--plot" in sys.argv)
