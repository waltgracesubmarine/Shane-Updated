#!/usr/bin/env python3
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # for seaborn
import os
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # type: ignore
from selfdrive.car.toyota.values import STEER_THRESHOLD
from scipy.signal import correlate
import seaborn as sns
from selfdrive.config import Conversions as CV

from common.realtime import DT_CTRL
from tools.lib.logreader import MultiLogIterator

MIN_SAMPLES = int(10 / DT_CTRL)


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
    v_ego = None

    all_msgs = sorted(lr, key=lambda msg: msg.logMonoTime)
    del lr

    for msg in tqdm(all_msgs):
      if msg.which() == 'liveLocationKalman':
        yaw_rate = msg.liveLocationKalman.angularVelocityCalibrated.value[2] * -150
      elif msg.which() == 'carState':
        v_ego = msg.carState.vEgo

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

      if engaged and not steering_pressed and None not in [torque_cmd, steer_angle, yaw_rate, v_ego]:  # creates uninterupted sections of engaged data
        if v_ego < 10 * CV.MPH_TO_MS:  # todo: experiment with 5 or lower
          continue
        data[-1].append({'engaged': engaged, 'torque_cmd': torque_cmd, 'steering_pressed': steering_pressed,
                         'steer_angle': steer_angle, 'time': msg.logMonoTime * 1e-9, 'yaw_rate': yaw_rate, 'v_ego': v_ego})
      elif len(data[-1]):
        data.append([])

  del all_msgs

  split = [sec for sec in data if len(sec) > MIN_SAMPLES]  # long enough sections
  del data
  assert len(split) > 0, "Not enough valid sections of samples"

  # print('max seq len: {}'.format(max([len(line) for line in split])))
  # print([len(line) for line in split])

  delays = []
  ptps = []
  seq_lens = []
  for idx, data in enumerate(split):
    torque = np.array([line['torque_cmd'] for line in data])
    angles = np.array([line['steer_angle'] for line in data])
    # rates = np.array([line['yaw_rate'] for line in data])
    if angles.std() == 0 or torque.std() == 0:
      print('warning: angles or torque std is 0! skipping...')
      continue
    angles_ptp = abs(angles.ptp())  # todo: abs shouldn't be needed, but just in case
    if angles_ptp <= 10:
      print('angle range too low ({} <= 10)! skipping...'.format(angles_ptp))
      continue
    print('peak to peak: {}'.format(angles.ptp(), np.ptp(angles)))

    # diff = max(torque) / max(angles)  # todo: not sure which normalization method is better. scale
    # angles *= diff

    Y1 = (angles - angles.mean()) / angles.std()  # todo: or regularization
    Y2 = (torque - torque.mean()) / torque.std()
    # rates = (rates - rates.mean()) / rates.std()

    plt.clf()
    # plt.ylim(-7, 10)
    plt.title('before offset')
    plt.plot(angles, label='angle')
    plt.plot(torque / (max(torque) / max(angles)), label='torque')
    # plt.plot(rates, label='rate')
    plt.legend()
    plt.savefig('/openpilot/steer_delay/plots/{}__before.png'.format(idx))

    xcorr = correlate(Y1, Y2)[angles.size - 1:]  # indexing forces positive offset (torque always precedes angle)
    time_shift = xcorr.argmax()

    if time_shift > 0:
      delays.append(time_shift)
      ptps.append(angles_ptp)
      seq_lens.append(len(data))

    print('len: {}'.format(len(data)))
    print('time shift: {}'.format(time_shift))
    print()

    torque = np.roll(torque, time_shift)
    plt.clf()
    # plt.ylim(-7, 10)
    plt.title('after offset ({})'.format(time_shift))
    plt.plot(angles, label='angle')
    plt.plot(torque/(max(torque) / max(angles)), label='torque')
    # plt.plot(rates, label='rate')
    plt.legend()
    plt.savefig('/openpilot/steer_delay/plots/{}_after.png'.format(idx))

  plt.clf()
  sns.distplot(delays, bins=40, label='delays')
  plt.legend()
  plt.savefig('/openpilot/steer_delay/dist.png')

  with open('/openpilot/steer_delay/data', 'wb') as f:
    pickle.dump(split, f)

  print('mean vels:  {}'.format([np.round(np.mean([line['v_ego'] * CV.MS_TO_MPH for line in sec]), 2) for sec in split]))  # todo: add vel list
  print('seq lens:   {}'.format(seq_lens))
  print('ptp angles: {}'.format(ptps))
  print('delays:     {}'.format(delays))

  print('\nmedian: {}'.format(np.median(delays)))
  print('mean: {}'.format(np.mean(delays)))


if __name__ == "__main__":
  # r = Route("14431dbeedbf3558%7C2020-11-10--22-24-34")
  # lr = MultiLogIterator(r.log_paths(), wraparound=False)

  find_steer_delay(plot="--plot" in sys.argv)
