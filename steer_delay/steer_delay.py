#!/usr/bin/env python3
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # type: ignore
from selfdrive.car.toyota.values import STEER_THRESHOLD
from scipy.signal import correlate

from common.realtime import DT_CTRL
from tools.lib.logreader import MultiLogIterator

MIN_SAMPLES = int(5 / DT_CTRL)


def to_signed(n, bits):
  if n >= (1 << max((bits - 1), 0)):
    n = n - (1 << max(bits, 0))
  return n


def find_steer_delay(lr, plot=False):
  data = [[]]
  engaged, steering_pressed = False, False
  torque_cmd, steer_angle = None, None

  all_msgs = sorted(lr, key=lambda msg: msg.logMonoTime)
  del lr

  for msg in tqdm(all_msgs):
    if msg.which() != 'can':
      continue

    for m in msg.can:
      if m.address == 0x2e4 and m.src == 128:
        engaged = bool(m.dat[0] & 1)
        torque_cmd = to_signed((m.dat[1] << 8) | m.dat[2], 16)
      elif m.address == 0x260 and m.src == 0:
        steering_pressed = abs(to_signed((m.dat[1] << 8) | m.dat[2], 16)) > STEER_THRESHOLD
      elif m.address == 0x25 and m.src == 0:
        steer_angle = to_signed(int(bin(m.dat[0])[2:].zfill(8)[4:] + bin(m.dat[1])[2:].zfill(8), 2), 12) * 1.5

    if engaged and not steering_pressed and torque_cmd is not None and steer_angle is not None:  # creates uninterupted sections of engaged data
      data[-1].append({'engaged': engaged, 'torque_cmd': torque_cmd, 'steering_pressed': steering_pressed, 'steer_angle': steer_angle, 'time': msg.logMonoTime * 1e-9})
    elif len(data[-1]):
      data.append([])

  del all_msgs

  split = [sec for sec in data if len(sec) > 100]  # long enough sections
  del data
  assert len(split) > 0, "Not enough valid sections of samples"

  print('max seq len: {}'.format(max([len(line) for line in split])))
  print([len(line) for line in split])

  for data in split:
    torque = np.array([line['torque_cmd'] for line in data])
    angles = np.array([line['steer_angle'] for line in data])
    n_samples = angles.size

    angles = (angles - angles.mean()) / angles.std()
    torque = (torque - torque.mean()) / torque.std()

    plt.clf()
    plt.plot(angles, label='angle')
    plt.plot(torque, label='torque')
    plt.legend()
    plt.savefig('/openpilot/steer_delay/test_before.png')

    xcorr = correlate(angles, torque)

    dt = np.arange(1 - n_samples, n_samples)
    recovered_time_shift = dt[xcorr.argmax()]
    print(len(data))
    print('time shift: {}'.format(recovered_time_shift))

    torque = np.roll(torque, recovered_time_shift)
    plt.clf()
    plt.plot(angles, label='angle')
    plt.plot(torque, label='torque')
    plt.legend()
    plt.savefig('/openpilot/steer_delay/test_after.png')
    input('press enter')


if __name__ == "__main__":
  # r = Route("14431dbeedbf3558%7C2020-11-10--22-24-34")
  # lr = MultiLogIterator(r.log_paths(), wraparound=False)
  use_dir = '/openpilot/steer_delay/rlogs/good'
  lr = MultiLogIterator([os.path.join(use_dir, i) for i in os.listdir(use_dir)], wraparound=False)
  find_steer_delay(lr, plot="--plot" in sys.argv)
