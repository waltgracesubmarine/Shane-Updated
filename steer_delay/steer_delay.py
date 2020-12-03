#!/usr/bin/env python3
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm   # type: ignore
from selfdrive.car.toyota.values import STEER_THRESHOLD
from scipy.signal import correlate

from common.realtime import DT_CTRL
from tools.lib.logreader import MultiLogIterator
import binascii

MIN_SAMPLES = 60 * 100


def to_signed(n, bits):
  if n >= (1 << max((bits - 1), 0)):
    n = n - (1 << max(bits, 0))
  return n


def find_steer_delay(lr, plot=False):
  data = []

  all_msgs = sorted(lr, key=lambda msg: msg.logMonoTime)
  del lr

  for msg in tqdm(all_msgs):
    if msg.which() == 'can':
      engaged, torque_cmd, steering_pressed, steer_angle = None, None, None, None
      for m in msg.can:
        if m.address == 0x2e4 and m.src == 128:
          engaged = bool(m.dat[0] & 1)
          torque_cmd = to_signed((m.dat[1] << 8) | m.dat[2], 16)
        elif m.address == 0x260 and m.src == 0:
          steering_pressed = abs(to_signed((m.dat[1] << 8) | m.dat[2], 16)) > STEER_THRESHOLD
        elif m.address == 0x25 and m.src == 0:
          steer_angle = to_signed(int(bin(m.dat[0])[2:].zfill(8)[4:] + bin(m.dat[1])[2:].zfill(8), 2), 12) * 1.5

      if engaged is not None and steering_pressed is not None and torque_cmd is not None and steer_angle is not None:
        if engaged and not steering_pressed:
          data.append({'engaged': engaged, 'torque_cmd': torque_cmd, 'steering_pressed': steering_pressed, 'steer_angle': steer_angle, 'time': msg.logMonoTime * 1e-9})

  del all_msgs


  # Now split data by time
  split = [[]]
  for idx, line in enumerate(data):  # split samples by time
    if idx > 0:  # can't get before first
      if line['time'] - data[idx - 1]['time'] > 1 / 20:  # 1/100 is rate but account for lag
        split.append([])
      split[-1].append(line)
  del data

  split = [sec for sec in split if len(sec) > 3 / DT_CTRL]  # long enough sections

  print('max seq len: {}'.format(max([len(line) for line in split])))
  print([len(line) for line in split])

  for data in split:
    torque = [line['torque_cmd'] for line in data]
    angles = [line['steer_angle'] for line in data]

    angles = np.array(angles)
    torque = np.array(torque)
    n_samples = angles.size

    angles = (angles - angles.mean()) / angles.std()
    torque = (torque - torque.mean()) / torque.std()

    plt.clf()
    plt.plot(angles)
    plt.plot(torque)
    plt.savefig('/openpilot/steer_delay/test_before.png')

    xcorr = correlate(angles, torque)

    dt = np.arange(1 - n_samples, n_samples)
    recovered_time_shift = dt[xcorr.argmax()]
    print(len(data))
    print('time shift: {}'.format(recovered_time_shift))

    torque = np.roll(torque, recovered_time_shift)
    plt.clf()
    plt.plot(angles)
    plt.plot(torque)
    plt.savefig('/openpilot/steer_delay/test_after.png')
    input('press enter')




if __name__ == "__main__":
  # r = Route("14431dbeedbf3558%7C2020-11-10--22-24-34")
  # lr = MultiLogIterator(r.log_paths(), wraparound=False)
  use_dir = '/openpilot/steer_delay/rlogs/good'
  lr = MultiLogIterator([os.path.join(use_dir, i) for i in os.listdir(use_dir)], wraparound=False)
  find_steer_delay(lr, plot="--plot" in sys.argv)
