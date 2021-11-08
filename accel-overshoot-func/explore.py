from opendbc.can.parser import CANParser
from tools.lib.logreader import MultiLogIterator
from cereal import car
from common.basedir import BASEDIR
from selfdrive.car.toyota.values import CAR as TOYOTA_CAR, DBC as TOYOTA_DBC
from tqdm import tqdm   # type: ignore
import pickle
import os
from common.realtime import DT_CTRL
import matplotlib.pyplot as plt
from common.filter_simple import FirstOrderFilter


os.chdir(os.path.join(BASEDIR, 'accel-overshoot-func'))


def chunkify(lst, n):
  return [lst[i:i+n] for i in range(len(lst))[::n]]



# ego is desired/cmd
def remap_accel(a_ego, j_ego, _c1=-0.5, _c2=1):

  if j_ego > 0 and a_ego > 0 or j_ego < 0 and a_ego < 0:
    _c1 = -0.5
  else:
    _c1 = 0.5
  return a_ego * (_c1 * abs(j_ego) + _c2)


with open('data', 'rb') as f:  # now dump
  data = pickle.load(f)
# data = chunkify(data, 500)


filter_slow = FirstOrderFilter(0, 0.5, DT_CTRL)
filter_fast = FirstOrderFilter(0, 0.05, DT_CTRL)
LOOK_BACK = 50  # 0.5 seconds
TO_SEC = 1 / (LOOK_BACK * DT_CTRL)
for sec_idx, sec in enumerate(data):
  for idx, line in enumerate(sec):
    filter_slow.update(line['accel_cmd'])
    filter_fast.update(line['accel_cmd'])
    if idx < LOOK_BACK:
      continue

    # line['j_ego'] = (line['accel_cmd'] - sec[idx - LOOK_BACK]['accel_cmd']) * TO_SEC
    line['j_ego'] = (filter_fast.x - filter_slow.x) * TO_SEC
    line['accel_cmd_slow'] = filter_slow.x
    line['accel_cmd_fast'] = filter_fast.x
  sec = sec[LOOK_BACK:]
  data[sec_idx] = sec


# ACC_LOOK_BACK = 15  # some lag from accel request to realization
# for sec_idx, sec in enumerate(data):
#   accel_cmds = [line['accel_cmd'] for line in sec]
#   for idx, line in enumerate(sec):
#     if idx < ACC_LOOK_BACK:
#       continue
#     line['accel_cmd'] = accel_cmds[idx - ACC_LOOK_BACK]
#   data[sec_idx] = sec[ACC_LOOK_BACK:]
#   # data[sec_idx] = sec

print([len(sec) for sec in data])

data = data[0]

plt.plot([line['a_ego'] for line in data], label='a_ego')
plt.plot([line['accel_cmd'] for line in data], label='accel_cmd')
plt.plot([line['j_ego'] for line in data], label='j_ego')
# plt.plot([line['accel_cmd_slow'] for line in data], label='accel_cmd_slow')
new_accel_cmd = [remap_accel(line['accel_cmd'], line['j_ego']) for line in data]
plt.plot(new_accel_cmd, label='new_accel_cmd')

plt.legend()
