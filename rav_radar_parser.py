import matplotlib
matplotlib.use('Qt5Agg')
import os
from tqdm import tqdm
os.environ['FILEREADER_CACHE'] = '1'
import matplotlib.pyplot as plt
from opendbc.can.parser import CANParser
from tools.lib.route import Route
from tools.lib.logreader import MultiLogIterator, LogReader
from collections import defaultdict
import numpy as np

signals = [
  ("INDEX", "ALT_RADAR"),
  ("NEW_SIGNAL_1", "ALT_RADAR"),
  ("NEW_SIGNAL_2", "ALT_RADAR"),
  ("NEW_SIGNAL_3", "ALT_RADAR"),
  ("NEW_SIGNAL_4", "ALT_RADAR"),
  ("NEW_SIGNAL_5", "ALT_RADAR"),
  ("NEW_SIGNAL_6", "ALT_RADAR"),
  ("NEW_SIGNAL_7", "ALT_RADAR"),
]

# byte 0: lateral distance?
# byte 1: distance?
# byte 2: id/counter?
# byte 3: distance?
# byte 6: id/counter?
# possible id/counter: byte 6, 2

cp = CANParser("toyota_nodsu_pt_generated", signals, enforce_checks=False, bus=1)

rt = Route("f751ac033bd10ce7|2022-02-05--09-17-41")
print('Got route')
lr = MultiLogIterator(rt.log_paths()[2:3])

index_list = []
# byte_0 = []
# byte_1 = []
# byte_2 = []
# byte_3 = []
# byte_4 = []
# byte_5 = []
# byte_6 = []
# byte_7 = []

idx_to_frame = defaultdict(list)  # idx_to_frame[track_idx][byte_idx] = (byte1, byte2,...)
idx_to_vals = defaultdict(lambda: defaultdict(list))  # idx_to_vals[track_idx][sig_name] = list of sigs

states = defaultdict(list)
new_states = defaultdict(list)

for msg in tqdm(lr):
  if msg.which() == "can":
    cp.update_string(msg.as_builder().to_bytes())
    if len(cp.updated["ALT_RADAR"]["INDEX"]):
      for frame in zip(*[cp.updated["ALT_RADAR"][sig[0]] for sig in signals]):
        # print(frame[0])
        # new_states[frame[0]].append(frame)
        # continue
        if len(states) < 100:
          states[len(states)].append(list(frame))
          continue

        frame_diffs = []
        for idx in range(len(states)):
          # filter out probable checksum
          frame_diffs.append(sum(np.abs(np.array(states[idx][-1][:-1]) - np.array(frame[:-1]))))

        closest_state = frame_diffs.index(min(frame_diffs))
        states[closest_state].append(frame)
        continue


        # diffs = []
        # already_added_idxs = []
        # for f in frame:
        #   diffs = []
        #   for s in states[-1]:
        #     diffs.append(abs(s - f))
        #   x.append(diffs)
        #
        #
        #
        # closest_matches = sorted(frame, key= lambda x: states[-1].index(sorted(states[-1], key=lambda i: abs(i - x))[0]))
        # states.append(closest_matches)
        # continue
        # # for idx, val in  enumerate(closest_matches):
        # #   states.append()
        # # for val in frame:


        track_idx = frame[0]
        index_list.append(track_idx)
        idx_to_frame[track_idx].append(frame[1:])
        for sig_name, val in zip(signals[1:], frame[1:]):
          idx_to_vals[track_idx][sig_name[0]].append(val)
      # byte_0 += (cp.updated["ALT_RADAR"]["INDEX"])
      #
      # byte_1 += (cp.updated["ALT_RADAR"]["NEW_SIGNAL_1"])
      # byte_2 += (cp.updated["ALT_RADAR"]["NEW_SIGNAL_2"])
      # byte_3 += (cp.updated["ALT_RADAR"]["NEW_SIGNAL_3"])
      # byte_4 += (cp.updated["ALT_RADAR"]["NEW_SIGNAL_4"])
      # byte_5 += (cp.updated["ALT_RADAR"]["NEW_SIGNAL_5"])
      # byte_6 += (cp.updated["ALT_RADAR"]["NEW_SIGNAL_6"])
      # byte_7 += (cp.updated["ALT_RADAR"]["NEW_SIGNAL_7"])

plt.clf()
plt.plot(index_list)
plt.show()
plt.pause(0.1)
raise Exception
track_idx = 0

for sig_name in idx_to_vals[track_idx]:
  print(sig_name)
  print(len(idx_to_vals[track_idx][sig_name]))
  plt.figure()
  plt.title('track idx: {}'.format(track_idx))
  plt.plot(idx_to_vals[track_idx][sig_name], label='signal={}'.format(sig_name))
  plt.legend()
  plt.show()
  # plt.pause(0.1)

# print(len(set(byte_0)))
# print(len(set(byte_1)))
# print(len(set(byte_3)))
# print(len(set(byte_4)))
# print(len(set(byte_5)))
# print(len(set(byte_6)))
# print(len(set(byte_7)))
