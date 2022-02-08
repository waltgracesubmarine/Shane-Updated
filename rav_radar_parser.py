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

# BO_ 128 ALT_RADAR: 8 XXX
#  SG_ INDEX : 7|8@0+ (1,0) [0|15] "" XXX
#  SG_ NEW_SIGNAL_1 : 15|8@0+ (1,0) [0|255] "" XXX
#  SG_ NEW_SIGNAL_2 : 23|8@0+ (1,0) [0|255] "" XXX
#  SG_ NEW_SIGNAL_3 : 31|8@1- (1,0) [0|255] "" XXX
#  SG_ NEW_SIGNAL_4 : 39|8@0+ (1,0) [0|255] "" XXX
#  SG_ NEW_SIGNAL_5 : 47|8@1- (1,0) [0|255] "" XXX
#  SG_ NEW_SIGNAL_6 : 55|8@0- (1,0) [0|255] "" XXX
#  SG_ NEW_SIGNAL_7 : 63|8@0- (1,0) [0|255] "" XXX


def get_bits(__byt):
  # Returns decimal data as bits displayed in cabana
  return tuple(map(int, bin(int(__byt))[2:].zfill(8)))


signals = [
  ("INDEX", "ALT_RADAR"),
  ("NEW_SIGNAL_1", "ALT_RADAR"),
  ("NEW_SIGNAL_2", "ALT_RADAR"),
  ("NEW_SIGNAL_3", "ALT_RADAR"),
  ("NEW_SIGNAL_4", "ALT_RADAR"),
  ("NEW_SIGNAL_5", "ALT_RADAR"),
  ("NEW_SIGNAL_6", "ALT_RADAR"),
  ("NEW_SIGNAL_7", "ALT_RADAR"),
  # ("TRACK_IDX1", "ALT_RADAR"),
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
# new_states = defaultdict(list)

for msg in tqdm(lr):
  if msg.which() == "can":
    cp.update_string(msg.as_builder().to_bytes())
    if len(cp.updated["ALT_RADAR"]["INDEX"]):
      for frame in zip(*[cp.updated["ALT_RADAR"][sig[0]] for sig in signals]):
        # print(frame[0])
        idx_bits = list(map(int, bin(int(frame[0]))[2:].zfill(8)))[4:8]
        idx_dat = idx_bits[0] * 2**3 + idx_bits[1] * 2**2 + idx_bits[2] * 2**1 + idx_bits[3] * 2**0
        states[idx_dat].append(frame)
        # print(frame)
        continue
        assert type(frame[0]) in [float, int]
        if len(states) < 18*2:
          states[len(states)].append(list(frame))
          continue

        frame_diffs = []
        for idx in range(len(states)):
          # [1:-1] filters out probable checksum
          frame_diffs.append(sum(np.abs(np.array(states[idx][-1][1:-1]) - np.array(frame[1:-1]))))

        closest_state = frame_diffs.index(min(frame_diffs))
        states[closest_state].append(frame)
        continue

plt.clf()
byt = 1
state = 12
plt.title('state={}'.format(state))
plt.plot([i[byt] for i in states[state][0]], label='byte={}'.format(byt))
plt.legend()
plt.show()

raise Exception

mismatches = defaultdict(lambda: defaultdict(int))
state_idx = 0
test_frame = states[state_idx][1::6][-1]

for test_y in range(8):
  test_bits = list(map(int, bin(int(test_frame[test_y]))[2:].zfill(8)))
  for test_x in range(8):

    for frame in states[state_idx][1::6]:
      byt = frame[test_y]
      bits = list(map(int, bin(int(byt))[2:].zfill(8)))
      if bits[test_x] != test_bits[test_x]:
        mismatches[test_y][test_x] += 1

for byt in range(8):
  if byt in mismatches:
    print(mismatches[byt].values())

plot_bit = (6, 0)  # byte, bit idx
bits_to_plot = []
for frame in states[state_idx]:
  bits = list(map(int, bin(int(frame[plot_bit[0]]))[2:].zfill(8)))
  bits_to_plot.append(bits[plot_bit[1]])
plt.plot(bits_to_plot)

# mismatches_by_count = {}
# print('Mismatches:')
# for msg in mismatches:
#   for byt_idx, byt in enumerate(mismatches):
#     for bit_idx, bit_mismatches in enumerate(byt):
#       # if bit_mismatches < 1000 and total_msgs[msg] > 1000:
#       perc_mismatched = 0  # round(bit_mismatches / total_msgs[msg] * 100, 2)
#       if perc_mismatched < 50:
#         mismatches_by_count[f'bit_mismatches={bit_mismatches} of {0} ({perc_mismatched}%), {byt_idx=}, {bit_idx=}'] = perc_mismatched
#         # print(f'{hex(msg)=}, bit_mismatches={bit_mismatches} of {total_msgs[msg]}, {byt_idx=}, {bit_idx=}')
#
# mismatches_sorted = sorted(mismatches_by_count, key=lambda msg: mismatches_by_count[msg], reverse=True)
# for msg in mismatches_sorted:
#   print(msg)




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
