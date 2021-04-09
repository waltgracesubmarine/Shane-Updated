from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad
from tensorflow.keras.models import Sequential
import numpy as np
import random
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dropout

from torque_model.helpers import LatControlPF, TORQUE_SCALE, random_chance
from torque_model.load import load_data
from sklearn.model_selection import train_test_split
from selfdrive.config import Conversions as CV
import seaborn as sns

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.chdir('C:/Git/openpilot-repos/op-smiskol-torque/torque_model')

# print(tf.config.optimizer.get_experimental_options())
# tf.config.optimizer.set_experimental_options({'constant_folding': True, 'pin_to_host_optimization': True, 'loop_optimization': True, 'scoped_allocator_optimization': True})
# print(tf.config.optimizer.get_experimental_options())

try:
  tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

pid = LatControlPF()



inputs = ['fut_steering_angle', 'steering_angle', 'fut_steering_rate', 'steering_rate', 'v_ego']
# inputs = ['fut_steering_angle', 'steering_angle', 'v_ego']

data, data_high_delay, data_sequences, data_stats = load_data()
# del data_high_delay, data_sequences
print(f'Number of samples: {len(data)}')

# data += data_high_delay

# Normalize data
NORMALIZE_DATA = False
if NORMALIZE_DATA:
  scales = {}
  for inp in inputs:
    dat = [line[inp] for line in data]
    scales[inp] = [min(dat), max(dat)]
  for line in data:
    for inp in inputs:
      line[inp] = np.interp(line[inp], scales[inp], [-1, 1])

x_train = []
for line in data:
  x_train.append([line[inp] for inp in inputs])

y_train = []
for line in data:
  # the torque key is set by the data loader, it can come from torque_eps or torque_cmd depending on engaged status
  y_train.append(line['torque'])

print(f'Output (torque) min/max: {[min(y_train), max(y_train)]}')

# x_train = []  # only use synthetic samples
# y_train = []

# sns.distplot([abs(line['steering_angle']) for line in data], bins=200)
# plt.title('steering angle')
# plt.pause(0.01)
# input()
# plt.clf()
# sns.distplot([abs(line['steering_rate']) for line in data], bins=200)
# plt.title('steering rate')
# plt.pause(0.01)
# input()
# plt.clf()
# sns.distplot([line['v_ego'] for line in data], bins=200)
# plt.title('speed')
# plt.pause(0.01)
# input()
# plt.clf()
# sns.distplot([abs(line['torque']) for line in data], bins=200)
# plt.title('torque')
# plt.pause(0.01)
# input()


x_train = np.array(x_train)
y_train = np.array(y_train) / TORQUE_SCALE

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.35)


model = Sequential()
model.add(Input(shape=x_train.shape[1:]))
model.add(Dense(8, activation=LeakyReLU()))
# model.add(Dropout(1/8))
model.add(Dense(16, activation=LeakyReLU()))
# model.add(Dropout(1/16))
# model.add(Dense(24, activation=LeakyReLU()))
model.add(Dense(1))

epochs = 100
starting_lr = .5
ending_lr = 0.001
decay = (starting_lr - ending_lr) / epochs

opt = Adam(learning_rate=starting_lr, amsgrad=True, decay=decay)
# opt = Adadelta(learning_rate=1)
model.compile(opt, loss='mae', metrics='mse')
try:
  model.fit(x_train, y_train, batch_size=1024, epochs=100, validation_data=(x_test, y_test))
  # model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_test, y_test))
  # model.fit(x_train, y_train, batch_size=256, epochs=20, validation_data=(x_test, y_test))
  # model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))
  # model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_test, y_test))
  # model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test))
  # model.fit(x_train, y_train, batch_size=16, epochs=5, validation_data=(x_test, y_test))
except KeyboardInterrupt:
  pass


def plot_random_samples():
  idxs = np.random.choice(range(len(x_train)), 50)
  x_test = x_train[idxs]
  y_test = y_train[idxs].reshape(-1) * TORQUE_SCALE
  pred = model.predict(np.array([x_test])).reshape(-1) * TORQUE_SCALE

  plt.figure(0)
  plt.clf()
  plt.plot(y_test, label='ground truth')
  plt.plot(pred, label='prediction')
  plt.legend()
  plt.show()


plot_random_samples()


def plot_sequence(sequence_idx=3, show_controller=True):  # plots what model would do in a sequence of data
  sequence = data_sequences[sequence_idx]

  plt.figure(0)
  plt.clf()

  if show_controller:
    controller = [pid.update(line['fut_steering_angle'], line['steering_angle'], line['v_ego']) * TORQUE_SCALE for line in sequence]  # what a pf controller would output
    plt.plot(controller, label='standard controller')

  ground_truth = [line['torque'] for line in sequence]
  plt.plot(ground_truth, label='ground truth')

  _x = [[line[inp] for inp in inputs] for line in sequence]
  pred = model.predict(np.array(_x)).reshape(-1) * TORQUE_SCALE
  plt.plot(pred, label='prediction')

  plt.legend()
  plt.show()


