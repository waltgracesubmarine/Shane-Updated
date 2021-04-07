from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad
from tensorflow.keras.models import Sequential
import numpy as np
import random
import os
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# print(tf.config.optimizer.get_experimental_options())
# tf.config.optimizer.set_experimental_options({'constant_folding': True, 'pin_to_host_optimization': True, 'loop_optimization': True, 'scoped_allocator_optimization': True})
# print(tf.config.optimizer.get_experimental_options())

# try:
#   tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
# except:
#   # Invalid device or cannot modify virtual devices once initialized.
#   pass


class LatControlPF:
  def __init__(self):
    self.k_p = 0.1
    self.k_f = 0.00006908923778520113
    # self.k_f = 0.00003

  def update(self, setpoint, measurement, speed):
    steer_feedforward = setpoint  # offset does not contribute to resistive torque
    _c1, _c2, _c3 = 0.35189607550172824, 7.506201251644202, 69.226826411091
    steer_feedforward *= _c1 * speed ** 2 + _c2 * speed + _c3
    # steer_feedforward *= speed ** 2

    error = setpoint - measurement

    p = error * self.k_p
    f = steer_feedforward * self.k_f

    return p + f  # multiply by 1500 to get torque units
    # return np.clip(p + steer_feedforward, -1, 1)  # multiply by 1500 to get torque units


pid = LatControlPF()

# inputs = ['des_steering_angle', 'des_steering_rate', 'steering_angle', 'steering_rate', 'v_ego']
inputs = ['des_steering_angle', 'steering_angle', 'v_ego']

n_samples = 10000
x_train = []
for _ in range(n_samples):
  sample = {'des_steering_angle': random.uniform(-10, 10), 'des_steering_rate': random.uniform(-10, 10), 'v_ego': random.uniform(0, 35)}

  sample['steering_rate'] = sample['des_steering_rate'] + random.uniform(-10, 10)
  sample['steering_angle'] = sample['des_steering_angle'] + random.uniform(-10, 10)
  x_train.append([sample[inp] for inp in inputs])
x_train = np.array(x_train)

y_train = []
for sample in x_train:
  y = pid.update(sample[inputs.index('des_steering_angle')], sample[inputs.index('steering_angle')], sample[inputs.index('v_ego')])
  y_train.append(y)
y_train = np.array(y_train)

model = Sequential()
model.add(Input(shape=x_train.shape[1:]))
model.add(Dense(3, activation=LeakyReLU()))
model.add(Dense(6, activation=LeakyReLU()))
model.add(Dense(1))

epochs = 100
starting_lr = 0.5
ending_lr = 0.001
decay = (starting_lr - ending_lr) / epochs

opt = Adam(learning_rate=1, amsgrad=True, decay=decay)
# opt = Adagrad()
model.compile(opt, loss='mae')
try:
  model.fit(x_train, y_train,
            batch_size=64, epochs=epochs,
            validation_split=0.1)
except KeyboardInterrupt:
  pass


def plot_random_samples():
  idxs = np.random.choice(range(len(x_train)), 50)
  x_test = x_train[idxs]
  y_test = y_train[idxs].reshape(-1) * 1500
  pred = model.predict(np.array([x_test])).reshape(-1) * 1500

  plt.clf()
  plt.plot(y_test, label='ground truth')
  plt.plot(pred, label='prediction')
  plt.legend()
  plt.show()


plot_random_samples()
