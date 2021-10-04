import numpy as np
import matplotlib.pyplot as plt
from kf_tests.tools import load_data

DT_CTRL = 0.01


class KalmanFilter:
  def __init__(self):
    self.x_hat = 0  # steering angle
    self.x_hat_d = 0  # steering angle rate

  def update(self):
    self.x_hat = self.x_hat + self.x_hat_d * DT_CTRL
    self.x_hat_d = self.x_hat_d

  def predict(self, measurement_A, measurement_b):
    # A is steering angle, B is steering rate
    A = 0.5
    self.x_hat = self.x_hat + A * (measurement_A - self.x_hat)

    B = 0.5
    self.x_hat_d = measurement_b
    return [self.x_hat, self.x_hat_d]


class KalmanFilter2:
  def __init__(self):
    self.x_hat = 0  # steering angle
    self.x_hat_d = 0  # steering angle rate
    self.x_hat_j = 0  # steering angle rate rate (angle jerk)

  def update(self):
    self.x_hat = self.x_hat + self.x_hat_d * DT_CTRL + self.x_hat_j * (DT_CTRL ** 2 / 2)
    self.x_hat_d = self.x_hat_d + self.x_hat_j * DT_CTRL
    self.x_hat_j = self.x_hat_j

  def predict(self, measurement_A, measurement_B):
    # A is steering angle, B is steering rate
    A = 0.5
    x_hat_prev = float(self.x_hat)
    self.x_hat = self.x_hat + A * (measurement_A - self.x_hat)

    B = 0.2
    # self.x_hat_d = self.x_hat_d + B * (measurement_B - self.x_hat_d)
    self.x_hat_d = self.x_hat_d + B * ((measurement_A - x_hat_prev) / DT_CTRL)

    G = .002
    # self.x_hat_j = self.x_hat_j + G * ((measurement_B - self.x_hat_d) / DT_CTRL - self.x_hat_j)
    self.x_hat_j = self.x_hat_j + G * ((measurement_A - x_hat_prev) / (0.5 * DT_CTRL ** 2))
    return [self.x_hat, self.x_hat_d]


kf = KalmanFilter()
kf2 = KalmanFilter2()

data = load_data('steering_wheel_data')

kf_estimates = []
kf2_estimates = []
actual_data = []
kf_seen_data = []

angle = 0
for idx, line in enumerate(data):
  if line['v_ego'] < 5:
    continue
  kf.update()
  kf2.update()
  # if idx % 5 == 0:
  angle = np.random.normal(line['angle_steers'], 0.1/3)
  angle_rate = np.random.normal(line['angle_steers_rate'], 0.1/3)
  kf.predict(angle, angle_rate)
  kf2.predict(angle, angle_rate)

  kf_estimates.append(kf.x_hat)
  kf2_estimates.append(kf2.x_hat)
  kf_seen_data.append(angle)
  actual_data.append(line['angle_steers'])

  if idx % 2 == 0:
    window = 200
    x_range = [max(0, idx - window+1), idx+1]
    x_val = range(*x_range)
    plt.clf()
    plt.plot(x_val, actual_data[-window:], label='real steering angle')
    plt.plot(x_val, kf_seen_data[-window:], label='angle kalman filter sees')
    # plt.plot(x_val, kf_estimates[-window:], label='kf estimate')
    plt.plot(x_val, kf2_estimates[-window:], label='kf2 estimate')
    plt.legend()
    plt.pause(0.001)


