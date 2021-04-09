import random
# from tensorflow import keras
from common.numpy_fast import interp
from torque_model.models.feedforward_model import predict as feedforward_predict

from selfdrive.config import Conversions as CV

TORQUE_SCALE = 1500


def feedforward(angle, speed):
  steer_feedforward = float(angle)  # offset does not contribute to resistive torque
  _c1, _c2, _c3 = 0.35189607550172824, 7.506201251644202, 69.226826411091
  steer_feedforward *= _c1 * speed ** 2 + _c2 * speed + _c3
  # steer_feedforward *= speed ** 2
  return steer_feedforward


# feedforward_model = keras.models.load_model('models/feedforward_model.h5', custom_objects={'LeakyReLU': keras.layers.LeakyReLU})


def model_feedforward(angle, speed):
  # return float(feedforward_model.predict_on_batch(np.array([[angle, angle, 0, speed]]))[0][0])
  return float(feedforward_predict([angle, angle, 0, speed])[0])


class LatControlPF:
  def __init__(self):
    self.k_f = 0.00006908923778520113
    # self.k_f = 0.00003
    self.speed = 0

  @property
  def k_p(self):
    return interp(self.speed, [20 * CV.MPH_TO_MS, 70 * CV.MPH_TO_MS], [.05, .15])

  def update(self, setpoint, measurement, speed, rate=0):
    self.speed = speed
    f = feedforward(setpoint, speed)
    # f = model_feedforward(setpoint, speed)
    f2 = 0#rate * speed ** 2 * 0.00003

    error = setpoint - measurement

    p = error * self.k_p
    f = f * self.k_f

    return p + f + f2  # multiply by 1500 to get torque units
    # return np.clip(p + steer_feedforward, -1, 1)  # multiply by 1500 to get torque units


def random_chance(percent: int):
  return random.randint(0, 100) < percent or percent == 100
