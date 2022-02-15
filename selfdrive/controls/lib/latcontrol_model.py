import math
import numpy as np
from common.basedir import BASEDIR
from selfdrive.controls.lib.drive_helpers import get_steer_max
from cereal import log
from common.numpy_fast import clip, interp
from common.realtime import DT_CTRL


class LatControlModel:
  def __init__(self, CP):
    # Model generated using Konverter: https://github.com/sshane/Konverter
    # model_weights_file = f'{BASEDIR}/models/steering/{CP.lateralTuning.model.name}_weights.npz'
    model_weights_file = f'{BASEDIR}/models/steering/toyota_big_data_v3_weights.npz'
    self.w, self.b = np.load(model_weights_file, allow_pickle=True)['wb']
    # these are in a specific order
    self.platforms = ['TOYOTA RAV4 HYBRID 2019',
                      'TOYOTA C-HR 2018',
                      'LEXUS RX 2020',
                      'TOYOTA AVALON HYBRID 2019',
                      'TOYOTA SIENNA 2018',
                      'TOYOTA MIRAI 2021',
                      'TOYOTA PRIUS TSS2 2021',
                      'TOYOTA C-HR HYBRID 2018',
                      'TOYOTA CAMRY HYBRID 2018',
                      'TOYOTA CAMRY 2021',
                      'LEXUS ES HYBRID 2019',
                      'TOYOTA PRIUS 2017',
                      'TOYOTA COROLLA HYBRID TSS2 2019',
                      'TOYOTA CAMRY HYBRID 2021',
                      'TOYOTA AVALON 2019',
                      'TOYOTA AVALON 2016',
                      'TOYOTA HIGHLANDER HYBRID 2018',
                      'TOYOTA HIGHLANDER HYBRID 2020',
                      'TOYOTA COROLLA 2017',
                      'TOYOTA RAV4 2019',
                      'LEXUS ES 2019',
                      'TOYOTA HIGHLANDER 2017',
                      'TOYOTA RAV4 HYBRID 2017',
                      'LEXUS RX 2016',
                      'TOYOTA CAMRY 2018',
                      'LEXUS RX HYBRID 2020',
                      'LEXUS NX HYBRID 2018',
                      'TOYOTA COROLLA TSS2 2019',
                      'TOYOTA RAV4 2017',
                      'LEXUS NX 2018',
                      'LEXUS RX HYBRID 2017',
                      'LEXUS NX 2020',
                      'LEXUS IS 2018',
                      'TOYOTA HIGHLANDER 2020']

    self.use_rates = True  # CP.lateralTuning.model.useRates
    self.sat_count_rate = 1.0 * DT_CTRL
    self.sat_limit = CP.steerLimitTimer
    self.one_hot_platform = self.get_one_hot(self.CP.carFingerprint)

    self.reset()

  def reset(self):
    self.sat_count = 0.0

  def _check_saturation(self, control, check_saturation, limit):
    saturated = abs(control) == limit

    if saturated and check_saturation:
      self.sat_count += self.sat_count_rate
    else:
      self.sat_count -= self.sat_count_rate

    self.sat_count = clip(self.sat_count, 0.0, 1.0)

    return self.sat_count > self.sat_limit

  def predict(self, x):
    x = np.array(x, dtype=np.float32)
    l0 = np.dot(x, self.w[0]) + self.b[0]
    l0 = np.where(l0 > 0, l0, l0 * 0.3)
    l1 = np.dot(l0, self.w[1]) + self.b[1]
    l1 = np.where(l1 > 0, l1, l1 * 0.3)
    l2 = np.dot(l1, self.w[2]) + self.b[2]
    return l2

  def get_one_hot(self, fp):
    if fp not in self.platforms:
      fp = 'TOYOTA COROLLA TSS2 2019'
    one_hot_platform = [1 if fp == platform else 0 for platform in self.platforms]
    assert sum(one_hot_platform) == 1
    return one_hot_platform

  def update(self, active, CS, CP, VM, params, last_actuators, desired_curvature, desired_curvature_rate):
    model_log = log.ControlsState.LateralModelState.new_message()
    model_log.steeringAngleDeg = float(CS.steeringAngleDeg)
    model_log.useRates = self.use_rates

    angle_steers_des_no_offset = math.degrees(VM.get_steer_from_curvature(-desired_curvature, CS.vEgo, params.roll))
    angle_steers_des = angle_steers_des_no_offset + params.angleOffsetDeg

    if CS.vEgo < 0.15 or not active:
      output_steer = 0.0
      model_log.active = False
    else:
      steers_max = get_steer_max(CP, CS.vEgo)
      pos_limit = steers_max
      neg_limit = -steers_max

      rate_des = math.degrees(VM.get_steer_from_curvature(-desired_curvature_rate, CS.vEgo, 0))

      # TODO: Can be sluggish when model is given rates, the issue is probably with the training data,
      # specifically the disturbances/perturbations fed into the model to recover from large errors
      # Basically, figure out a better way to train the model to recover without random samples and using a PF controller as the output
      rate_des = rate_des if self.use_rates else 0
      rate = CS.steeringRateDeg if self.use_rates else 0
      model_input = [angle_steers_des, CS.steeringAngleDeg, rate_des, rate, CS.vEgo, CS.aEgo]
      model_input += self.one_hot_platform

      output_steer = self.predict(model_input)[0]
      output_steer = clip(output_steer, neg_limit, pos_limit)
      output_steer = float(output_steer * CP.lateralTuning.model.multiplier)

      # TODO: we don't know this yet
      # if output_steer < 0:  # model doesn't like right curves
      #   _90_degree_bp = interp(CS.vEgo, [17.8816, 31.2928], [1., 1.1])  # 40 to 70 mph, 90 degree brakepoint
      #   multiplier = interp(abs(CS.steeringAngleDeg), [0, 90.], [1.27, _90_degree_bp])
      #   output_steer = float(output_steer * multiplier)

      model_log.active = True
      model_log.output = output_steer

      check_saturation = (CS.vEgo > 10) and not CS.steeringRateLimited and not CS.steeringPressed
      model_log.saturated = self._check_saturation(output_steer, check_saturation, steers_max)

    return output_steer, angle_steers_des, model_log
