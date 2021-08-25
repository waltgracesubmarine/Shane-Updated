#!/usr/bin/env python3
from opendbc.can.parser import CANParser
from tools.lib.logreader import MultiLogIterator
from cereal import car
from tqdm import tqdm   # type: ignore
import pickle
import os

DT_CTRL = 0.01
MIN_SAMPLES = 5 / DT_CTRL  # seconds to frames
os.chdir('/home/shane/Git/op-torque')


def load_and_process_rlogs(lrs, file_name):
  data = [[]]

  for lr in lrs:
    engaged, v_ego, a_ego, gear_shifter = None, None, None, None
    steering_angle, des_steering_angle, des_steering_rate = None, None, None
    last_engaged = None

    last_time = 0
    can_updated = False

    signals = [
      ("SPORT_ON", "GEAR_PACKET", 0),
      ("STEER_REQUEST", "STEERING_LKA", 0),
      ("STEER_TORQUE_CMD", "STEERING_LKA", 0),
      ("STEER_TORQUE_DRIVER", "STEER_TORQUE_SENSOR", 0),
      ("STEER_TORQUE_EPS", "STEER_TORQUE_SENSOR", 0),
      ("STEER_RATE_FRACTION", "STEER_ANGLE_SENSOR", 0),
      ("STEER_RATE", "STEER_ANGLE_SENSOR", 0),
    ]
    cp = CANParser("toyota_corolla_2017_pt_generated", signals)

    all_msgs = sorted(lr, key=lambda msg: msg.logMonoTime)

    for msg in tqdm(all_msgs):
      if msg.which() == 'carState':
        v_ego = msg.carState.vEgo
        a_ego = msg.carState.aEgo
        steering_angle = msg.carState.steeringAngleDeg
        last_engaged = bool(engaged)
        engaged = msg.carState.cruiseState.enabled
        gear_shifter = msg.carState.gearShifter
      # elif msg.which() == 'carControl':  # todo: maybe get eps torque
      #   apply_accel = msg.carControl.actuators.gas - msg.carControl.actuators.brake

      if msg.which() not in ['can', 'sendcan']:
        continue
      cp_updated = cp.update_string(msg.as_builder().to_bytes())  # usually all can signals are updated so we don't need to iterate through the updated list

      for u in cp_updated:
        if u == 0x3bc:  # GEAR_PACKET
          can_updated = True

      sport_on = bool(cp.vl['GEAR_PACKET']['SPORT_ON'])
      user_override = abs(cp.vl['STEER_TORQUE_SENSOR']['STEER_TORQUE_DRIVER']) > 100

      torque_cmd = cp.vl['STEERING_LKA']['STEER_TORQUE_CMD']
      torque_eps = cp.vl['STEER_TORQUE_SENSOR']['STEER_TORQUE_EPS']
      torque_driver = cp.vl['STEER_TORQUE_SENSOR']['STEER_TORQUE_DRIVER']
      steer_req = cp.vl['STEERING_LKA']['STEER_REQUEST'] == 1

      steering_rate = cp.vl['STEER_ANGLE_SENSOR']['STEER_RATE'] + cp.vl['STEER_ANGLE_SENSOR']['STEER_RATE_FRACTION']
      # steering_rate_fraction = cp.vl['STEER_ANGLE_SENSOR']['STEER_RATE_FRACTION']

      if msg.which() != 'can':  # only store when can is updated
        continue

      if abs(msg.logMonoTime - last_time) * 1e-9 > 1 / 20:
        print('TIME BREAK!')
        print(abs(msg.logMonoTime - last_time) * 1e-9)

      should_gather = not engaged
      if engaged and steer_req and not user_override:
        should_gather = True

      if (engaged is not None and can_updated and gear_shifter == car.CarState.GearShifter.drive and
              not sport_on and should_gather and engaged == last_engaged and  # creates uninterupted sections of engaged data
              abs(msg.logMonoTime - last_time) * 1e-9 < 1 / 20):  # also split if there's a break in time
        data[-1].append({'v_ego': v_ego, 'a_ego': a_ego, 'steering_angle': steering_angle, 'steering_rate': steering_rate,
                         # 'steering_rate_fraction': steering_rate_fraction,
                         'engaged': engaged, 'torque_cmd': torque_cmd, 'torque_eps': torque_eps, 'torque_driver': torque_driver,
                         'time': msg.logMonoTime * 1e-9})
      elif len(data[-1]):  # if last list has items in it, append new empty section
        data.append([])

      last_time = msg.logMonoTime

  del all_msgs

  print('Max seq. len: {}'.format(max([len(line) for line in data])))

  data = [sec for sec in data if len(sec) > MIN_SAMPLES]  # long enough sections

  with open(file_name, 'wb') as f:  # now dump
    pickle.dump(data, f)
  return data


if __name__ == "__main__":
  use_dir = 'torque_model/rlogs/use'
  route_dirs = [f for f in os.listdir(use_dir) if '.ini' not in f and f != 'exclude']
  route_files = [[os.path.join(use_dir, i, f) for f in os.listdir(os.path.join(use_dir, i)) if f != 'exclude' and '.ini' not in f] for i in route_dirs]
  lrs = [MultiLogIterator(rd, wraparound=False) for rd in route_files]
  load_and_process_rlogs(lrs, file_name='torque_model/data')
