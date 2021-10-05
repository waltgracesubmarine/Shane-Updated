import ast
import pandas as pd
import numpy as np

def load_data_csv(fn):
  df = pd.read_csv(fn, header=None)

  dataset = df.to_numpy()  # zss is broken, use angle_steers!
  print(dataset.shape)
  keys = ['angle_steers', 'shitty_angle', 'zss', 'output_steer', 'wheel_speeds.fl', 'wheel_speeds.fr', 'wheel_speeds.rl', 'wheel_speeds.rr']
  data = []
  for line in dataset:
    data.append(dict(zip(keys, line)))
  return data

def load_data(fn):
  data = []
  keys = None
  with open(fn, 'r') as f:
    for line in f.read().split('\n'):
      if keys is None:
        keys = ast.literal_eval(line)
        continue
      try:
        line = ast.literal_eval(line)
        data.append(dict(zip(keys, line)))
      except:
        pass
  return data
