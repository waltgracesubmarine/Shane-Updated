import numpy as np
import matplotlib.pyplot as plt
import ast

with open('eager_accel_debug-good', 'a') as f:
  data = f.read().split('\n')

data = [ast.literal_eval(line) for line in data if len(data) > 0]