import numpy as np
import matplotlib.pyplot as plt
import ast
import os

os.chdir(os.getcwd())

with open('eager_accel_debug-good', 'r') as f:
  unparsed_data = f.read().split('\n')

data = []

for l in unparsed_data:
  if len(l) > 5:
    data.append(ast.literal_eval(l))

a_ego = [l['a_ego'] for l in data]
a_target = [l['a_target'] for l in data]
v_ego = [l['v_ego'] for l in data]
v_target = [l['v_target'] for l in data]
method = [l['eager_accel_method'] for l in data]

# plt.plot(v_ego, label='v_ego')
plt.plot(a_ego, label='a_ego')
# plt.plot(v_target, label='v_target')
plt.plot(a_target, label='a_target')
plt.plot([0 if l is None else l for l in method], label='method')
plt.legend()
plt.show()
