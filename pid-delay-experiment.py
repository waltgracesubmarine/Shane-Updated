import numpy as np
import matplotlib.pyplot as plt
DT_CTRL = 0.01


class LatControlINDI:
  def __init__(self):
    self.reset()

  def reset(self):
    self.delayed_output = 0.

  def update(self, output_steer, RC=1):
    alpha = 1. - DT_CTRL / (RC + DT_CTRL)
    self.delayed_output = self.delayed_output * alpha + output_steer * (1. - alpha)

    return self.delayed_output


indi = LatControlINDI()

# A fake desired torque to the wheel from any lat controller
y = np.concatenate((np.linspace(0, 0, 50), np.linspace(0, 400, 10), np.linspace(400, 50, 10), np.linspace(50, 0, 100), np.linspace(0, -200, 10), np.linspace(-200, -850, 20), np.linspace(-850, 0, 150)))

x = np.linspace(0, len(y) / 100, len(y))
plt.plot(x, y, label='original output')

RC = [1., .75, .3, .1]  # timeConstantV for INDI (output smoothing)
for _RC in RC:
  indi.reset()
  y_delayed = []
  for _y in y:
    y_delayed.append(indi.update(_y, _RC))
  plt.plot(x, y_delayed, label=f'delayed output (RC={_RC})')

plt.xlabel('seconds')
plt.ylabel('torque output')
plt.legend()

