# stateless averaging class
class KalmanMean:
  def __init__(self):
    self.x_hat = 0  # predicted state of x
    self.N = 0  # how many measurements we've been given

  @property
  def Kn(self):
    # Also known as the Kalman Gain, Kn
    return 1 / self.N

  def predict(self, measurement):
    # Kalman - State update equation
    self.N += 1
    self.x_hat = self.x_hat + self.Kn * (measurement - self.x_hat)
    return self.x_hat
