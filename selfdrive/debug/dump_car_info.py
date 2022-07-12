import os
import pickle

from common.basedir import BASEDIR
from selfdrive.car.docs import get_all_car_info

with open(os.path.join(BASEDIR, '../old_car_info'), 'wb') as f:
  pickle.dump(get_all_car_info(), f)
