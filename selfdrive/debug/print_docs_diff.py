print("Start of print_docs_diff")
import os
import pickle

from common.basedir import BASEDIR
from selfdrive.car.docs import get_all_car_info
from selfdrive.car.docs_definitions import Column


def pretty_row(row, exclude=[Column.MAKE, Column.MODEL]):
  return {k.value: v for k, v in row.items() if k not in exclude}


with open(os.path.join(BASEDIR, 'old_car_info'), 'rb') as f:
  old_car_info = pickle.load(f)


old_car_info = {f'{i.make} {i.model}': i for i in old_car_info}
new_car_info = {f'{i.make} {i.model}': i for i in get_all_car_info()}

added_cars = set(new_car_info) - set(old_car_info)
deleted_cars = set(old_car_info) - set(new_car_info)

# print(added_cars, deleted_cars)
if len(added_cars):
  print('This PR adds these car entries:')
  for k in added_cars:
    car_info = new_car_info[k]
    print('{}: {}'.format(k, pretty_row(car_info.row)))

print()
if len(deleted_cars):
  print('This PR removes these car entries:')
  for k in deleted_cars:
    car_info = old_car_info[k]
    print('{}: {}'.format(k, pretty_row(car_info.row)))

# for new_car, new_car_info in new_car_info.items():
#   if new_car in old_car_info and new_car_info.row != old_car_info[new_car].row:
#     print('Diff in car: {}'.format(new_car_info.row))

# diffs = []
# for car_info in get_all_car_info():
