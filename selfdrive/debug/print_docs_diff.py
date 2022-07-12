#!/usr/bin/env python3
import os
import pickle

from common.basedir import BASEDIR
from selfdrive.car.docs import get_all_car_info
from selfdrive.car.docs_definitions import Column

STAR_ICON = '<a href="##"><img valign="top" src="https://raw.githubusercontent.com/commaai/openpilot/master/docs/assets/icon-star-{}.svg" width="22" /></a>'
COLUMNS = "|" + "|".join([column.value for column in Column]) + "|"
COLUMN_HEADER = "|---|---|---|:---:|:---:|:---:|:---:|:---:|"


def pretty_row(row, exclude=[Column.MAKE, Column.MODEL]):
  return {k.value: v for k, v in row.items() if k not in exclude}


def load_base_car_info():
  with open(os.path.join(BASEDIR, '../openpilot_cache/old_car_info'), 'rb') as f:  # TODO: rename to base
    return pickle.load(f)


def print_car_info_diff():
  base_car_info = load_base_car_info()

  base_car_info = {f'{i.make} {i.model}': i for i in base_car_info}
  new_car_info = {f'{i.make} {i.model}': i for i in get_all_car_info()}

  markdown_builder = ["### ⚠️ This PR makes changes to [CARS.md](../blob/master/docs/CARS.md) ⚠️"]

  # TODO: car changing tiers

  # TODO: diffs in cars
  # TODO: once we identify changes, need to remove it from base_car_info and new_car_info so they don't pick up changes in make+model+years as new/removed cars
  # for new_car, new_car_info in new_car_info.items():
  #   if new_car in base_car_info and new_car_info.row != base_car_info[new_car].row:
  #     print('Diff in car: {}'.format(new_car_info.row))

  # diffs = []
  # for car_info in get_all_car_info():

  deleted_cars = set(base_car_info) - set(new_car_info)
  if len(deleted_cars):
    markdown_builder.append("# Removed")
    markdown_builder.append(COLUMNS)
    markdown_builder.append(COLUMN_HEADER)
    for k in deleted_cars:
      car_info = base_car_info[k]
      markdown_builder.append("|" + "|".join([car_info.get_column(column, STAR_ICON, '{}') for column in Column]) + "|")

  added_cars = set(new_car_info) - set(base_car_info)
  if len(added_cars):
    markdown_builder.append("# Added")
    markdown_builder.append(COLUMNS)
    markdown_builder.append(COLUMN_HEADER)
    for k in added_cars:
      car_info = new_car_info[k]
      markdown_builder.append("|" + "|".join([car_info.get_column(column, STAR_ICON, '{}') for column in Column]) + "|")

  print("\n".join(markdown_builder))


if __name__ == "__main__":
  print_car_info_diff()
