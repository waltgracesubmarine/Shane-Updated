#!/usr/bin/env python3
import os
import pickle
from collections import defaultdict

from common.basedir import BASEDIR
from selfdrive.car.docs import get_all_car_info
from selfdrive.car.docs_definitions import Column

STAR_ICON = '<a href="##"><img valign="top" src="https://raw.githubusercontent.com/commaai/openpilot/master/docs/assets/icon-star-{}.svg" width="22" /></a>'
COLUMNS = "|" + "|".join([column.value for column in Column]) + "|"
COLUMN_HEADER = "|---|---|---|:---:|:---:|:---:|:---:|:---:|"
ARROW_SYMBOL = "➡️"
# EXCLUDE_COLUMNS = [Column.MAKE, Column.MODEL]  # these are used as keys, so exclude diffs

# def combine_models(car_info):
#   model_car_info = defaultdict(list)
#   for car in car_info:
#     print(car.hi)
#     model_car_info[car.car_fingerprint].append(car)
#   return model_car_info


def pretty_row(row, exclude=[Column.MAKE, Column.MODEL]):
  return {k.value: v for k, v in row.items() if k not in exclude}


def load_base_car_info():
  with open(os.path.join(BASEDIR, '../openpilot_cache/old_car_info'), 'rb') as f:  # TODO: rename to base
    return pickle.load(f)


def get_diff(base_car, new_car):
  # print(base_car.row)
  # print(new_car.row)
  diff = {}
  # print(base_car.row == new_car.row)
  for column, value in base_car.row.items():
    if value != new_car.row[column]:
      diff[column] = (value, new_car.row[column])
  # print(diff)
  return diff


def print_car_info_diff():
  base_car_info = load_base_car_info()

  base_car_info = {f'{i.make} {i.model}': i for i in base_car_info}
  new_car_info = {f'{i.make} {i.model}': i for i in get_all_car_info()}

  markdown_builder = ["### ⚠️ This PR makes changes to [CARS.md](../blob/master/docs/CARS.md) ⚠️"]

  # # TODO: car changing tiers
  # # base_model_car_info = combine_models(load_base_car_info())
  # new_model_car_info = combine_models(get_all_car_info())
  # for car, cars in new_model_car_info.items():
  #   print(len(cars))
  #
  # return

  markdown_builder.append("# Changes")
  for base_car_model, base_car in base_car_info.items():
    if base_car_model not in new_car_info:
      continue

    new_car = new_car_info[base_car_model]
    diff = get_diff(base_car, new_car)  # TODO: can just return new value (or old)
    if not len(diff):
      continue

    row_builder = []
    for column in Column:
      if column not in diff:
        row_builder.append(new_car.get_column(column, STAR_ICON, '{}'))
      else:
        row_builder.append(base_car.get_column(column, STAR_ICON, '{}') + ARROW_SYMBOL + new_car.get_column(column, STAR_ICON, '{}'))
      # c = car_info.get_column(column, STAR_ICON, '{}') for column in Column

    markdown_builder.append("|" + "|".join(row_builder) + "|")

    # print(diff)
    # print(row_builder)
    # print()



  # return
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
