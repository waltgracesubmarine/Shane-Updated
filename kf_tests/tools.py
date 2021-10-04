import ast


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
