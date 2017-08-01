import numpy as np

def createTFIDFMatrix(data):
  total = sum(data[k] for k in data)
  tf = dict((k, float(data[k])/total) for k in data)

  print(tf)
