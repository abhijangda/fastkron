import fastkroncpp
from functools import reduce

class PyFastKron:
  def __init__(self):
    self.cpp_handle = fastkroncpp.pyFastKronInit()
    if self.cpp_handle == None:
      raise Exception("")

  def resultTempSizes(self, x, fs):
    return fastkroncpp.pyKronGeMMSizes(self.cpp_handle, 
                                       x.shape[0], len(fs), 
                                       [f.shape[0] for f in fs],
                                       [f.shape[1] for f in fs])
  
  def kmm(self, x, ps, qs, y):
    if (x.shape[1] != reduce((lambda a, b: a * b), ps)):
      return None
    # fastkroncpp.pyKronSGEMM(self.cpp_handle, x )