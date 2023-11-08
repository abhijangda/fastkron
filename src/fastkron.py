import fastkroncpp
from functools import reduce

class PyFastKron:
  def __init__(self):
    self.cpp_handle = fastkroncpp.pyFastKronInit()
    if self.cpp_handle == None:
      raise Exception("")

  def ps(self, fs):
    return [f.shape[0] for f in fs]
  
  def qs(self, fs):
    return [f.shape[1] for f in fs]

  def resultTempSizes(self, x, fs):
    (rs, ts) = fastkroncpp.pyKronGeMMSizes(self.cpp_handle, 
                                           x.shape[0], len(fs), 
                                           self.ps(fs), self.qs(fs))
    return ((x.shape[0], rs//x.shape[0]), (x.shape[0], ts//x.shape[0]))
  
  def kmm(self, x, fs, y, t1, t2):
    if (x.shape[1] != reduce((lambda a, b: a * b), self.ps(fs))):
      return None
    fastkroncpp.pyKronSGEMM(self.cpp_handle, x.shape[0], len(fs), 
                            self.ps(fs), self.qs(fs), 
                            x.data_ptr(), [f.data_ptr() for f in fs],
                            y.data_ptr(), t1.data_ptr(), t2.data_ptr())