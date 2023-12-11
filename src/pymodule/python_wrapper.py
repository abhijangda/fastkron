import ctypes

libKron = ctypes.cdll.LoadLibrary("libKron.so")
def kronSGEMM(x, factors, y):
    libKron.kronSGEMM(ctypes.c_void_p(x), ctypes.c_void_p(factors))