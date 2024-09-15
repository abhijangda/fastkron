import platform
from .fastkronbase import fastkronX86, fastkronCUDA

if fastkronX86 != None and fastkronCUDA != None:
  __version__ = fastkronX86.version() + "+" + fastkronCUDA.version()
elif fastkronX86 != None:
  __version__ = fastkronX86.version()
elif fastkronCUDA != None:
  __version__ = fastkronCUDA.version()
else:
  __version__ = "1.0+shuffle"