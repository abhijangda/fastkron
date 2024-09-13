import platform

if platform.system() == "Linux" and platform.processor() == "x86_64":
  from . import FastKron

  __version__ = FastKron.version()
else:
  __version__ = "1.0+shuffle"