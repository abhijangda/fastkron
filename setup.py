from pathlib import Path
import shutil
import subprocess
import os

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
   def __init__(self) -> None:
      super().__init__("FastKronNative", sources=[])

class CMakeBuild(build_ext):
   def build_extension(self, ext: CMakeExtension) -> None:
      self.build_dir = Path("build/").resolve()
      if self.build_dir.exists():
         shutil.rmtree(self.build_dir)
      self.build_dir.mkdir()
      # os.chdir(self.build_dir)
      subprocess.run(
         ["cmake", ".."], cwd=self.build_dir, check=True
      )
      subprocess.run(
         ["make", "FastKron", "-j"], cwd=self.build_dir, check=True
      )
      # subprocess.getstatusoutput("cmake ..")
      # subprocess.getstatusoutput("make FastKron")
      # os.chdir("../")
      
setup(name='fastkroncpp', version='1.0', \
      ext_modules=[CMakeExtension()],
      cmdclass={"build_ext": CMakeBuild})