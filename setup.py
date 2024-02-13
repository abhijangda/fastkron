from pathlib import Path
import shutil
import subprocess
import os

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
   def __init__(self) -> None:
      super().__init__("libFastKron", sources=[])

class CMakeBuild(build_ext):
   def build_extension(self, ext: CMakeExtension) -> None:
      self.build_dir = Path("build/").resolve()
      ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)

      if self.build_dir.exists():
         shutil.rmtree(self.build_dir)
      self.build_dir.mkdir()
      # os.chdir(self.build_dir)
      subprocess.run(["cmake", ".."],            cwd=self.build_dir, check=True)
      subprocess.run(["make", "FastKron", "-j"], cwd=self.build_dir, check=True)
      shutil.copy(Path.cwd() / "build/libFastKron.so", ext_fullpath)
      
setup(name='pyfastkron', version='1.0')
      # ext_modules=[CMakeExtension()],
      # cmdclass={"build_ext": CMakeBuild})