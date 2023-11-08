from distutils.core import setup, Extension
import distutils
distutils.log.set_verbosity(0)
setup(name='fastkroncpp', version='1.0', \
   ext_modules=[Extension('fastkroncpp', ['src/pymodule.cpp'], include_dirs=["/usr/local/cuda/include/"], 
                          library_dirs=["/home/saemal/KroneckerGPU"], libraries = ["Kron"])])

