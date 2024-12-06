import os
import pathlib
import sys

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as build_ext_orig


class CMakeExtension(Extension):

    def __init__(self, name):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])


class build_ext(build_ext_orig):

    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.parent.mkdir(parents=True, exist_ok=True)
        # example of cmake args
        config = 'Debug' if self.debug else 'Release'
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(extdir.parent.absolute()),
            '-DCMAKE_BUILD_TYPE=' + config,
            '-DPYMODULE=ON',
        ]
        if 'X86' in ext.name:
          cmake_args += ['-DENABLE_CUDA=OFF', '-DENABLE_X86=ON']
        elif 'CUDA' in ext.name:
          cmake_args += ['-DENABLE_CUDA=ON', '-DENABLE_X86=OFF']

        cmake_args += [f'-DPYTHON_EXECUTABLE={sys.executable}']

        # example of build args
        build_args = [
            '--config ' + config,
            '-j'
        ]
        os.chdir(str(build_temp))
        self.spawn(['cmake', str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.'] + build_args)
        # Troubleshooting: if fail on line above then delete all possible
        # temporary CMake files including "CMakeCache.txt" in top level dir.
        os.chdir(str(cwd))

def find_version(*file_paths):
    try:
        with io.open(os.path.join(os.path.dirname(__file__), *file_paths), encoding="utf8") as fp:
            version_file = fp.read()
        version_match = re.search(r"^__version__ = version = ['\"]([^'\"]*)['\"]", version_file, re.M)
        return version_match.group(1)
    except Exception:
        return None


if "BUILD_ANY_WHEEL" in os.environ:
    setup(
        packages=['pyfastkron'],
        version=find_version("pyfastkron", "version.py"),
    )
else:
    setup(
        packages=['pyfastkron'],
        ext_modules=[CMakeExtension('pyfastkron.FastKronX86'),
                     CMakeExtension('pyfastkron.FastKronCUDA')],
        version=find_version("pyfastkron", "version.py"),
        cmdclass={
            'build_ext': build_ext,
        }
    )
