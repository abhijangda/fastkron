import os
import shutil
import subprocess

#pyproject.toml
#[global]
#index-url = "https://download.pytorch.org/whl/cu121"

def run_command(command):
  print("Running ", command, " in directory ", os.getcwd())
  (s, o) = subprocess.getstatusoutput(command)
  if s != 0:
    print (f"Error running {command}\n", o)
    assert False
  return s, o

def build_wheel(backends, cuda_versions, torch_version):
  if 'cuda' in backends:
    for ver in cuda_versions:
      (s, _) = run_command(f"python3 -m build --wheel")
      if s == 0:
        fs = os.listdir("dist/")
        for f in fs:
          if '.whl' in f:
            split = f.split('-')
            split[1] = split[1]+"+cu"+ver+"torch"+torch_version
            shutil.move(f"dist/{f}", f"dist/{'-'.join(split)}")
  else:
    (s, _) = run_command(f"python3 -m build --wheel -C cmake.define.ENABLE_CUDA=OFF")
    if s == 0:
        fs = os.listdir("dist/")
        for f in fs:
          if '.whl' in f:
            split = f.split('-')
            split[1] = split[1]+"torch"+torch_version
            shutil.move(f"dist/{f}", f"dist/{'-'.join(split)}")

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description = "Build Python Wheels")
  parser.add_argument('-backends', required=True, type=str, nargs="+")
  parser.add_argument('-cuda-vers', required=False, type=str, nargs="+")
  parser.add_argument('-torch-ver', required=True, type=str)

  args = parser.parse_args()

  if args.backends is not None:
    for backend in args.backends:
      assert backend in ['x86', 'cuda']
  
  if 'cuda' in args.backends:
    assert args.cuda_vers is not None, "Specify CUDA Versions"

  if args.cuda_vers is not None:
    for ver in args.cuda_vers:
      assert ver in ['118', '122', '124']
  
  build_wheel(args.backends, args.cuda_vers, args.torch_ver)