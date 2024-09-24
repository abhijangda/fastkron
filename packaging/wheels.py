import os
import shutil
import subprocess

#pyproject.toml
#[global]
#index-url = "https://download.pytorch.org/whl/cu121"

docker_create_container = "docker run -d -v $(pwd):/fastkron --name fastkron_build -it sameli/manylinux_2_28_x86_64_cuda_12.3:latest"
docker_kill_container = "docker kill fastkron_build"
docker_rm_container = "docker rm fastkron_build"
docker_exec = f"docker exec -it fastkron_build"

def run_command(command):
  print("Running ", command, " in directory ", os.getcwd())
  (s, o) = subprocess.getstatusoutput(command)
  if s != 0:
    print (f"Error running {command}\n", o)
    assert False
  return s, o

def build_wheel(python_version):
  docker_fk_dir = "/fastkron/"
  host_fk_dir = os.getcwd()
  docker_packaging = os.path.join(docker_fk_dir, "packaging")
  bdist_dir = "dist"
  docker_bdist_dir = os.path.join(docker_fk_dir, bdist_dir)

  (s, o) = run_command(f"{docker_exec} sh {docker_packaging}/manylinux_docker_build.sh cp{python_version} {docker_fk_dir}")
  (s, o) = run_command(f"{docker_exec} auditwheel repair {docker_bdist_dir}/*whl -w {docker_bdist_dir}/")

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description = "Build Python Wheels")
  parser.add_argument('-python-version', required=True, type=str, nargs="+")

  args = parser.parse_args()
  if len(args.python_version) > 0:
    print("Create container")

    run_command(docker_create_container)

    print(f"Building for Python versions: {args.python_version}")
    for py in args.python_version:
      print(f"Building for Python {py}")
      build_wheel(py)
    
    run_command(docker_kill_container)
    run_command(docker_rm_container)