import os
import shutil
import subprocess

#pyproject.toml
#[global]
#index-url = "https://download.pytorch.org/whl/cu121"

docker_create_container = "docker run -d -v $(pwd):/fastkron --name fastkron_build -it sameli/manylinux_2_28_x86_64_cuda_12.3:latest"
docker_kill_container = "docker kill fastkron_build"
docker_rm_container = "docker rm fastkron_build"
docker_exec = f"docker exec fastkron_build"
docker_remove_gcc_12 = f"{docker_exec} yum remove gcc-toolset-12* -y"
docker_install_gcc_11 = f"{docker_exec} yum install gcc-toolset-11* -y"
docker_install_git = f"{docker_exec} yum install git -y"
docker_git_add_safe_dir = f"{docker_exec} git config --global --add safe.directory /fastkron"
gcc_11_path = "PATH=/opt/rh/gcc-toolset-11/root/usr/bin:$PATH"

host_fk_dir = os.getcwd()
bdist_dir = "dist"
docker_fk_dir = "/fastkron/"
docker_packaging = os.path.join(docker_fk_dir, "packaging")
docker_bdist_dir = os.path.join(docker_fk_dir, bdist_dir)

def run_command(command):
  print("Running ", command, " in directory ", os.getcwd())
  (s, o) = subprocess.getstatusoutput(command)
  if s != 0:
    print (f"Error running {command}\n", o)
    assert False
  return s, o

def build_wheel(python_version):
  (s, o) = run_command(f"{docker_exec} sh {docker_packaging}/manylinux_docker_build.sh cp{python_version} {docker_fk_dir}")

def test_wheel(python_version):
  python_dir = f"/opt/python/cp{python_version}-cp{python_version}/bin/"
  pip = os.path.join(python_dir, "pip")
  python = os.path.join(python_dir, "python")
  for f in os.listdir(os.path.join(host_fk_dir, bdist_dir)):
    if f"cp{python_version}-manylinux_2_28_x86_64.whl" in f:
      (s, o) = run_command(f"{docker_exec} {pip} install {docker_bdist_dir}/{f}")

  (s, o) = run_command(f"{docker_exec} {python} {docker_fk_dir}/tests/python/test_wheels.py")

def audit_wheel(python_version):
  for f in os.listdir(os.path.join(host_fk_dir, bdist_dir)):
    if f"cp{python_version}-linux_x86_64.whl" in f:
      (s, o) = run_command(f"{docker_exec} auditwheel repair {docker_bdist_dir}/{f} -w {docker_bdist_dir}/")

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description = "Build Python Wheels")
  parser.add_argument('-python-version', required=True, type=str, nargs="+")

  args = parser.parse_args()
  if len(args.python_version) > 0:
    print("Create container")

    run_command(docker_create_container)
    run_command(docker_install_git)
    run_command(docker_git_add_safe_dir)

    print(f"Building for Python versions: {args.python_version}")
    for py in args.python_version:
      print(f"Building for Python {py}")
      build_wheel(py)

    print(f"Auditing wheels")
    for py in args.python_version:
      audit_wheel(py)

    print(f"Test wheels")
    for py in args.python_version:
      test_wheel(py)

    run_command(docker_kill_container)
    run_command(docker_rm_container)