import os
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11_stubgen
import shutil
from glob import glob
from pathlib import Path
import sys
import platform

__version__ = "0.0.1"

# get libraries
include_dirs = [Path(__file__).parent.parent.parent.joinpath('include')]
library_dirs = []
libraries = []
# set os specific arguments
if sys.platform == "linux":
    extra_compile_args = [
        "-O3", 
        "-DNDEBUG", 
        "-march=native",
        "-ffast-math",
        "-fopenmp"]
    extra_link_args = ["-fopenmp"]
    extra_objects = []
    libraries = []
elif sys.platform == "darwin":
    if platform.processor() == "arm":
        extra_compile_args = [
            "-O3", 
            "-DNDEBUG", 
            "-mcpu=apple-a14",
            "-ffast-math"]
        extra_link_args = []
        extra_objects = []
        libraries = []
    else:
        extra_compile_args = [
        "-O3", 
        "-DNDEBUG", 
        "-march=native",
        "-ffast-math",
        "-fopenmp"]
        # "-DEIGEN_DONT_PARALLELIZE"]
        extra_link_args = ["-fopenmp"]
        extra_objects = []
        libraries = []
elif sys.platform == "win32":
    extra_compile_args = [
        "/O2",
        "/arch:AVX"]
    extra_link_args = ["/LTCG"]
    extra_objects = []


name = 'mjp_inference'
source = sorted(glob('**/*.cpp', recursive=True))


ext_modules = [
    Pybind11Extension(
        name,
        source,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        extra_objects=extra_objects
    ),
]

setup(
    name = name,
    version = __version__,
    description = 'C++ package for forward simulation and inference of MJPs',
    cmdclass={"build_ext": build_ext},
    ext_modules = ext_modules,
    # setup_requires = ["pybind11==2.6.0"],
    # install_requires = ["pybind11==2.6.0"],
)
    

# get current and build directory
cur_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(cur_dir, 'build')

# copy to above directory
for dirpath, subdirs, filenames in os.walk(build_dir):
    for filename in filenames:
        if ('.so' in filename or '.pyd' in filename) and name in filename:
            file_path = os.path.join(dirpath, filename)
            target_path = os.path.join(cur_dir, filename)
            shutil.copyfile(file_path, target_path)

# create stubs
output_folder = Path(__file__).parent.joinpath('tmp')
args = [name, '-o', str(output_folder), '--no-setup-py']
pybind11_stubgen.main(args)
file_path = output_folder.joinpath(f'{name}-stubs', '__init__.pyi')
target_path = output_folder.parent.joinpath(f'{name}.pyi')
if target_path.exists():
    target_path.unlink()
file_path.rename(target_path)
shutil.rmtree(output_folder)