# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from setuptools import setup
from setuptools.command.build_ext import build_ext as build_ext_orig
from distutils.core import Extension
import os
import pathlib


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class build_ext(build_ext_orig):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        cmake_args = [
            "-S",
            "crs4/cpp",
            "-B",
            self.build_temp,
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + str(extdir.parent.absolute()),
        ]
        build_args = [
            "--build",
            self.build_temp,
        ]
        self.spawn(["cmake"] + cmake_args)
        self.spawn(["cmake"] + build_args)


setup(
    name="cassandra-dali-plugin",
    version="0.1",
    author="Francesco Versaci, Giovanni Busonera",
    author_email="francesco.versaci@gmail.com, giovanni.busonera@crs4.it",
    description="Cassandra data loader for ML pipelines",
    packages=["crs4/cassandra_utils"],
    url="https://github.com/crs4/cassandra-dali-plugin",
    ext_modules=[CMakeExtension("crs4cassandra")],
    cmdclass={
        "build_ext": build_ext,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
    ],
    install_requires=[
        "cassandra-driver",
        "tqdm",
    ],
    python_requires=">=3.6",
)
