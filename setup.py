"""
Setup script for Quantitative Systems Simulator (QSS).

This package provides Monte Carlo simulation and portfolio risk analysis tools.
"""

import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """CMake extension for building C++ code."""

    def __init__(self, name: str, sourcedir: str = ""):
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    """Custom build extension for CMake-based builds."""

    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # CMake configuration
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE=Release",
        ]

        build_args = ["--config", "Release"]

        # Parallel build
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            try:
                import multiprocessing
                build_args += ["-j", str(multiprocessing.cpu_count())]
            except (ImportError, NotImplementedError):
                pass

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        # Run CMake
        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args],
            cwd=build_temp,
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args],
            cwd=build_temp,
            check=True,
        )


# Read long description from README
long_description = ""
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")


setup(
    name="quantitative-systems-simulator",
    version="0.1.0",
    author="Raja Hussain",
    author_email="",
    description="Portfolio Risk, Monte Carlo Simulation, and Statistical Analytics Engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quantitative-systems-simulator",

    # Package configuration
    packages=find_packages(where="python"),
    package_dir={"": "python"},

    # C++ extension
    ext_modules=[CMakeExtension("qss_core", sourcedir="cpp_core")],
    cmdclass={"build_ext": CMakeBuild},

    # Python requirements
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "plotly>=5.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "mypy>=0.900",
            "flake8>=4.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "full": [
            "statsmodels>=0.13.0",
            "seaborn>=0.11.0",
            "jupyter>=1.0.0",
            "openpyxl>=3.0.0",
        ],
    },

    # Metadata
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    keywords="monte-carlo simulation portfolio risk var finance quantitative",

    # Entry points
    entry_points={
        "console_scripts": [
            "qss=qss.cli:main",
        ],
    },

    # Include package data
    include_package_data=True,
    zip_safe=False,
)
