# Installation

## Python

Requirements:

* Python 3.8 or newer
* a C++17 compiler - Visual Studio 2019 or newer on Windows (the "Desktop development with C++"
  workload), GCC 7+ or Clang 5+ on Linux, Xcode command line tools on macOS

Everything else needed for the build (CMake, pybind11, scikit-build-core) is downloaded by pip.
Nothing else, in particular no Boost, is needed.

Install the latest version from GitHub:

```bash
pip install git+https://github.com/peter-ch/MultiNEAT.git
```

or from a local clone:

```bash
git clone https://github.com/peter-ch/MultiNEAT.git
cd MultiNEAT
pip install .
```

Check the installation:

```bash
python -c "import MultiNEAT; print(MultiNEAT.__version__)"
```

### Optional packages

| Package | Used for |
|---|---|
| `numpy` | passing numpy arrays to `NeuralNetwork.Input()`, visualization |
| `opencv-python` | `DrawPhenotype` / `Draw` network images |
| `matplotlib` | `plot_nn` |
| `networkx` | `Genome2NX` - genomes as networkx graphs |
| `pygame` | watching the physics examples |
| `progressbar2` | progress bars in `EvaluateGenomeList_Serial` |
| `ipyparallel` | `EvaluateGenomeList_Parallel` on an IPython cluster |

```bash
pip install "MultiNEAT[viz,graph] @ git+https://github.com/peter-ch/MultiNEAT.git"
```

## Development setup

To work on MultiNEAT itself, build the extension in place so `import MultiNEAT` from the repository
root uses your build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
```

This builds the Python module into `MultiNEAT/` (next to `__init__.py`), the static C++ library and
the `multineat_xor` example. CMake finds pybind11 either as a CMake package or in the active Python
environment (`pip install pybind11`). Pick the Python with `-DPython_EXECUTABLE=/path/to/python`.

Then run the tests (see [CONTRIBUTING](https://github.com/peter-ch/MultiNEAT/blob/master/CONTRIBUTING.md)):

```bash
python -m pytest tests
```

### CMake options

| Option | Default | |
|---|---|---|
| `MULTINEAT_BUILD_PYTHON` | ON | build the Python extension module |
| `MULTINEAT_BUILD_CPP_LIBRARY` | ON | build the static C++ library `multineat` |
| `MULTINEAT_BUILD_EXAMPLES` | ON | build `multineat_xor` and register it as a test |

## Conda and Docker

`conda/meta.yaml` is a conda-build recipe (`conda build conda`) and the `Dockerfile` builds an image with
MultiNEAT installed and tested:

```bash
docker build -t multineat .
docker run --rm multineat
```

## Troubleshooting

* **`error: Microsoft Visual C++ 14.0 or greater is required`** - install Visual Studio or the
  "Build Tools for Visual Studio" with the C++ workload.
* **CMake hangs at "Configuring" on Windows** - if `CMAKE_TOOLCHAIN_FILE` points to vcpkg in your
  environment and there is a `vcpkg.json` in the folder, vcpkg starts installing packages. Unset the
  variable, or pass `-DVCPKG_MANIFEST_MODE=OFF` (the pip build already does that).
* **`ImportError: cannot import name '_MultiNEAT'`** - you are importing the source folder instead of
  the installed package. Run Python from another directory, or build in place as described above.
