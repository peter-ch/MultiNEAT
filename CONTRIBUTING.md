# Contributing to MultiNEAT

Bug reports, fixes, examples and documentation improvements are welcome.

## Setting up

```bash
git clone https://github.com/peter-ch/MultiNEAT.git
cd MultiNEAT
pip install pybind11 pytest numpy
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
```

This builds the extension into `MultiNEAT/` (so `import MultiNEAT` from the repository root uses it),
the static C++ library and the `multineat_xor` example. Rebuild after every C++ change. Alternatively
`pip install -e .` installs an editable package.

On Windows, if `CMAKE_TOOLCHAIN_FILE` points to vcpkg, add `-DVCPKG_MANIFEST_MODE=OFF` (or
`-DCMAKE_TOOLCHAIN_FILE=`).

## Tests

```bash
python -m pytest tests              # Python tests (from the repository root, with the in-place build)
ctest --test-dir build -C Release   # C++ smoke test
```

Every bug fix should come with a test in `tests/`. Tests must be fast (the whole suite runs in seconds)
and deterministic (use fixed seeds).

To check the internal consistency checks, configure with `-DCMAKE_CXX_FLAGS=-DDEBUG` - this turns on
the `ASSERT`s in the C++ code.

## Code layout

* `src/` - the C++ library (see [docs/cpp.md](docs/cpp.md#source-layout)) and `PythonBindings.cpp`
* `MultiNEAT/` - the Python package: helpers (`__init__.py`), visualization (`viz.py`), the physics
  engine wrapper (`physics2d.py`)
* `tests/` - pytest tests
* `examples/` - example experiments
* `docs/` - the documentation (MkDocs)

## Guidelines

* The C++ code must stay free of external dependencies and compile without warnings with
  MSVC (`/W4`) and GCC/Clang (`-Wall -Wextra`).
* Python-specific code in the core files goes inside `#ifdef PYTHON_BINDINGS`.
* New members that should survive pickling must be added to the class' `serialize()` method.
* Follow the style of the surrounding code (`m_` member prefix, `a_` argument prefix, `t_` local
  prefix in the core).
* Keep the public API backwards compatible where possible.

## Documentation

The documentation is in `docs/` and is built with [MkDocs](https://www.mkdocs.org/):

```bash
pip install "mkdocs==1.6.1" "mkdocs-material==9.8.0"
mkdocs serve        # preview at http://127.0.0.1:8000
```

Two pages are generated from the code - regenerate them after changing parameters or bindings:

```bash
python docs/tools/gen_parameters.py   # docs/parameters.md from src/Parameters.h
python docs/tools/gen_api.py          # docs/api.md from the compiled module
```

Parameter descriptions come from the comments above the members in `src/Parameters.h`, method
descriptions from the docstrings in `src/PythonBindings.cpp`.

The documentation is published to GitHub Pages by `.github/workflows/docs.yml`.

## Releases

1. Update the version in `pyproject.toml`, `conda/meta.yaml` and `MultiNEAT/__init__.py`.
2. Add the changes to `CHANGELOG.md`.
3. Tag the release (`git tag v0.6.0 && git push --tags`) - CI builds the wheels and the source
   distribution as artifacts, ready to be uploaded to PyPI.
