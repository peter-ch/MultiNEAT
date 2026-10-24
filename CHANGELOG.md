# Changelog

## 0.6.0

### Build and packaging

* Boost is no longer needed. The core is plain C++17 and the Python bindings use pybind11
  (Boost.Python, Boost.Serialization, Boost.Graph, Boost.Random, Boost.Variant... were replaced by
  the standard library and a small built-in serializer). This removes the source of nearly all
  installation problems (#22, #25, #32, #36, #43, #48, #53, #55, #57).
* `pip install .` builds everything (scikit-build-core + CMake). The package is ready for PyPI (#33),
  wheels are built with cibuildwheel.
* The outdated Cython bindings, `setup.py`, Travis/AppVeyor scripts, the MinGW CMake file and old IDE
  project files were removed. CI runs on GitHub Actions (Linux, Windows, macOS).
* The C++ library can be built and used without Python (`-DMULTINEAT_BUILD_PYTHON=OFF`);
  `src/Main.cpp` is a pure C++ XOR example and smoke test (#26).
* Compiles without warnings with MSVC `/W4` and GCC `-Wall -Wextra -Wpedantic`.
* `Assert.h` was renamed to `NeatAssert.h` - on case-insensitive file systems it shadowed `<assert.h>`.

### New features

* ES-HyperNEAT was rewritten (the old implementation was disabled and didn't compile). It works with
  substrates of any dimensionality - quadtree in 2D, octree in 3D, etc. `examples/TestESHyperNEAT_xor_3d.py`
  is now a real 3D example (#63).
* `MultiNEAT.physics2d` - a small 2D physics engine written in C++ (circles, thick segments, gravity,
  restitution, friction, collision callbacks). The examples no longer need pymunk.
* Novelty search examples that work out of the box (#20): `examples/NoveltySearch_simple.py` (no extra
  dependencies) and `examples/NoveltySearch.py` (deceptive maze).
* Python classes can derive from `PhenotypeBehavior` and override its methods.
* `NeuralNetwork.Input()` accepts any sequence - lists, tuples, numpy arrays of any dtype.
* Pickling now includes all traits, the trait definitions in `Parameters` and the full RNG state, so
  traits work in parallel evaluation (#41). `Substrate`, `Species`, `RNG` and `Population` can be pickled (#34).
* The old `Genome(ID, inputs, hidden, outputs, ...)` constructor works again, with `FS_NEAT_links`
  optional (#58, #40).
* Selection: tournament, truncation and roulette selection are all available and documented (#54).
* Parameters: `TournamentSelection`, `TournamentSize`, `EliteFraction`, `NeuronTries`,
  `ActivationFunction_Relu_Prob`, `ActivationFunction_Softplus_Prob` are exposed; the trait mutation
  probabilities are saved and loaded (#28, #30).

### Bug fixes

* `Parameters.Load()` froze on every file written by `Parameters.Save()` (#14).
* The genome, network, innovation database and population loaders looped forever (and could use all the
  memory) on malformed, truncated or missing files - they now fail with an error (#64, #66).
* Files are always written and read with `.` as the decimal separator, independent of the locale (#64).
* `NeuralNetwork.Load()` of ES-HyperNEAT networks failed because of uninitialized neuron and connection
  fields (#66).
* Mutation bookkeeping: `did_mutate` was overwritten in loops (#61); `Mutate_NeuronBiases` clamped the
  wrong field (#29).
* `Scale(vector, min, max)` ignored the target range; `GetMaxMin` was wrong for negative values (#60).
* Missing `#include`s (#25, #27).
* Epoch deleted the founding member of every new species, which led to crashes and an empty population
  (#40). New species now survive.
* Negative, NaN and infinite fitness values no longer trigger asserts or break selection (#26).
* Roulette selection could read out of bounds and crash (#54).
* `MutateGenome` and trait mutation could loop forever when no mutation was possible.
* Reproduction could hang for a very long time when the population had many clones (`AllowClones = False`);
  `ConstraintTrials` now defaults to 1000 instead of 2000000 (#38).
* `RemoveLinkGene` removed the wrong link (or all links) during cleanup and link removal.
* `Mate` picked the less fit parent's gene when it meant to prefer the fitter one, and ignored the
  averaging flag.
* FS-NEAT seed genomes could hang or create duplicate links.
* `Genome.CalculateDepth` was exponential (and could overflow the stack) on recurrent networks.
* The population size drifted above `PopulationSize`; top-up clones got duplicate IDs.
* Many smaller fixes: `GetLeader`/`GetBestGenome` with negative fitness, interspecies crossover in
  `Epoch`, delta coding choosing arbitrary species, novelty search sparseness out of bounds, Hebbian
  learning sign flip for negative weights, leaky integrator division by zero, RTRL in-place updates and
  out-of-bounds access, softplus overflow, `Rounded()` for negative values, modulo by zero for zero
  intervals, uninitialized members in several classes.
* `viz.py` works with current NumPy versions and doesn't print warnings on import.
