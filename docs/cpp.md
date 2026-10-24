# Using the C++ library

The core of MultiNEAT is plain C++17 with no dependencies, and can be used without Python.

## Building

```bash
cmake -S . -B build -DMULTINEAT_BUILD_PYTHON=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release        # runs the XOR example as a test
cmake --install build --prefix /usr/local  # optional: lib/libmultineat.a and include/MultiNEAT/*.h
```

In your own CMake project the simplest way is to add MultiNEAT as a subdirectory:

```cmake
set(MULTINEAT_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(MULTINEAT_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(MultiNEAT)
target_link_libraries(my_experiment PRIVATE multineat)
```

## Example

`src/Main.cpp` is a complete example. The essentials:

```cpp
#include "Genome.h"
#include "Population.h"
#include "NeuralNetwork.h"
#include "Parameters.h"

using namespace NEAT;

Parameters params;
params.PopulationSize = 150;

GenomeInitStruct ints;
ints.NumInputs = 3;     // the last input is the bias
ints.NumOutputs = 1;
Genome seed(params, ints);

Population pop(seed, params, true, 1.0, /* RNG seed */ 0);

for (int generation = 0; generation < 300; generation++)
{
    for (auto &species : pop.m_Species)
    {
        for (auto &genome : species.m_Individuals)
        {
            NeuralNetwork net;
            genome.BuildPhenotype(net);
            std::vector<double> inputs = {1.0, 0.0, 1.0};
            net.Input(inputs);
            net.Activate();
            double fitness = /* ... */ net.Output()[0];
            genome.SetFitness(fitness);
            genome.SetEvaluated();
        }
    }
    pop.Epoch();
}
```

The classes and methods are the same as in Python (`Population`, `Species`, `Genome`, `NeuralNetwork`,
`Substrate`, `Parameters`, `RNG`, `Physics2D::Space` ...); the Python names map directly to the C++
ones, and the members exposed as Python attributes are public C++ members with an `m_` prefix
(`pop.Species` is `pop.m_Species`, `species.Individuals` is `species.m_Individuals`).

C++ specific features:

* `params.CustomConstraints` - a `bool (*)(Genome&)` function returning true if a genome fails your
  topology constraints.
* Novelty search with `PhenotypeBehavior` subclasses: `Population::InitPhenotypeBehaviorData()` and
  `Population::NoveltySearchTick()`.
* Serialization to strings: `SerializeToString(obj)` / `DeserializeFromString(str, obj)` from
  `Serialization.h` work for every class that can be pickled in Python.

Errors (bad files, invalid arguments) are reported with standard exceptions (`std::runtime_error`,
`std::invalid_argument`). Compile with `-DDEBUG` to enable internal consistency checks (`ASSERT`).

## Source layout

| File | |
|---|---|
| `Genome.*` | genomes, mutations, crossover, phenotype building, HyperNEAT |
| `ESHyperNEAT.cpp` | the ES-HyperNEAT phenotype builder |
| `Genes.h`, `Traits.h` | neuron/link genes and traits |
| `Population.*`, `Species.*` | the evolutionary algorithm |
| `Innovation.*` | the innovation database |
| `NeuralNetwork.*` | phenotypes, activation, learning |
| `Substrate.*` | HyperNEAT substrates |
| `Parameters.*` | all parameters, loading and saving |
| `Random.*` | the random number generator |
| `Serialization.h` | the archives used for pickling |
| `Physics2D.*` | the physics engine |
| `PythonBindings.cpp` | the pybind11 module (only compiled with `PYTHON_BINDINGS`) |
