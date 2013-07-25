# What is it?

MultiNEAT is a portable software library for performing neuroevolution, a form of machine learning that trains neural networks with a genetic algorithm. It is based on NEAT, an advanced method for evolving neural networks through complexification. The neural networks in NEAT begin evolution with very simple genomes which grow over successive generations. The individuals in the evolving population are grouped by similarity into species, and each of them can compete only with the individuals in the same species. The combined effect of speciation, starting from the simplest initial structure and the correct matching of the genomes through marking genes with historical markings yields an algorithm which is proven to be very effective in many domains and benchmarks against other methods. NEAT was developed around 2002 by Kenneth Stanley in the University of Texas at Austin.

Please see the website for information:
http://multineat.com

# Build Instructions

    mkdir build
    cd build
    cmake ..
    make

## Dependencies

- cmake
- boost-random
- boost-serialization

## CMake parameters

- `PYTHON_BINDING` : ON/OFF

  Build the Python binding  
  Dependencies : python, distutils, boost-python

- `NODE_BINDING` : ON/OFF

  Build the Node binding  
  Dependencies : node, node-gyp

  - `NODE_GYP_EXECUTABLE` : `node-gyp` by default
  - `NODE_GYP_TARGET` : The target version of Node

## Examples

- Verbose make

    cmake .. -CMAKE_VERBOSE_MAKEFILE=ON

- [node-webkit build](https://github.com/rogerwang/node-webkit)

    cmake .. -DCMAKE_BUILD_TYPE=Release -DPYTHON_BINDINGS=OFF -DNODE_BINDINGS=ON -DCMAKE_OSX_ARCHITECTURES=i386 -DNODE_GYP_EXECUTABLE=nw-gyp -DNODE_GYP_TARGET=0.6.2

# Licence

[GNU Lesser General Public License v3.0](http://www.gnu.org/licenses/lgpl.html)
