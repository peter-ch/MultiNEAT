#!/usr/bin/python
from __future__ import print_function
from distutils.core import setup, Extension
from distutils.core import setup
import sys

if sys.version_info[0] < 3:
    lb = 'boost_python'
else:
    lb = 'boost_python3'

try:
    from Cython.Build import cythonize

    # easy way to turn Boost.Python on/off
    1/0

    setup(name='MultiNEAT',
          version='0.2',
          py_modules=['MultiNEAT'],
          ext_modules = cythonize([Extension('_MultiNEAT',
                                             ['_MultiNEAT.pyx',
                                              'src/Genome.cpp',
                                              'src/Innovation.cpp',
                                              'src/NeuralNetwork.cpp',
                                              'src/Parameters.cpp',
                                              'src/PhenotypeBehavior.cpp',
                                              'src/Population.cpp',
                                              'src/Random.cpp',
                                              'src/Species.cpp',
                                              'src/Substrate.cpp',
                                              'src/Utils.cpp'],
                                     extra_compile_args=['-O3', '-march=native'])],
                                    ))

except Exception as ex:
    print('Cython is not present, trying boost::python (with boost::random and boost::serialization)')

    setup(name='MultiNEAT',
          version='0.2',
          py_modules=['MultiNEAT'],
          ext_modules=[Extension('_MultiNEAT', ['src/Genome.cpp',
                                                'src/Innovation.cpp',
                                                'src/NeuralNetwork.cpp',
                                                'src/Parameters.cpp',
                                                'src/PhenotypeBehavior.cpp',
                                                'src/Population.cpp',
                                                'src/PythonBindings.cpp',
                                                'src/Random.cpp',
                                                'src/Species.cpp',
                                                'src/Substrate.cpp',
                                                'src/Utils.cpp'],
                                 libraries=[lb,
                                            'boost_serialization'],
                                 extra_compile_args=['-O3', '-march=native',
                                                     '-DUSE_BOOST_PYTHON',
                                                     '-DUSE_BOOST_RANDOM', 
                                                    '-std=gnu++11'
                                                     ])
                       ])
