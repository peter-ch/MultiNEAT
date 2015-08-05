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
          version='0.3',
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
                                  extra_compile_args=['-O3', '-march=native', #'/EHsc', # for Windows
                                                      '-std=gnu++11',
                                                      '-g',
                                                      '-Wall'
                 ])
],
                                    ))

except Exception as ex:
    print('Cython is not present, trying boost::python (with boost::random and boost::serialization)')

    setup(name='MultiNEAT',
          version='0.3',
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
                                            
                                 # for Windows                                 
                                 #libraries=[],
                                 #include_dirs = ['C:/Users/Peter/Desktop/boost_1_58_0'],
                                 #library_dirs = ['C:/Users/Peter/Desktop/boost_1_58_0/stage/lib'],

                                 extra_compile_args=['-O3', '-march=native', #'/EHsc', # for Windows
                                                     '-DUSE_BOOST_PYTHON',
                                                     '-DUSE_BOOST_RANDOM', 
                                                    '-std=gnu++11',
                                                    '-g',
                                                    '-Wall'
                                                     ])
                       ])
