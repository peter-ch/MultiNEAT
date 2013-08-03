

from distutils.core import setup, Extension

setup(name='MultiNEAT',
      version='0.1',
      py_modules=['MultiNEAT'],
      ext_modules=[Extension('_MultiNEAT', ['lib/Genome.cpp',
                                            'lib/Innovation.cpp',
                                            'lib/Main.cpp',
                                            'lib/NeuralNetwork.cpp',
                                            'lib/Parameters.cpp',
                                            'lib/PhenotypeBehavior.cpp',
                                            'lib/Population.cpp',
                                            'lib/PythonBindings.cpp',
                                            'lib/Random.cpp',
                                            'lib/Species.cpp',
                                            'lib/Substrate.cpp',
                                            'lib/Utils.cpp'],
                             libraries=['boost_python',
                                        'boost_serialization'])]
      )
