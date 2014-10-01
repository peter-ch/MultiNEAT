

from distutils.core import setup, Extension

setup(name='MultiNEAT',
      version='0.1',
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
                             libraries=['boost_python',
                                        'boost_serialization'],
                             extra_compile_args=['-O3', '-march=native'])
                   ])
