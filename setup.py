#!/usr/bin/python
from __future__ import print_function
from setuptools import setup, Extension
import sys
import os

''' Note:

to build Boost.Python on Windows with mingw

bjam target-os=windows/python=3.4 toolset=gcc variant=debug,release link=static,shared threading=multi runtime-link=shared cxxflags="-include cmath "


also insert this on top of boost/python.hpp :

#include <cmath>   //fix  cmath:1096:11: error: '::hypot' has not been declared

'''


def getExtensions():
    platform = sys.platform
    if sys.version_info[0] < 3:
        lb = 'boost_python'
    else:
        lb = 'boost_python3'  # in Ubuntu 14 there is only 'boost_python-py34'
    extensionsList = []
    sources = ['src/Genome.cpp',
               'src/Innovation.cpp',
               'src/NeuralNetwork.cpp',
               'src/Parameters.cpp',
               'src/PhenotypeBehavior.cpp',
               'src/Population.cpp',
               'src/Random.cpp',
               'src/Species.cpp',
               'src/Substrate.cpp',
               'src/Utils.cpp']

    extra = ['-march=native',
             '-g',
             '-Wall'
             ]

    if platform == 'darwin':
        extra += ['-stdlib=libc++',
             '-std=c++11',]
        os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'
    else:
        extra += ['-std=gnu++11']

    if 'win' in platform and platform != 'darwin':
        extra.append('/EHsc')
    else:
        extra.append('-w')

    prefix = os.getenv('PREFIX')
    if len(prefix) > 0:
        extra += ["-I{}/include".format(prefix)]

    build_sys = os.getenv('MN_BUILD')

    if build_sys is None:
        if os.path.exists('_MultiNEAT.cpp'):
            sources.insert(0, '_MultiNEAT.cpp')
            extra.append('-O3')
            extensionsList.extend([Extension('MultiNEAT._MultiNEAT',
                                             sources,
                                             extra_compile_args=extra)],
                                  )
        else:
            print('Source file is missing and MN_BUILD environment variable is not set.\n'
                  'Specify either \'cython\' or \'boost\'. Example to build in Linux with Cython:\n'
                  '\t$ export MN_BUILD=cython')
            exit(1)
    elif build_sys == 'cython':
        from Cython.Build import cythonize
        sources.insert(0, '_MultiNEAT.pyx')
        extra.append('-O3')
        extensionsList.extend(cythonize([Extension('MultiNEAT._MultiNEAT',
                                                   sources,
                                                   extra_compile_args=extra)],
                                        ))
    elif build_sys == 'boost':
        sources.insert(0, 'src/PythonBindings.cpp')
        libs = [lb, 'boost_system', 'boost_serialization']

        if 'win' in platform:
            libs = ["libboost_python3-vc140-mt-1_65_1",
                "libboost_numpy3-vc140-mt-1_65_1",
                "libboost_system-vc140-mt-1_65_1",
                "libboost_serialization-vc140-mt-1_65_1",
                ]

        # for Windows
        # libraries= ['libboost_python-mgw48-mt-1_58',
        #            'libboost_serialization-mgw48-mt-1_58'],
        # include_dirs = ['C:/MinGW/include', 'C:/Users/Peter/Desktop/boost_1_58_0'],
        # library_dirs = ['C:/MinGW/lib', 'C:/Users/Peter/Desktop/boost_1_58_0/stage/lib'],
        extra.extend(['-DUSE_BOOST_PYTHON', '-DUSE_BOOST_RANDOM'])
        extensionsList.append(Extension('MultiNEAT._MultiNEAT',
                                        sources,
                                        libraries=libs,
                                        extra_compile_args=extra)
                              )
    else:
        raise AttributeError('Unknown tool: {}'.format(build_sys))

    return extensionsList


setup(name='multineat',
      version='0.5', # Update version in conda/meta.yaml as well
      packages=['MultiNEAT'],
      ext_modules=getExtensions())
