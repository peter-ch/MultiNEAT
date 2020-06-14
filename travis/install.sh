#!/usr/bin/env bash
set -e
set -x

if [[ "$(uname -s)" == 'Darwin' ]]; then
  if [[ "$TRAVIS_PYTHON_VERSION" == "2.7" ]]; then
    MINICONDA_URL=https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh
  else
    MINICONDA_URL=https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
  fi

  brew install wget || true
else 
  if [[ "$TRAVIS_PYTHON_VERSION" == "2.7" ]]; then
    MINICONDA_URL=https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
  else
    MINICONDA_URL=https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
  fi
fi

wget $MINICONDA_URL -O miniconda.sh;

bash miniconda.sh -b -p $HOME/miniconda

source $HOME/miniconda/etc/profile.d/conda.sh

conda config --set always_yes yes --set changeps1 no

conda update -q conda

# Useful for debugging any issues with conda
conda info -a

conda create -q -n test-environment python=3.6 anaconda-client conda-build
conda activate test-environment

