#!/usr/bin/env bash
set -e
set -x

conda config --set always_yes yes --set changeps1 no

conda update -q conda
source /opt/conda/etc/profile.d/conda.sh

conda info -a

conda create -q -n test-environment python=3.6 anaconda-client conda-build
conda activate test-environment