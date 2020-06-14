#!/usr/bin/env bash
set -e
set -x

script_dir=$(dirname $0)

pushd .
cd "$script_dir/.."

source $HOME/miniconda/etc/profile.d/conda.sh
conda activate test-environment

yes | anaconda login --username $CONDA_LOGIN_USERNAME --password $CONDA_LOGIN_PASSWORD \
    --hostname MultiNEAT--TRAVIS-${TRAVIS_OS_NAME}--BUILD-${TRAVIS_BUILD_NUMBER}--PYTHON-${CONDA_PY}
conda config --set anaconda_upload ${CONDA_UPLOAD:-no}

conda build conda/ -c defaults -c conda-forge

popd
