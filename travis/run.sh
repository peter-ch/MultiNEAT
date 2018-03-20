#!/usr/bin/env bash
set -e
set -x

source $HOME/miniconda/etc/profile.d/conda.sh
conda activate test-environment

anaconda login --username $CONDA_LOGIN_USERNAME --password $CONDA_LOGIN_PASSWORD
conda config --set anaconda_upload ${CONDA_UPLOAD:-no}

conda build conda/
