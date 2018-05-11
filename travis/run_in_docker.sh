#!/usr/bin/env bash
set -e
set -x

docker run \
  -it \
  --rm \
  -v $PWD:/neat \
  -e CONDA_UPLOAD \
  -e CONDA_LOGIN_USERNAME \
  -e CONDA_LOGIN_PASSWORD  \
  -e TRAVIS_OS_NAME \
  -e TRAVIS_BUILD_NUMBER \
  -e CONDA_PY \
  multineat:4.8 \
  /bin/bash -c "/neat/travis/run.sh"
