#!/usr/bin/env bash
set -e
set -x

docker run -it --rm -v $PWD:/neat multineat:latest \
  -e CONDA_UPLOAD \
  -e CONDA_LOGIN_USERNAME \
  -e CONDA_LOGIN_PASSWORD  \
  /bin/bash -c "/neat/travis/run.sh"
