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
  multineat:latest \
  /bin/bash -c "/neat/travis/run.sh"
