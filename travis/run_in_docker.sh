#!/usr/bin/env bash
set -e
set -x

docker run -it --rm -v $PWD:/neat multineat:latest /bin/bash -c "/neat/travis/run.sh"
