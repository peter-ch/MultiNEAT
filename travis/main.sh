#!/usr/bin/env bash
set -e
set -x

if [[ "$(uname -s)" == 'Darwin' ]]; then
  ./travis/install.sh
  ./travis/run.sh
else 
  ./travis/build_docker.sh
  ./travis/run_in_docker.sh
fi