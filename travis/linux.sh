#!/usr/bin/env bash
set -e
set -x

./travis/build_docker.sh
./travis/run_in_docker.sh