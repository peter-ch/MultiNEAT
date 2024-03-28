#!/usr/bin/env bash
set -e
set -x

script_dir=$(dirname $0)

docker build "$script_dir/.."  -f "$script_dir/Dockerfile" -t multineat:4.8
