#!/usr/bin/env bash

set -x
set -e

TESTS="examples/TestTraits.py examples/NoveltySearch.py examples/TestNEAT_xor.py examples/TestHyperNEAT_xor.py"

echo $TESTS | xargs -n 1 -P 4 python

