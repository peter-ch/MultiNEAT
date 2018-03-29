FROM condaforge/linux-anvil:latest

MAINTAINER Anton Matosov "https://github.com/anton-matosov"

ADD ./travis /neat_build/travis

# RUN apt-get update \
#     && apt-get install -y --force-yes wget \
#     && /neat_build/travis/install.sh \
#     && rm -rf /var/lib/apt/lists/*

RUN export PATH="/opt/conda/bin:${PATH}" \
    && conda update -q conda \
    && conda config --set always_yes yes --set changeps1 no \
    && conda create -q -n test-environment python=3.6 anaconda-client conda-build \
    && conda activate test-environment
