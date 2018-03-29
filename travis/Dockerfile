FROM gcc:4.8

MAINTAINER Anton Matosov "https://github.com/anton-matosov"

ADD ./travis /neat_build/travis

RUN apt-get update \
    && apt-get install -y --force-yes wget \
    && /neat_build/travis/install.sh \
    && rm -rf /var/lib/apt/lists/*


