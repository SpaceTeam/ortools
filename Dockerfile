# pull base image
FROM --platform=linux/amd64 ubuntu:jammy

# Install base utilities
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev wget \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 --no-cache-dir install --upgrade pip \
  && rm -rf /var/lib/apt/lists/*

# install prerequesites for ortools
RUN apt-get update && \
    apt-get -y install --no-install-recommends \
    openjdk-11-jre \
    git \
    && \
    : "remove cache" && \
    apt-get autoremove -y -qq && \
    rm -rf /var/lib/apt/lists/*

# copy code
COPY . /ortools
WORKDIR /ortools

# install orhelper fork
RUN mkdir /projects && cd /projects && \
    git clone https://github.com/SpaceTeam/orhelper.git && \
    cd orhelper && \
    pip install -e .

# install ortools
RUN pip install -e .

# get OR
RUN wget https://github.com/openrocket/openrocket/releases/download/release-22.02/OpenRocket-22.02.jar

# default command
CMD diana -c examples/dispersion_analysis.ini