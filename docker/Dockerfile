FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

WORKDIR /workspace

RUN apt update \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install -y zsh tmux wget git cmake \
    g++ gcc zlib1g-dev libsndfile1 build-essential zlib1g-dev \
    libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev \
    libffi-dev curl libbz2-dev pkg-config make \
    python3-dev python3-pip\
    && apt clean