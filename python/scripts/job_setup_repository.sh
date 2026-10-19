#!/bin/sh

set -eu

if [ "$#" -ne 2 ] || [ "$1" != "--directory" ] || [ -z "$2" ]; then
    printf "Usage: %s --directory PATH\n" "$0" >&2
    exit 2
fi

directory=$2

repository_url="https://github.com/gregrahn/join-order-benchmark.git"
repository="$directory/join-order-benchmark"

setup_repository() {
    clone_url=$1
    clone_path=$2

    if [ -d "$clone_path/.git" ]; then
        return 0
    fi

    if [ -z "${COLAB_RELEASE_TAG:-}" ] && [ -z "${GITHUB_ACTIONS:-}" ] && [ -z "${CODESPACES:-}" ] && [ -t 0 ]; then
        printf "clone %s to %s? (y/N) " "$clone_url" "$clone_path"
        read -r answer
        if [ "$answer" = y ] || [ "$answer" = Y ]; then
            :
        else
            exit 1
        fi
    fi

    git clone "$clone_url" "$clone_path"
}


if [ ! -d "$directory" ]; then
    mkdir -p "$directory"
fi

setup_repository "$repository_url" "$repository"
