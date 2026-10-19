#!/bin/sh

set -eu

if [ "$#" -ne 2 ] || [ "$1" != "--directory" ] || [ -z "$2" ]; then
    printf "Usage: %s --directory PATH\n" "$0" >&2
    exit 2
fi

directory=$2

data_url="https://event.cwi.nl/da/job/imdb.tgz"
schema_url="https://raw.githubusercontent.com/gregrahn/join-order-benchmark/master/schema.sql"

expected_sha384="9552ac83b39de85e3240e5e26a4e1dd6d555585e54371ca0955e25c9d58dc624d3d3a3aef7bf9d49d730f0da7659ceef"

mkdir -p "$directory"

archive="$directory/imdb.tgz"
data_directory="$directory/imdb"
schema_path="$directory/schema.sql"

verify_sha384() {
    file_path=$1
    expected_hash=$2

    if [ -z "$expected_hash" ]; then
        return 0
    fi

    actual_hash=$(sha384sum "$file_path" | cut -d' ' -f1)
    if [ "$actual_hash" != "$expected_hash" ]; then
        printf "checksum mismatch, expected %s, saw %s for %s\n" "$expected_hash" "$actual_hash" "$file_path" >&2
        exit 1
    fi
}

download() {
    download_url=$1
    download_path=$2
    expected_hash=${3:-}

    if [ -f "$download_path" ]; then
        verify_sha384 "$download_path" "$expected_hash"
        return 0
    fi

    if [ -z "${COLAB_RELEASE_TAG:-}" ] && [ -z "${GITHUB_ACTIONS:-}" ] && [ -z "${CODESPACES:-}" ] && [ -t 0 ]; then
        printf "download %s to %s? (y/N) " "$download_url" "$download_path"
        read -r answer
        if [ "$answer" = y ] || [ "$answer" = Y ]; then
            :
        else
            exit 1
        fi
    fi

    curl "$download_url" -o "$download_path"
    verify_sha384 "$download_path" "$expected_hash"
}

download "$data_url" "$archive" "$expected_sha384"

if [ ! -d "$data_directory" ]; then
    mkdir -p "$data_directory"
    tar -xf "$archive" -C "$data_directory"
fi

download "$schema_url" "$schema_path"
