#!/bin/sh

set -eu

if [ "$#" -ne 2 ] || [ "$1" != "--directory" ] || [ -z "$2" ]; then
    printf "Usage: %s --directory PATH\n" "$0" >&2
    exit 2
fi

directory=$2

verify_sha384() {
    file_path=$1
    expected_hash=$2

    if [ -z "$expected_hash" ]; then
        return 0
    fi

    actual_hash=$(sha384sum "$file_path" | cut -d" " -f1)
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

    curl -L "$download_url" -o "$download_path"
    verify_sha384 "$download_path" "$expected_hash"
}

data_url="https://database.lichess.org/lichess_db_puzzle.csv.zst"

dataset_directory="$directory/lichess"

mkdir -p "$dataset_directory"

download "$data_url" "$dataset_directory/puzzles.csv.zst"

if [ ! -f "$dataset_directory/puzzles.csv" ]; then
    sudo apt-get update
    sudo apt-get install -y zstd

    zstd -dc "$dataset_directory/puzzles.csv.zst" > "$dataset_directory/puzzles.csv"
fi
