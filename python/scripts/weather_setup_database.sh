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

base_url="https://noaa-ghcn-pds.s3.amazonaws.com"

dataset_directory="$directory/weather"

mkdir -p "$dataset_directory"

download "$base_url/ghcnd-stations.txt" "$dataset_directory/stations.txt"
download "$base_url/ghcnd-inventory.txt" "$dataset_directory/inventory.txt"

awk '$1 ~ /^US/ && $4 == "TMAX" && $6 >= 2020 {print $1}' "$dataset_directory/inventory.txt" | head -n 50 > "$dataset_directory/tmax_stations.txt"

while read -r station; do
    download "$base_url/csv/by_station/$station.csv" "$dataset_directory/$station.csv"
done < "$dataset_directory/tmax_stations.txt"
