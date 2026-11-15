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

data_url="https://data-openjustice.doj.ca.gov/sites/default/files/dataset/2023-03/RIPA-Stop-Data-2021.zip"
expected_sha384="b5fe294e5e766a91b9e4e5270e5cf3406604f3d50d24f93a945a73f7fc8c3e8aeb0233de4a57a1c1d67b5ea75310a8a3"

dataset_directory="$directory/policing"

mkdir -p "$dataset_directory"

download "$data_url" "$dataset_directory/ripa_2021.zip" "$expected_sha384"

if [ ! -d "$dataset_directory/csv" ]; then
    sudo apt-get update
    sudo apt-get install -y unzip xlsx2csv

    mkdir -p "$dataset_directory/xlsx" "$dataset_directory/csv"
    unzip -q "$dataset_directory/ripa_2021.zip" -d "$dataset_directory/xlsx"

    find "$dataset_directory/xlsx" -name "*.xlsx" | while read -r workbook; do
        name=$(basename "$workbook" .xlsx | sed 's/^RIPA Stop Data _ //; s/ /_/g; s/,//g')
        PYTHONWARNINGS="ignore::DeprecationWarning" xlsx2csv "$workbook" "$dataset_directory/csv/$name.csv"
        printf "converted %s to %s.csv\n" "$workbook" "$name"
    done

    rm -rf "$dataset_directory/xlsx"
fi
