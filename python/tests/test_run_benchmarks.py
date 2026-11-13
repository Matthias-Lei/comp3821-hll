"""Tests for run_benchmarks.py. A tiny shell script stands in for the real benchmark program."""
import json
import sys
from pathlib import Path

import pytest

from run_benchmarks import BASE_DIR, benchmark_command, display_path, find_streams, run_benchmark, save_json

needs_posix = pytest.mark.skipif(sys.platform == "win32", reason="uses a shell script as a fake benchmark")


def fake_benchmark(directory: Path, script: str) -> Path:
    """Writes an executable shell script that pretends to be the benchmark."""
    path = directory / "benchmarks"
    path.write_text("#!/bin/sh\n" + script)
    path.chmod(0o755)
    return path


def test_find_streams_lists_cebs_files_in_order(tmp_path: Path) -> None:
    for name in ["b.cebs", "a.cebs", "notes.txt"]:
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "dir.cebs").mkdir()
    assert [path.name for path in find_streams(tmp_path)] == ["a.cebs", "b.cebs"]


def test_benchmark_command() -> None:
    assert benchmark_command(Path("b"), Path("s.cebs"), None) == ["b", "s.cebs", "--benchmark_format=json"]
    assert benchmark_command(Path("b"), Path("s.cebs"), "BM_LinearCounting")[-1] == "--benchmark_filter=BM_LinearCounting"


def test_display_path() -> None:
    assert display_path(BASE_DIR / "build" / "benchmarks") == "./build/benchmarks"
    outside = BASE_DIR.parent / "elsewhere" / "benchmarks"
    assert display_path(outside) == outside.resolve().as_posix()


@needs_posix
def test_run_benchmark_returns_json_with_command(tmp_path: Path) -> None:
    binary = fake_benchmark(tmp_path, """echo '{"benchmarks": [{"name": "BM_X/bits:8/seed:1"}]}'\n""")
    output = run_benchmark(binary, tmp_path / "s.cebs", "BM_X")
    assert output["benchmarks"][0]["name"] == "BM_X/bits:8/seed:1"
    stream = (tmp_path / "s.cebs").resolve().as_posix()
    assert output["command"] == f"{binary.resolve().as_posix()} {stream} --benchmark_format=json --benchmark_filter=BM_X"


@needs_posix
def test_run_benchmark_skips_a_failed_stream(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    binary = fake_benchmark(tmp_path, "echo 'CEBS: bad magic' >&2\nexit 1\n")
    assert run_benchmark(binary, tmp_path / "s.cebs", None) is None
    assert "skipped s.cebs: CEBS: bad magic" in capsys.readouterr().err


@needs_posix
def test_run_benchmark_skips_output_that_is_not_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    binary = fake_benchmark(tmp_path, "echo 'not json'\n")
    assert run_benchmark(binary, tmp_path / "s.cebs", None) is None
    assert "skipped s.cebs: output is not JSON" in capsys.readouterr().err


def test_save_json_creates_the_directory(tmp_path: Path) -> None:
    path = tmp_path / "results" / "s.json"
    save_json({"benchmarks": []}, path)
    assert json.loads(path.read_text()) == {"benchmarks": []}
