import json
import socket

import pytest

from spark_supply_optimizer import cli
from spark_supply_optimizer.cli import (
    runtime_config,
    summarize_stage_reports,
    validate_s3_endpoint_reachable,
    write_json_report,
)
from spark_supply_optimizer.config import SparkRuntimeConfig


def test_write_json_report_creates_parent_directory(tmp_path) -> None:
    report_path = tmp_path / "reports" / "benchmark.json"

    write_json_report(report_path, {"metrics": {"seconds": 1.25}})

    assert json.loads(report_path.read_text(encoding="utf-8")) == {
        "metrics": {"seconds": 1.25}
    }


def test_summarize_stage_reports_computes_pipeline_totals(tmp_path) -> None:
    reports_dir = tmp_path / "reports"
    write_json_report(reports_dir / "prepare.json", {"duration_seconds": 10.0})
    write_json_report(reports_dir / "train.json", {"duration_seconds": 20.0})
    write_json_report(reports_dir / "predict.json", {"duration_seconds": 5.0})

    summary = summarize_stage_reports(reports_dir)

    assert summary["totals"]["prepare_train_seconds"] == 30.0
    assert summary["totals"]["prepare_train_predict_seconds"] == 35.0


def test_main_reports_missing_prepare_output_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeSpark:
        def stop(self) -> None:
            pass

    def raise_missing_features(*args, **kwargs) -> None:
        raise RuntimeError(
            "Required path does not exist: s3a://m5-data/artifacts/features. "
            "Run `m5-spark prepare` first."
        )

    monkeypatch.setattr(cli, "build_spark", lambda _: FakeSpark())
    monkeypatch.setattr(cli, "train_model", raise_missing_features)

    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "train",
                "--features-dir",
                "s3a://m5-data/artifacts/features",
                "--model-dir",
                "s3a://m5-data/artifacts/model",
            ]
        )

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "Required path does not exist: s3a://m5-data/artifacts/features" in captured.err
    assert "m5-spark prepare" in captured.err


def test_runtime_args_keep_all_cpu_but_use_safe_docker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPARK_MASTER", raising=False)
    monkeypatch.delenv("SPARK_DRIVER_MEMORY", raising=False)
    monkeypatch.setenv("SPARK_LOCAL_DIR", "/app/artifacts/spark-tmp")
    monkeypatch.setenv("SPARK_S3_ENDPOINT", "http://minio:9000")

    args = cli.build_parser().parse_args(["prepare"])
    config = runtime_config(args)

    assert config.master == "local[*]"
    assert config.driver_memory is None
    assert config.local_dir == "/app/artifacts/spark-tmp"
    assert config.s3_endpoint == "http://minio:9000"
    assert config.auto_tune is True


def test_s3_endpoint_preflight_reports_missing_minio_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_unknown_host(*args, **kwargs):
        raise socket.gaierror("Name or service not known")

    monkeypatch.setattr("spark_supply_optimizer.cli.socket.getaddrinfo", raise_unknown_host)

    with pytest.raises(RuntimeError, match="docker compose up -d minio") as error:
        validate_s3_endpoint_reachable(SparkRuntimeConfig(s3_endpoint="http://minio:9000"))

    assert "docker compose run --rm upload-data" in str(error.value)
    assert "--no-deps" in str(error.value)
