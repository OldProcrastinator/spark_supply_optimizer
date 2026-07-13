"""Command line interface for the M5 Spark baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from spark_supply_optimizer.config import M5FeatureConfig, SparkRuntimeConfig
from spark_supply_optimizer.m5 import (
    benchmark_feature_scan,
    predict_latest,
    prepare_features,
    train_model,
)
from spark_supply_optimizer.spark import build_spark, spark_runtime_report


def runtime_config(args: argparse.Namespace) -> SparkRuntimeConfig:
    """Create Spark runtime config from parsed CLI arguments."""

    return SparkRuntimeConfig(
        master=args.master,
        shuffle_partitions=args.shuffle_partitions,
        executor_memory=args.executor_memory,
        driver_memory=args.driver_memory,
        local_dir=args.local_dir,
        s3_endpoint=args.s3_endpoint,
        s3_access_key=args.s3_access_key,
        s3_secret_key=args.s3_secret_key,
        auto_tune=not args.no_auto_tune,
    )


def feature_config(args: argparse.Namespace) -> M5FeatureConfig:
    """Create M5 feature config from parsed CLI arguments."""

    return M5FeatureConfig(
        validation_days=args.validation_days,
        limit_days=args.limit_days,
        output_partitions=getattr(args, "output_partitions", None),
        unpivot_chunk_size=getattr(args, "unpivot_chunk_size", 16),
        auto_partitions=getattr(args, "auto_output_partitions", False),
    )


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    """Add Spark runtime options shared by all subcommands."""

    parser.add_argument(
        "--master",
        default=os.environ.get("SPARK_MASTER", "local[*]"),
        help="Spark master URL, for example spark://host:7077.",
    )
    parser.add_argument(
        "--shuffle-partitions",
        type=int,
        default=int(os.environ["SPARK_SHUFFLE_PARTITIONS"])
        if os.environ.get("SPARK_SHUFFLE_PARTITIONS")
        else None,
    )
    parser.add_argument("--executor-memory", default=os.environ.get("SPARK_EXECUTOR_MEMORY"))
    parser.add_argument("--driver-memory", default=os.environ.get("SPARK_DRIVER_MEMORY"))
    parser.add_argument("--local-dir", default=os.environ.get("SPARK_LOCAL_DIR"))
    parser.add_argument("--no-auto-tune", action="store_true")
    parser.add_argument("--s3-endpoint", default=os.environ.get("SPARK_S3_ENDPOINT"))
    parser.add_argument("--s3-access-key", default=os.environ.get("AWS_ACCESS_KEY_ID"))
    parser.add_argument("--s3-secret-key", default=os.environ.get("AWS_SECRET_ACCESS_KEY"))


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(prog="m5-spark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Build Parquet features from raw M5 CSV files.")
    add_runtime_args(prepare)
    prepare.add_argument("--data-dir", default="data")
    prepare.add_argument("--output-dir", default="artifacts/features")
    prepare.add_argument("--validation-days", type=int, default=28)
    prepare.add_argument("--limit-days", type=int, default=None)
    prepare.add_argument("--output-partitions", type=int, default=None)
    prepare.add_argument("--unpivot-chunk-size", type=int, default=16)
    prepare.add_argument("--auto-output-partitions", action="store_true")
    prepare.add_argument("--report-path", default="artifacts/reports/prepare.json")

    train = subparsers.add_parser("train", help="Train the Spark ML baseline.")
    add_runtime_args(train)
    train.add_argument("--features-dir", default="artifacts/features")
    train.add_argument("--model-dir", default="artifacts/model")
    train.add_argument("--validation-days", type=int, default=28)
    train.add_argument("--limit-days", type=int, default=None)
    train.add_argument("--max-depth", type=int, default=6)
    train.add_argument("--num-trees", type=int, default=40)
    train.add_argument("--report-path", default="artifacts/reports/train.json")

    predict = subparsers.add_parser(
        "predict",
        help="Score the latest horizon from prepared features.",
    )
    add_runtime_args(predict)
    predict.add_argument("--features-dir", default="artifacts/features")
    predict.add_argument("--model-dir", default="artifacts/model")
    predict.add_argument("--output-dir", default="artifacts/predictions")
    predict.add_argument("--horizon", type=int, default=28)
    predict.add_argument("--report-path", default="artifacts/reports/predict.json")

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Time a Spark aggregation over prepared features.",
    )
    add_runtime_args(benchmark)
    benchmark.add_argument("--features-dir", default="artifacts/features")
    benchmark.add_argument("--report-path", default="artifacts/reports/benchmark.json")

    summary = subparsers.add_parser("summarize-reports", help="Summarize pipeline stage reports.")
    summary.add_argument("--reports-dir", default="artifacts/reports")
    summary.add_argument("--output-path", default="artifacts/reports/pipeline_summary.json")

    pipeline = subparsers.add_parser(
        "run-pipeline",
        help="Run prepare, train, predict, benchmark, and summarize.",
    )
    add_runtime_args(pipeline)
    pipeline.add_argument("--data-dir", default="data")
    pipeline.add_argument("--features-dir", default="artifacts/features")
    pipeline.add_argument("--model-dir", default="artifacts/model")
    pipeline.add_argument("--predictions-dir", default="artifacts/predictions")
    pipeline.add_argument("--reports-dir", default="artifacts/reports")
    pipeline.add_argument("--validation-days", type=int, default=28)
    pipeline.add_argument("--limit-days", type=int, default=None)
    pipeline.add_argument("--output-partitions", type=int, default=None)
    pipeline.add_argument("--unpivot-chunk-size", type=int, default=16)
    pipeline.add_argument("--auto-output-partitions", action="store_true")
    pipeline.add_argument("--max-depth", type=int, default=6)
    pipeline.add_argument("--num-trees", type=int, default=40)
    pipeline.add_argument("--horizon", type=int, default=28)
    pipeline.add_argument("--skip-benchmark", action="store_true")

    return parser


def write_json_report(path: str | Path, payload: dict[str, object]) -> None:
    """Write a human-readable JSON report to the host-mounted artifacts directory."""

    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_stage_report(
    stage: str,
    started_at: float,
    spark,
    inputs: dict[str, object],
    outputs: dict[str, object],
    result: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a compact run report for comparing one-machine and cluster runs."""

    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "stage": stage,
        "duration_seconds": time.perf_counter() - started_at,
        "resources": spark_runtime_report(spark),
        "inputs": inputs,
        "outputs": outputs,
        "result": result or {},
    }


def summarize_stage_reports(reports_dir: str | Path) -> dict[str, object]:
    """Read stage reports and compute total pipeline timing."""

    root = Path(reports_dir)
    stages = []
    for stage in ("prepare", "train", "predict", "benchmark"):
        path = root / f"{stage}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        stages.append(
            {
                "stage": stage,
                "duration_seconds": payload.get("duration_seconds"),
                "master": payload.get("resources", {}).get("spark", {}).get("master"),
                "deploy_mode": payload.get("resources", {}).get("spark", {}).get("deploy_mode"),
                "executor_host_count": payload.get("resources", {})
                .get("cluster_observed", {})
                .get("executor_host_count"),
                "executor_count": payload.get("resources", {})
                .get("cluster_observed", {})
                .get("executor_count"),
                "report_path": str(path),
            }
        )

    train_pipeline_stages = [stage for stage in stages if stage["stage"] in {"prepare", "train"}]
    full_pipeline_stages = [
        stage for stage in stages if stage["stage"] in {"prepare", "train", "predict"}
    ]
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "stages": stages,
        "totals": {
            "prepare_train_seconds": sum(
                float(stage["duration_seconds"] or 0.0) for stage in train_pipeline_stages
            ),
            "prepare_train_predict_seconds": sum(
                float(stage["duration_seconds"] or 0.0) for stage in full_pipeline_stages
            ),
        },
    }


def run_pipeline(spark, args: argparse.Namespace) -> dict[str, object]:
    """Run the full training pipeline and write per-stage reports."""

    reports_dir = Path(args.reports_dir)
    config = feature_config(args)

    started_at = time.perf_counter()
    prepare_features(spark, args.data_dir, args.features_dir, config)
    write_json_report(
        reports_dir / "prepare.json",
        build_stage_report(
            "prepare",
            started_at,
            spark,
            inputs={"data_dir": args.data_dir, "limit_days": args.limit_days},
            outputs={"features_dir": args.features_dir},
            result={"output_partitions": args.output_partitions},
        ),
    )

    started_at = time.perf_counter()
    metrics = train_model(
        spark,
        args.features_dir,
        args.model_dir,
        config,
        max_depth=args.max_depth,
        num_trees=args.num_trees,
    )
    write_json_report(
        reports_dir / "train.json",
        build_stage_report(
            "train",
            started_at,
            spark,
            inputs={
                "features_dir": args.features_dir,
                "max_depth": args.max_depth,
                "num_trees": args.num_trees,
            },
            outputs={"model_dir": args.model_dir},
            result={"validation_metrics": metrics},
        ),
    )

    started_at = time.perf_counter()
    predict_latest(spark, args.features_dir, args.model_dir, args.predictions_dir, args.horizon)
    write_json_report(
        reports_dir / "predict.json",
        build_stage_report(
            "predict",
            started_at,
            spark,
            inputs={
                "features_dir": args.features_dir,
                "model_dir": args.model_dir,
                "horizon": args.horizon,
            },
            outputs={"predictions_dir": args.predictions_dir},
        ),
    )

    if not args.skip_benchmark:
        started_at = time.perf_counter()
        benchmark = benchmark_feature_scan(spark, args.features_dir)
        write_json_report(
            reports_dir / "benchmark.json",
            build_stage_report(
                "benchmark",
                started_at,
                spark,
                inputs={"features_dir": args.features_dir},
                outputs={},
                result=benchmark,
            ),
        )

    summary = summarize_stage_reports(reports_dir)
    write_json_report(reports_dir / "pipeline_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    """Run a CLI command."""

    args = build_parser().parse_args(argv)
    if args.command == "summarize-reports":
        summary = summarize_stage_reports(args.reports_dir)
        write_json_report(args.output_path, summary)
        print(json.dumps({**summary, "report_path": args.output_path}, indent=2))
        return

    try:
        spark = build_spark(runtime_config(args))
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from error

    try:
        if args.command == "prepare":
            started_at = time.perf_counter()
            prepare_features(spark, args.data_dir, args.output_dir, feature_config(args))
            report = build_stage_report(
                "prepare",
                started_at,
                spark,
                inputs={"data_dir": args.data_dir, "limit_days": args.limit_days},
                outputs={"features_dir": args.output_dir},
                result={"output_partitions": args.output_partitions},
            )
            write_json_report(args.report_path, report)
            print(json.dumps({**report, "report_path": args.report_path}, indent=2))
        elif args.command == "train":
            started_at = time.perf_counter()
            metrics = train_model(
                spark,
                args.features_dir,
                args.model_dir,
                feature_config(args),
                max_depth=args.max_depth,
                num_trees=args.num_trees,
            )
            report = build_stage_report(
                "train",
                started_at,
                spark,
                inputs={
                    "features_dir": args.features_dir,
                    "max_depth": args.max_depth,
                    "num_trees": args.num_trees,
                },
                outputs={"model_dir": args.model_dir},
                result={"validation_metrics": metrics},
            )
            write_json_report(args.report_path, report)
            print(json.dumps({**report, "report_path": args.report_path}, indent=2))
        elif args.command == "predict":
            started_at = time.perf_counter()
            predict_latest(spark, args.features_dir, args.model_dir, args.output_dir, args.horizon)
            report = build_stage_report(
                "predict",
                started_at,
                spark,
                inputs={
                    "features_dir": args.features_dir,
                    "model_dir": args.model_dir,
                    "horizon": args.horizon,
                },
                outputs={"predictions_dir": args.output_dir},
            )
            write_json_report(args.report_path, report)
            print(json.dumps({**report, "report_path": args.report_path}, indent=2))
        elif args.command == "benchmark":
            started_at = time.perf_counter()
            benchmark = benchmark_feature_scan(spark, args.features_dir)
            report = build_stage_report(
                "benchmark",
                started_at,
                spark,
                inputs={"features_dir": args.features_dir},
                outputs={},
                result=benchmark,
            )
            write_json_report(args.report_path, report)
            print(json.dumps({**report, "report_path": args.report_path}, indent=2))
        elif args.command == "run-pipeline":
            summary = run_pipeline(spark, args)
            report_path = str(Path(args.reports_dir) / "pipeline_summary.json")
            print(
                json.dumps(
                    {**summary, "report_path": report_path},
                    indent=2,
                )
            )
        else:
            raise ValueError(f"Unsupported command: {args.command}")
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
