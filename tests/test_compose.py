import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compose_service_block(compose_text: str, service: str) -> str:
    pattern = rf"(?ms)^  {re.escape(service)}:\n(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:|\Z)"
    match = re.search(pattern, compose_text)
    if match is None:
        raise AssertionError(f"Service {service!r} was not found in docker-compose.yml.")
    return match.group("body")


def yaml_scalar(block: str, key: str) -> str:
    pattern = rf"(?m)^    {re.escape(key)}:\s*[\"']?(?P<value>[^\"'\n]+)[\"']?\s*$"
    match = re.search(pattern, block)
    if match is None:
        raise AssertionError(f"Key {key!r} was not found in service block.")
    return match.group("value")


def env_scalar(block: str, key: str) -> str:
    pattern = rf"(?m)^      {re.escape(key)}:\s*[\"']?(?P<value>[^\"'\n]+)[\"']?\s*$"
    match = re.search(pattern, block)
    if match is None:
        raise AssertionError(f"Environment key {key!r} was not found in service block.")
    return match.group("value")


def test_local_spark_workers_do_not_cap_cpu_or_memory_by_default() -> None:
    compose_text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    block = compose_service_block(compose_text, "spark-worker")

    assert "SPARK_WORKER_CORES" not in block
    assert "SPARK_WORKER_MEMORY" not in block


def test_multihost_spark_worker_limits_are_opt_in() -> None:
    compose_text = (ROOT / "docker-compose.multihost.yml").read_text(encoding="utf-8")
    block = compose_service_block(compose_text, "spark-worker")

    assert "SPARK_WORKER_CORES: ${SPARK_WORKER_CORES:-}" in block
    assert "SPARK_WORKER_MEMORY: ${SPARK_WORKER_MEMORY:-}" in block


def test_cluster_driver_services_have_resolvable_spark_local_ip() -> None:
    compose_text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    for service in ("driver", "pipeline-cluster-s3"):
        block = compose_service_block(compose_text, service)

        assert yaml_scalar(block, "hostname") == env_scalar(block, "SPARK_LOCAL_IP")


def test_cluster_driver_services_publish_spark_application_ui_port() -> None:
    compose_text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    for service in ("driver", "pipeline-cluster-s3"):
        block = compose_service_block(compose_text, service)

        assert '- "4040:4040"' in block


def test_readme_uses_service_mode_driver_without_recreating_scaled_workers() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "docker compose --profile driver up --no-deps pipeline-cluster-s3" in readme
    assert "docker compose rm -f pipeline-cluster-s3" in readme


def test_readme_documents_safe_local_scaled_worker_resources() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "unexpected EOF" in readme
    assert '$env:SPARK_WORKER_CORES="4"' in readme
    assert '$env:SPARK_WORKER_MEMORY="8g"' in readme
