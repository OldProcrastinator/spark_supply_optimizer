import pytest

from spark_supply_optimizer.config import SparkRuntimeConfig
from spark_supply_optimizer.spark import build_spark


@pytest.fixture(scope="session")
def spark():
    session = build_spark(
        SparkRuntimeConfig(
            app_name="unit-test-spark",
            master="local[2]",
            shuffle_partitions=2,
            auto_tune=False,
        )
    )
    try:
        yield session
    finally:
        session.stop()
