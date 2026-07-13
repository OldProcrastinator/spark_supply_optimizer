#!/usr/bin/env sh
set -eu

SPARK_CLASS="$(python -c 'from pathlib import Path; import pyspark; print(Path(pyspark.__file__).parent / "bin" / "spark-class")')"

exec "$SPARK_CLASS" org.apache.spark.deploy.master.Master \
    --host "${SPARK_MASTER_HOST:-0.0.0.0}" \
    --port "${SPARK_MASTER_PORT:-7077}" \
    --webui-port "${SPARK_MASTER_WEBUI_PORT:-8080}"
