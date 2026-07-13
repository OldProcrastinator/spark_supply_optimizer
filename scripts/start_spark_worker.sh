#!/usr/bin/env sh
set -eu

SPARK_CLASS="$(python -c 'from pathlib import Path; import pyspark; print(Path(pyspark.__file__).parent / "bin" / "spark-class")')"
ARGS="org.apache.spark.deploy.worker.Worker ${SPARK_MASTER_URL:-spark://spark-master:7077}"

if [ -n "${SPARK_WORKER_CORES:-}" ]; then
    ARGS="$ARGS --cores ${SPARK_WORKER_CORES}"
fi

if [ -n "${SPARK_WORKER_MEMORY:-}" ]; then
    ARGS="$ARGS --memory ${SPARK_WORKER_MEMORY}"
fi

exec "$SPARK_CLASS" $ARGS
