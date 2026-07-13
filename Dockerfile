FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONPATH=/app/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_LINK_MODE=copy
ENV SPARK_JARS_IVY=/opt/spark-ivy
ENV SPARK_S3A_PACKAGES=org.apache.hadoop:hadoop-aws:3.4.2

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps tini \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/spark-ivy \
    && chmod 777 /opt/spark-ivy

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --extra dev

RUN python -c "from pyspark.sql import SparkSession; spark = SparkSession.builder.master('local[1]').appName('prefetch-s3a-jars').config('spark.jars.packages', 'org.apache.hadoop:hadoop-aws:3.4.2').config('spark.jars.ivy', '/opt/spark-ivy').getOrCreate(); spark.stop()"

COPY scripts ./scripts
COPY tests ./tests
RUN chmod +x scripts/*.sh

ENTRYPOINT ["tini", "--"]
CMD ["m5-spark", "--help"]
