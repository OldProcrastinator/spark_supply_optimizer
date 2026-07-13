# Spark Supply Optimizer

Учебный PySpark-проект для Kaggle M5 Forecasting - Accuracy. Проект специально
собран так, чтобы локальный запуск и запуск на нескольких компьютерах были похожи
на нормальный production ML setup:

```text
код        -> Docker image
данные     -> MinIO / S3-compatible object storage
вычисления -> Spark driver + Spark master + Spark workers
результаты -> MinIO bucket + локальные JSON reports
```

Данные не копируются на каждый worker. Один компьютер поднимает MinIO, CSV один раз
загружаются в bucket, а Spark читает и пишет `s3a://m5-data/...`.

## Быстрый Запуск На Одном Компьютере

Требования:

- Docker Desktop или Docker Engine с Compose plugin.
- CSV M5 в папке `data/`:
  `sales_train_validation.csv`, `calendar.csv`, `sell_prices.csv`.

Команды:

```powershell
git clone <repo-url>
cd spark_supply_optimizer
docker compose build
docker compose up -d minio
docker compose run --rm upload-data
docker compose run --rm pipeline-s3
```

Что произойдет:

- `minio` поднимет локальное S3-хранилище.
- `upload-data` создаст bucket `m5-data` и загрузит `data/*.csv` в `s3a://m5-data/data`.
- `pipeline-s3` выполнит `prepare -> train -> predict -> benchmark -> summary`.
- Features, model и predictions будут лежать в MinIO:
  `s3a://m5-data/artifacts/...`.
- JSON-отчеты будут доступны локально в `artifacts/reports/`.

MinIO UI:

```text
http://localhost:9001
login: minioadmin
password: minioadmin123
```

## Быстрый Запуск Spark-Кластера На Одном Компьютере

Это режим для проверки standalone Spark: master, workers, driver и MinIO живут на
одной машине, но данные все равно идут через S3.

```powershell
docker compose build
docker compose up -d minio
docker compose run --rm upload-data
docker compose up -d spark-master
docker compose up -d --scale spark-worker=2 spark-worker
docker compose --profile driver run --rm pipeline-cluster-s3
```

Spark UI:

```text
http://localhost:8080
```

Остановить:

```powershell
docker compose down
```

## Запуск На Нескольких Компьютерах

Все компьютеры должны быть в одной LAN/VPN. Это нужно и для Spark, и для MinIO:

- Spark master слушает `7077`.
- Spark master UI слушает `8080`.
- MinIO API слушает `9000`.
- MinIO UI слушает `9001`.
- Driver должен быть доступен workers по сети.

Роли:

- **storage/master machine** - поднимает MinIO и Spark master.
- **worker machines** - поднимают Spark workers.
- **driver machine** - запускает `m5-spark run-pipeline`.

На всех машинах:

```bash
git clone <repo-url>
cd spark_supply_optimizer
cp .env.example .env
docker compose -f docker-compose.multihost.yml build
```

В `.env` укажите реальные IP:

```env
SPARK_MASTER_HOST=192.168.1.10
SPARK_DRIVER_HOST=192.168.1.20
MINIO_ENDPOINT=http://192.168.1.10:9000
```

На storage/master machine:

```bash
docker compose -f docker-compose.multihost.yml --profile storage up -d minio
docker compose -f docker-compose.multihost.yml --profile upload run --rm upload-data
docker compose -f docker-compose.multihost.yml --profile master up -d spark-master
```

На каждой worker machine:

```bash
docker compose -f docker-compose.multihost.yml --profile worker up -d spark-worker
```

На driver machine:

```bash
docker compose -f docker-compose.multihost.yml --profile driver run --rm driver
```

Проверки:

- Spark UI: `http://MASTER_IP:8080`
- MinIO UI: `http://MINIO_IP:9001`
- Reports: `artifacts/reports/` на driver machine

## Что Где Происходит

```text
src/spark_supply_optimizer/
  cli.py       # CLI: prepare/train/predict/benchmark/run-pipeline
  config.py    # Spark runtime и feature engineering config
  spark.py     # SparkSession, Java checks, S3A/MinIO config, runtime reports
  m5.py        # M5 CSV loading, features, model training, prediction, benchmark
  metrics.py   # validation metrics

scripts/
  start_spark_master.sh / .ps1
  start_spark_worker.sh / .ps1

data/       # локальная папка только для исходной загрузки CSV в MinIO
artifacts/  # локальные reports; тяжелые Spark artifacts лежат в MinIO
```

Pipeline:

```text
s3a://m5-data/data/*.csv
  -> prepare
  -> s3a://m5-data/artifacts/features
  -> train
  -> s3a://m5-data/artifacts/model
  -> predict
  -> s3a://m5-data/artifacts/predictions
```

Команды CLI:

- `m5-spark prepare` - строит Parquet features.
- `m5-spark train` - обучает Spark ML `RandomForestRegressor`.
- `m5-spark predict` - пишет прогноз.
- `m5-spark benchmark` - запускает Spark aggregation для сравнения скорости.
- `m5-spark summarize-reports` - собирает JSON reports.
- `m5-spark run-pipeline` - запускает все этапы подряд.

## Почему MinIO

Сетевые папки SMB/NFS требуют ручной настройки на каждой машине и легко ломаются
из-за прав, путей и Windows/Linux различий. MinIO проще:

- поднимается одной Docker-командой;
- дает S3-compatible API;
- workers не хранят локальные копии датасета;
- модель, features и predictions лежат в одном месте;
- локальный запуск похож на cloud/production ML.

## Тесты

Обычные тесты:

```powershell
docker compose run --rm test
docker compose run --rm lint
```

Интеграционный тест Spark + MinIO/S3A:

```powershell
docker compose run --rm test-minio
```

Обычный `pytest` в Docker Compose тоже запускает MinIO-тест. Service `dev` и
service `test` зависят от `upload-data`, поэтому compose сначала поднимает MinIO,
создает bucket и загружает CSV, а потом запускает тесты.

## Полезные Команды

Перезалить CSV в MinIO после изменения `data/`:

```powershell
docker compose run --rm upload-data
```

Быстрый debug pipeline на части дней:

```powershell
docker compose run --rm pipeline-s3 m5-spark run-pipeline --data-dir s3a://m5-data/data --features-dir s3a://m5-data/artifacts/features --model-dir s3a://m5-data/artifacts/model --predictions-dir s3a://m5-data/artifacts/predictions --reports-dir artifacts/reports --limit-days 120
```

Почистить контейнеры:

```powershell
docker compose down --remove-orphans
```

Почистить MinIO volume со всеми данными:

```powershell
docker compose down -v
```

## PyCharm

Нормальный workflow в PyCharm такой:

1. MinIO запускается через Docker Compose.
2. Данные загружаются в MinIO через `upload-data`.
3. Python-код запускается и отлаживается через Docker Compose interpreter `dev`.
4. В run configurations вы запускаете `m5-spark` как обычный Python module, но
   входы/выходы указываете в `s3a://m5-data/...`.

### Interpreter

Сначала соберите image:

```powershell
docker compose build dev
```

В PyCharm Professional:

1. `Settings | Project | Python Interpreter`.
2. `Add Interpreter`.
3. `On Docker Compose`.
4. `Configuration files`: `docker-compose.yml`.
5. `Service`: `dev`.
6. `Python interpreter path`: `/opt/venv/bin/python`.
7. В run configurations оставляйте `Working directory`: `$PROJECT_DIR$`.

После этого в конфигурациях **не надо** выбирать Docker вручную. Вы создаете
обычные Python/pytest конфигурации, а в поле `Python interpreter` выбираете
созданный Docker Compose interpreter.

### MinIO Перед Отладкой

Перед запуском pipeline поднимите MinIO и загрузите CSV:

```powershell
docker compose up -d minio
docker compose run --rm upload-data
```

MinIO UI:

```text
http://localhost:9001
login: minioadmin
password: minioadmin123
```

### Run Configuration: Full Pipeline

Создайте `Python` configuration:

```text
Name: m5 run pipeline s3
Run: Module name
Module name: spark_supply_optimizer.cli
Parameters: run-pipeline --data-dir s3a://m5-data/data --features-dir s3a://m5-data/artifacts/features --model-dir s3a://m5-data/artifacts/model --predictions-dir s3a://m5-data/artifacts/predictions --reports-dir artifacts/reports
Python interpreter: Docker Compose interpreter service dev
Working directory: $PROJECT_DIR$
```

Environment variables:

```text
SPARK_S3_ENDPOINT=http://minio:9000;AWS_ACCESS_KEY_ID=minioadmin;AWS_SECRET_ACCESS_KEY=minioadmin123
```

Эта конфигурация запускает весь pipeline в `local[*]` Spark mode, но данные и
артефакты читает/пишет через MinIO.

Для быстрой отладки можно добавить в `Parameters`:

```text
--limit-days 120
```

### Run Configurations: Отдельные Этапы

Все конфигурации создаются как `Python`, `Run: Module name`,
`Module name: spark_supply_optimizer.cli`, с тем же interpreter, working directory
и environment variables.

Prepare:

```text
Name: m5 prepare s3
Parameters: prepare --data-dir s3a://m5-data/data --output-dir s3a://m5-data/artifacts/features --report-path artifacts/reports/prepare.json
```

Train:

```text
Name: m5 train s3
Parameters: train --features-dir s3a://m5-data/artifacts/features --model-dir s3a://m5-data/artifacts/model --report-path artifacts/reports/train.json --max-depth 6 --num-trees 40
```

Predict:

```text
Name: m5 predict s3
Parameters: predict --features-dir s3a://m5-data/artifacts/features --model-dir s3a://m5-data/artifacts/model --output-dir s3a://m5-data/artifacts/predictions --report-path artifacts/reports/predict.json
```

Benchmark:

```text
Name: m5 benchmark s3
Parameters: benchmark --features-dir s3a://m5-data/artifacts/features --report-path artifacts/reports/benchmark.json
```

Summary:

```text
Name: m5 summarize reports
Parameters: summarize-reports --reports-dir artifacts/reports --output-path artifacts/reports/pipeline_summary.json
```

Порядок ручного запуска:

```text
prepare -> train -> predict -> benchmark -> summarize-reports
```

`train` нельзя запускать до `prepare`, потому что еще нет
`s3a://m5-data/artifacts/features`. `predict` нельзя запускать до `train`, потому
что еще нет `s3a://m5-data/artifacts/model`.

### Run Configuration: Spark Cluster

Если локально поднят standalone Spark cluster:

```powershell
docker compose up -d spark-master
docker compose up -d --scale spark-worker=2 spark-worker
```

Добавьте в `Parameters` любой CLI-конфигурации:

```text
--master spark://spark-master:7077
```

Пример full pipeline через cluster:

```text
run-pipeline --master spark://spark-master:7077 --data-dir s3a://m5-data/data --features-dir s3a://m5-data/artifacts/features --model-dir s3a://m5-data/artifacts/model --predictions-dir s3a://m5-data/artifacts/predictions --reports-dir artifacts/reports
```

Spark UI:

```text
http://localhost:8080
```

### Pytest Configurations

Все unit-тесты:

```text
Name: pytest all
Run: Module name
Module name: pytest
Parameters: -v
Python interpreter: Docker Compose interpreter service dev
Working directory: $PROJECT_DIR$
```

MinIO integration test:

```text
Name: pytest minio
Run: Module name
Module name: pytest
Parameters: -v tests/integration/test_minio_s3_runtime.py
Python interpreter: Docker Compose interpreter service dev
Working directory: $PROJECT_DIR$
Environment variables: MINIO_INTEGRATION_ENDPOINT=http://minio:9000;MINIO_BUCKET=m5-data;AWS_ACCESS_KEY_ID=minioadmin;AWS_SECRET_ACCESS_KEY=minioadmin123
```

Перед `pytest minio` должен быть запущен MinIO и выполнен `upload-data`:

```powershell
docker compose up -d minio
docker compose run --rm upload-data
```

### Breakpoints

Breakpoints в `cli.py`, `m5.py`, `spark.py` будут работать в обычном PyCharm
debug run, потому что проект смонтирован в контейнер как `/app`, а
`PYTHONPATH=/app/src` задан в Docker image.

Если Spark выполняется в `local[*]`, breakpoint в driver-коде работает ожидаемо.
Если запуск идет через standalone cluster, часть работы уходит в executors на
workers; breakpoint в Python driver-коде все равно полезен для CLI, построения
SparkSession, чтения конфигов и orchestration, но не остановит JVM-часть Spark ML.

## Локальный Запуск Без Docker

Не основной путь. Spark требует Java и Hadoop/S3A-настройки, поэтому поддерживаемый
вариант для этого проекта - Docker Compose.

Для быстрых unit-тестов без Spark runtime:

```powershell
uv sync --extra dev
uv run pytest tests/test_*.py
uv run ruff check .
```
