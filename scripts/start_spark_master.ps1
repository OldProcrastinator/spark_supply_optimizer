param(
    [string]$HostIp = "0.0.0.0",
    [int]$Port = 7077,
    [int]$WebUiPort = 8080
)

$ErrorActionPreference = "Stop"
$SparkClass = Join-Path $PSScriptRoot "..\.venv\Lib\site-packages\pyspark\bin\spark-class.cmd"

& $SparkClass org.apache.spark.deploy.master.Master `
    --host $HostIp `
    --port $Port `
    --webui-port $WebUiPort
