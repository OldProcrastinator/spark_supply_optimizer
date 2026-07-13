param(
    [Parameter(Mandatory = $true)]
    [string]$MasterUrl,
    [int]$Cores = 0,
    [string]$Memory = ""
)

$ErrorActionPreference = "Stop"
$SparkClass = Join-Path $PSScriptRoot "..\.venv\Lib\site-packages\pyspark\bin\spark-class.cmd"

$ArgsList = @("org.apache.spark.deploy.worker.Worker", $MasterUrl)
if ($Cores -gt 0) {
    $ArgsList += @("--cores", "$Cores")
}
if ($Memory -ne "") {
    $ArgsList += @("--memory", $Memory)
}

& $SparkClass @ArgsList
