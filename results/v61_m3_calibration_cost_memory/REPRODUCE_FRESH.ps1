$ErrorActionPreference = 'Stop'
$sourceRoot = $PSScriptRoot
$repoRoot = Split-Path (Split-Path $sourceRoot -Parent) -Parent
$taskDestination = Join-Path $repoRoot ('results/v61_m3_cost_memory_reproduction_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
if (Test-Path -LiteralPath $taskDestination) { throw 'Destination already exists; refusing overwrite' }
New-Item -ItemType Directory -Path $taskDestination | Out-Null
New-Item -ItemType Directory -Path (Join-Path $taskDestination 'M3') | Out-Null
foreach ($name in @('build_measurement.py','measure_once.py','check_measurement.py','summarize_measurement.py','compare_saved_tensors.py','plot_memory.py')) {
    Copy-Item -LiteralPath (Join-Path $sourceRoot $name) -Destination (Join-Path $taskDestination $name)
}
Copy-Item -LiteralPath (Join-Path $sourceRoot 'M3/measurement.py') -Destination (Join-Path $taskDestination 'M3/measurement.py')
$pythonPath = Join-Path $repoRoot '.venv-gptq/Scripts/python.exe'
& $pythonPath -B (Join-Path $taskDestination 'build_measurement.py')
if ($LASTEXITCODE -ne 0) { throw 'Source generation failed' }
& $pythonPath -B (Join-Path $taskDestination 'check_measurement.py')
if ($LASTEXITCODE -ne 0) { throw 'Measurement precheck failed' }
& $pythonPath -B (Join-Path $taskDestination 'measure_once.py')
if ($LASTEXITCODE -ne 0) { throw ('Measurement failed; preserve all records and valid block checkpoints at ' + $taskDestination) }
Write-Output ('Full measurement saved at ' + $taskDestination)
