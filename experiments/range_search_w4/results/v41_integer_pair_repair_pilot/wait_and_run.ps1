$ErrorActionPreference = 'Stop'
$pilotOut = $PSScriptRoot
$pilotRoot = Split-Path (Split-Path (Split-Path (Split-Path $pilotOut)))
$v40BudgetPath = Join-Path (Split-Path $pilotOut) 'range_w4_winner_union_v40/BUDGET.json'
$pilotPython = Join-Path $pilotRoot '.venv-gptq/Scripts/python.exe'
$previousState = ''
$idleObservations = 0
while ($true) {
    $v40Budget = Get-Content -LiteralPath $v40BudgetPath -Raw | ConvertFrom-Json
    $runningPython = @(Get-Process -Name python,pythonw -ErrorAction SilentlyContinue)
    $v40Done = ($v40Budget.completed_phases -contains 'B') -or ($v40Budget.status -match 'STOPPED|FAILED|COMPLETE.*FINAL|COMPLETE.*AUDIT')
    $stateLine = "v40=$($v40Budget.status), elapsed=$($v40Budget.gpu_experiment_elapsed_seconds), python_count=$($runningPython.Count)"
    if ($stateLine -ne $previousState) { Write-Output $stateLine; $previousState = $stateLine }
    if ($v40Done -and $runningPython.Count -eq 0) { $idleObservations++ } else { $idleObservations = 0 }
    if ($idleObservations -ge 2) { break }
    Start-Sleep -Seconds 20
}
$gate = @{ status='GPU_IDLE_AFTER_V40'; checked_utc=(Get-Date).ToUniversalTime().ToString('o'); v40_status=$v40Budget.status; v40_completed_phases=$v40Budget.completed_phases; python_process_count=0; consecutive_idle_observations=$idleObservations }
$gate | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $pilotOut 'GPU_GATE.json') -Encoding utf8
Write-Output 'GPU gate passed; launching bounded pilot'
& $pilotPython -B -X utf8 (Join-Path $pilotOut 'run_pilot.py')
$pilotStatus = Get-Content -LiteralPath (Join-Path $pilotOut 'RUN_STATUS.json') -Raw | ConvertFrom-Json
if ($pilotStatus.status -eq 'COMPLETE') { & $pilotPython -B -X utf8 (Join-Path $pilotOut 'summarize.py') }
