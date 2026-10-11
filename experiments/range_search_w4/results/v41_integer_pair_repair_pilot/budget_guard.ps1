# Start beside run_pilot.py; repeat this guard for a permitted retry.
# The first attempt's epoch remains the deadline, so diagnostic gaps count.
$ErrorActionPreference = 'Stop'
$pilotBudgetOut = $PSScriptRoot
$pilotAttemptLog = Join-Path $pilotBudgetOut 'ATTEMPT_FREEZE_LOG.jsonl'
while (-not (Test-Path -LiteralPath $pilotAttemptLog)) { Start-Sleep -Seconds 1 }
$pilotAttemptRows = @(Get-Content -LiteralPath $pilotAttemptLog | ForEach-Object { $_ | ConvertFrom-Json })
$pilotFirstEpoch = [double]$pilotAttemptRows[0].freeze.started_unix
$pilotLatest = $pilotAttemptRows[-1]
$pilotDeadline = $pilotFirstEpoch + 900
$pilotPid = [int]$pilotLatest.freeze.pid
$pilotProcess = Get-Process -Id $pilotPid -ErrorAction SilentlyContinue
if ($pilotProcess) { $pilotIdentity = $pilotProcess.StartTime.ToUniversalTime().Ticks }
$pilotForced = $false
while ($true) {
    $pilotNow = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000.0
    $pilotProcess = Get-Process -Id $pilotPid -ErrorAction SilentlyContinue
    if (-not $pilotProcess) { break }
    if ($pilotProcess.StartTime.ToUniversalTime().Ticks -ne $pilotIdentity) { throw 'PID identity changed' }
    if ($pilotNow -ge $pilotDeadline) {
        if ($pilotProcess.ProcessName -notmatch '^python') { throw 'Unexpected budget process' }
        Stop-Process -Id $pilotPid
        $pilotForced = $true
        break
    }
    Start-Sleep -Seconds 1
}
@{ first_started_unix=$pilotFirstEpoch; deadline_unix=$pilotDeadline; observed_ended_unix=$pilotNow; whole_round_wall_seconds=$pilotNow-$pilotFirstEpoch; between_attempt_diagnostic_seconds=[double]$pilotLatest.freeze.started_unix-$pilotFirstEpoch-[double]$pilotLatest.prior_GPU_seconds; forced_stop=$pilotForced; hard_cap_seconds=900 } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $pilotBudgetOut 'GPU_WALL_BUDGET_GUARD.json') -Encoding utf8
