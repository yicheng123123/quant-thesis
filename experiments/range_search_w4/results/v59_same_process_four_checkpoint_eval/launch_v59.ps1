$ErrorActionPreference = 'Stop'
$root = 'E:\quant-thesis'
$out = Join-Path $root 'experiments\range_search_w4\results\v59_same_process_four_checkpoint_eval'
$marker = Join-Path $out 'START_EPOCH.json'
if (-not (Test-Path -LiteralPath $marker)) {
    $now = [DateTimeOffset]::UtcNow
    $record = [ordered]@{
        version = 'v59'
        start_unix = $now.ToUnixTimeMilliseconds() / 1000.0
        start_utc = $now.ToString('o')
        pre_start_audit_charge_seconds = 60.0
        pre_start_charge_basis = 'Conservative charge for read-only source/runtime inspection completed before this marker; separate from GPU time.'
    }
    $json = $record | ConvertTo-Json -Depth 5
    [System.IO.File]::WriteAllText($marker, $json + "`n", [System.Text.UTF8Encoding]::new($false))
}
Set-Location -LiteralPath $root
& (Join-Path $root '.venv\Scripts\python.exe') (Join-Path $out 'run_v59_eval.py')
exit $LASTEXITCODE
