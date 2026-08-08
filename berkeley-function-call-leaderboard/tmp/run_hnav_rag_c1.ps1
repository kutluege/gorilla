$env:PYTHONUTF8 = "1"
$env:REMOTE_OPENAI_BASE_URL = "http://localhost:8000/v1"
Set-Location "C:\Users\USER\Desktop\Tezim\BFCL\gorilla\berkeley-function-call-leaderboard"
# server-instance identity (the field stage4 failed to record)
$created = (Invoke-RestMethod http://localhost:8000/v1/models).data[0].created
"launch $(Get-Date -Format o) pid=$PID server_created=$created" | Out-File -Append -Encoding utf8 tmp\hnav_rag_c1_launcher.log
# warm-up gate: the freshly provisioned server wedges early requests
# (attempts 2+3 froze on their first template-shaped completion). Require
# 3 consecutive fast template completions before launching; abort after
# ~10 min of failures.
$prompt = "<|im_start|>user`nReply with OK.<|im_end|>`n<|im_start|>assistant`n"
$body = @{model = "Qwen/Qwen3-4B-Instruct-2507"; prompt = $prompt; max_tokens = 8; temperature = 0.0} | ConvertTo-Json
$ok = 0; $tries = 0
while ($ok -lt 3 -and $tries -lt 40) {
    $tries++
    try {
        $t0 = Get-Date
        Invoke-RestMethod -Uri http://localhost:8000/v1/completions -Method Post -Body $body -ContentType "application/json" -TimeoutSec 15 | Out-Null
        if (((Get-Date) - $t0).TotalSeconds -lt 10) { $ok++ } else { $ok = 0 }
    } catch { $ok = 0; Start-Sleep -Seconds 15 }
}
"warmup ok=$ok tries=$tries $(Get-Date -Format o)" | Out-File -Append -Encoding utf8 tmp\hnav_rag_c1_launcher.log
if ($ok -lt 3) {
    "ABORT warmup failed $(Get-Date -Format o)" | Out-File -Append -Encoding utf8 tmp\hnav_rag_c1_launcher.log
    exit 1
}
# benchmark-integrity gate: refuse to launch on a dirty benchmark surface
& C:\Users\USER\miniconda3\envs\BFCL\python.exe bfcl_eval\scripts\integrity_gate.py --since b0b7d12 --out gov_logs\hnav_rag\integrity_at_launch.json *>> tmp\hnav_rag_c1_console.log
if ($LASTEXITCODE -ne 0) {
    "ABORT integrity gate failed $(Get-Date -Format o)" | Out-File -Append -Encoding utf8 tmp\hnav_rag_c1_launcher.log
    exit 1
}
& C:\Users\USER\miniconda3\envs\BFCL\python.exe bfcl_eval\scripts\run_gov_replicates.py `
    --replicates 5 `
    --arms-json "@gov_logs/hnav_rag_c1_arms.json" `
    --result-root result_hnav_rag_c1 `
    --score-root score_hnav_rag_c1 `
    --gov-log-root gov_logs/hnav_rag_c1 `
    --manifest result_hnav_rag_c1/manifest.jsonl `
    *>> tmp\hnav_rag_c1_console.log
$code = $LASTEXITCODE
$created2 = (Invoke-RestMethod http://localhost:8000/v1/models).data[0].created
"exit $(Get-Date -Format o) code=$code server_created_end=$created2" | Out-File -Append -Encoding utf8 tmp\hnav_rag_c1_launcher.log
