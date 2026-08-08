$env:PYTHONUTF8 = "1"
$env:REMOTE_OPENAI_BASE_URL = "http://localhost:8000/v1"
Set-Location "C:\Users\USER\Desktop\Tezim\BFCL\gorilla\berkeley-function-call-leaderboard"
# server-instance identity (the field stage4 failed to record)
$created = (Invoke-RestMethod http://localhost:8000/v1/models).data[0].created
"launch $(Get-Date -Format o) pid=$PID server_created=$created" | Out-File -Append -Encoding utf8 tmp\hnav_rag_c1_launcher.log
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
