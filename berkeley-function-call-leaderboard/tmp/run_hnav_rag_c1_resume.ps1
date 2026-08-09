$env:PYTHONUTF8 = "1"
$env:REMOTE_OPENAI_BASE_URL = "http://localhost:8000/v1"
Set-Location "C:\Users\USER\Desktop\Tezim\BFCL\gorilla\berkeley-function-call-leaderboard"
"launch $(Get-Date -Format o) pid=$PID resume=rep$($env:C1_START_REP)/$($env:C1_START_ARM)" | Out-File -Append -Encoding utf8 tmp\hnav_rag_c1_launcher.log
& C:\Users\USER\miniconda3\envs\BFCL\python.exe bfcl_eval\scripts\integrity_gate.py --since b0b7d12 --out gov_logs\hnav_rag\integrity_at_launch.json *>> tmp\hnav_rag_c1_console.log
if ($LASTEXITCODE -ne 0) { "ABORT integrity gate failed $(Get-Date -Format o)" | Out-File -Append -Encoding utf8 tmp\hnav_rag_c1_launcher.log; exit 1 }
& C:\Users\USER\miniconda3\envs\BFCL\python.exe bfcl_eval\scripts\run_gov_replicates.py `
    --replicates 5 `
    --start-replicate $env:C1_START_REP `
    --start-arm $env:C1_START_ARM `
    --arms-json "@gov_logs/hnav_rag_c1_arms.json" `
    --result-root result_hnav_rag_c1 `
    --score-root score_hnav_rag_c1 `
    --gov-log-root gov_logs/hnav_rag_c1 `
    --manifest result_hnav_rag_c1/manifest.jsonl `
    *>> tmp\hnav_rag_c1_console.log
"exit $(Get-Date -Format o) code=$LASTEXITCODE" | Out-File -Append -Encoding utf8 tmp\hnav_rag_c1_launcher.log
