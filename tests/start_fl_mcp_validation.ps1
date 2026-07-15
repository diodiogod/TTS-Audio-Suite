param(
    [switch]$RestartComfyUI,
    [switch]$NoBrowser,
    [int]$TimeoutSeconds = 90
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "fl_mcp_validation_helpers.ps1")

$envFile = Join-Path $PSScriptRoot ".env.local"
if (-not (Test-Path -LiteralPath $envFile)) {
    throw "Missing tests/.env.local. Configure TTS_SUITE_TEST_COMFY_ROOT and TTS_SUITE_TEST_VENV_PYTHON."
}

$settings = Read-DotEnv $envFile
$comfyRoot = $settings["TTS_SUITE_TEST_COMFY_ROOT"]
$comfyPython = $settings["TTS_SUITE_TEST_VENV_PYTHON"]
$flRoot = Join-Path $comfyRoot "custom_nodes\ComfyUI_FL-MCP"
$flPython = Join-Path $flRoot ".mcp_venv\Scripts\python.exe"
$codexLauncher = Join-Path $flRoot "codex_mcp_server.cmd"

foreach ($requiredPath in @($comfyRoot, $comfyPython, $flRoot, $flPython, $codexLauncher)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) { throw "Required live-validation path not found: $requiredPath" }
}

$listener = Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($RestartComfyUI -and $listener) {
    $owner = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
    if ($owner.CommandLine -notmatch "main.py") {
        throw "Refusing to stop unverified process $($listener.OwningProcess) on port 8188."
    }
    Stop-Process -Id $listener.OwningProcess -Force
    Start-Sleep -Seconds 2
    $listener = $null
}

if (-not $listener) {
    $root = Quote-PowerShell $comfyRoot
    $python = Quote-PowerShell $comfyPython
    $command = "`$env:PYTHONUTF8='1'; `$env:PYTHONIOENCODING='utf-8'; Set-Location -LiteralPath $root; & $python 'main.py' '--listen' '127.0.0.1'"
    Start-VisibleTerminal "ComfyUI :8188" $comfyRoot $command
}

$comfyHealth = Wait-JsonEndpoint "http://127.0.0.1:8188/system_stats" $TimeoutSeconds
if (-not $comfyHealth) { throw "ComfyUI did not become ready on port 8188 within $TimeoutSeconds seconds." }

$backendHealth = Wait-JsonEndpoint "http://127.0.0.1:8000/health" 2
if (-not $backendHealth) {
    $root = Quote-PowerShell $flRoot
    $python = Quote-PowerShell $flPython
    $command = "`$env:PYTHONUTF8='1'; `$env:PYTHONIOENCODING='utf-8'; Set-Location -LiteralPath $root; & $python 'backend/server.py'"
    Start-VisibleTerminal "FL-MCP bridge :8000" $flRoot $command
    $backendHealth = Wait-JsonEndpoint "http://127.0.0.1:8000/health" $TimeoutSeconds
}
if (-not $backendHealth) { throw "FL-MCP bridge did not become ready on port 8000 within $TimeoutSeconds seconds." }

if (-not $NoBrowser) {
    Start-Process "http://127.0.0.1:8188"
}

$configuredSession = ""
$sessionLine = Get-Content -LiteralPath $codexLauncher | Where-Object { $_ -match '^set "FL_MCP_SESSION_ID=(.+)"$' } | Select-Object -First 1
if ($sessionLine -match '^set "FL_MCP_SESSION_ID=(.+)"$') { $configuredSession = $Matches[1] }

$sessionsPayload = Wait-JsonEndpoint "http://127.0.0.1:8000/api/sessions" 20
$sessions = @()
if ($sessionsPayload) {
    if ($sessionsPayload.sessions) { $sessions = @($sessionsPayload.sessions) }
    elseif ($sessionsPayload -is [System.Array]) { $sessions = @($sessionsPayload) }
}

$frontendSessions = @($sessions | Where-Object { $_.has_frontend -or $_.frontend_connected -or $_.frontendConnected })
$matchingSession = $frontendSessions | Where-Object {
    $id = $_.session_id
    if (-not $id) { $id = $_.sessionId }
    $id -eq $configuredSession
} | Select-Object -First 1

$result = [ordered]@{
    comfyui_ready = $true
    comfyui_url = "http://127.0.0.1:8188"
    fl_backend_ready = $true
    fl_backend_url = "http://127.0.0.1:8000"
    configured_session_id = $configuredSession
    frontend_session_count = $frontendSessions.Count
    configured_session_has_frontend = [bool]$matchingSession
    configured_session_has_mcp = [bool]($matchingSession -and ($matchingSession.has_mcp -or $matchingSession.mcp_connected -or $matchingSession.mcpConnected))
    next_action = if (-not $matchingSession) {
        "Refresh ComfyUI and verify the FL-MCP sidebar session; update codex_mcp_server.cmd if its session ID changed."
    } elseif (-not ($matchingSession.has_mcp -or $matchingSession.mcp_connected -or $matchingSession.mcpConnected)) {
        "Restart/open a Codex task now so its configured FL-MCP stdio server joins this browser session."
    } else {
        "Ready: call mcp_capability_audit once, then use only targeted FL-MCP tools."
    }
}

$result | ConvertTo-Json -Depth 5
