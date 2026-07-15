$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "fl_mcp_validation_helpers.ps1")

$settingsPath = Join-Path $PSScriptRoot ".env.local"
if (-not (Test-Path -LiteralPath $settingsPath)) {
    throw "Missing tests/.env.local."
}

$settings = Read-DotEnv $settingsPath
$comfyRoot = $settings["TTS_SUITE_TEST_COMFY_ROOT"]
$flRoot = Join-Path $comfyRoot "custom_nodes\ComfyUI_FL-MCP"
$flPython = Join-Path $flRoot ".mcp_venv\Scripts\python.exe"
$mcpServer = Join-Path $flRoot "backend\mcp_server.py"
$backendServer = Join-Path $flRoot "backend\server.py"
$canonicalLauncher = Join-Path $flRoot "codex_mcp_server.cmd"

foreach ($requiredPath in @($flPython, $mcpServer, $backendServer, $canonicalLauncher)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required FL-MCP path not found: $requiredPath"
    }
}

$backendHealth = Wait-JsonEndpoint "http://127.0.0.1:8000/health" 2
if (-not $backendHealth) {
    $root = Quote-PowerShell $flRoot
    $python = Quote-PowerShell $flPython
    $command = "`$env:PYTHONUTF8='1'; `$env:PYTHONIOENCODING='utf-8'; Set-Location -LiteralPath $root; & $python 'backend/server.py'"
    Start-VisibleTerminal "FL-MCP bridge :8000" $flRoot $command
    $backendHealth = Wait-JsonEndpoint "http://127.0.0.1:8000/health" 30
}
if (-not $backendHealth) {
    throw "FL-MCP backend did not become ready on port 8000."
}

$sessionLine = Get-Content -LiteralPath $canonicalLauncher |
    Where-Object { $_ -match '^set "FL_MCP_SESSION_ID=(.+)"$' } |
    Select-Object -First 1
if ($sessionLine -notmatch '^set "FL_MCP_SESSION_ID=(.+)"$') {
    throw "FL_MCP_SESSION_ID was not found in $canonicalLauncher."
}

$env:COMFYUI_SERVER_URL = "http://127.0.0.1:8188"
$env:FL_MCP_MODE = "subprocess"
$env:FL_MCP_SESSION_ID = $Matches[1]
$env:FL_MCP_WS_URL = "ws://127.0.0.1:8000/ws"

& $flPython $mcpServer
exit $LASTEXITCODE
