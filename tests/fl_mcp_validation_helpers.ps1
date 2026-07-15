function Read-DotEnv([string]$Path) {
    $values = @{}
    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#") -or -not $trimmed.Contains("=")) { continue }
        $name, $value = $trimmed.Split("=", 2)
        $values[$name.Trim()] = $value.Trim().Trim('"').Trim("'")
    }
    return $values
}

function Quote-PowerShell([string]$Value) {
    return "'" + $Value.Replace("'", "''") + "'"
}

function Start-VisibleTerminal([string]$Title, [string]$WorkingDirectory, [string]$Command) {
    $encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Command))
    if (Get-Command wt.exe -ErrorAction SilentlyContinue) {
        & wt.exe -w "tts-audio-suite-validation" new-tab --title $Title -d $WorkingDirectory powershell.exe -NoExit -EncodedCommand $encodedCommand | Out-Null
        return
    }
    Start-Process -FilePath "powershell.exe" -WorkingDirectory $WorkingDirectory -ArgumentList @("-NoExit", "-EncodedCommand", $encodedCommand)
}

function Wait-JsonEndpoint([string]$Uri, [int]$Timeout) {
    $deadline = (Get-Date).AddSeconds($Timeout)
    do {
        try { return Invoke-RestMethod -Uri $Uri -TimeoutSec 3 }
        catch { Start-Sleep -Milliseconds 750 }
    } while ((Get-Date) -lt $deadline)
    return $null
}

