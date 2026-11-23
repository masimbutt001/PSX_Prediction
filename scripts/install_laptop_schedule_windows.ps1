# Sets up the PSX daily run on this Windows laptop (alternative to the GitHub cloud run).
#
# Creates a Task Scheduler task that checks every hour and at log-on. Each check runs
# `psx schedule run-daily --catch-up`, which only does work once per PSX trading day after
# market close, so a run missed while the laptop was off or asleep happens at the next check.
#
# Install:   powershell -ExecutionPolicy Bypass -File scripts\install_laptop_schedule_windows.ps1
# Remove:    powershell -ExecutionPolicy Bypass -File scripts\install_laptop_schedule_windows.ps1 -Uninstall
param([switch]$Uninstall)

$ErrorActionPreference = "Stop"
$TaskName = "PSX Daily Pipeline"

if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Removed scheduled task '$TaskName'."
    exit 0
}

$RepoDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Uv = (Get-Command uv -ErrorAction Stop).Source
$LogFile = Join-Path $RepoDir "data\scheduler.out.log"
New-Item -ItemType Directory -Force -Path (Join-Path $RepoDir "data") | Out-Null

$Command = "cd /d `"$RepoDir`" && `"$Uv`" run psx schedule run-daily --catch-up >> `"$LogFile`" 2>&1"
$Action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c $Command" -WorkingDirectory $RepoDir

$Hourly = New-ScheduledTaskTrigger -Once -At (Get-Date).Date `
    -RepetitionInterval (New-TimeSpan -Hours 1) -RepetitionDuration (New-TimeSpan -Days 3650)
$AtLogOn = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 2) -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger @($Hourly, $AtLogOn) `
    -Settings $Settings -Description "Runs the PSX end-of-day pipeline once per trading day." -Force | Out-Null

Write-Host "Installed '$TaskName'. It checks every hour and runs once per PSX trading day after close."
Write-Host "Output goes to $LogFile"
