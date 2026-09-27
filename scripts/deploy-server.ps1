# Deploys the web server (README "Web server"): builds the committed tree, copies the release
# build to %LOCALAPPDATA%\007 Solver Server, points the per-user start-up entry at it and
# restarts it. scripts/post-commit runs this in the background after each commit on main.
param(
    # Also deploy uncommitted changes, so the live server may match no commit.
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$log = Join-Path $repo 'build\deploy-server.log'
$buildLog = Join-Path $repo 'build\deploy-build.log'
$serverDir = Join-Path $env:LOCALAPPDATA '007 Solver Server'
# A name of its own keeps the desktop app's installer from closing the server as the app.
$server = Join-Path $serverDir '007solver-server.exe'

function Write-Log([string]$message) {
    Add-Content -Path $log -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $message"
}

function Test-Clean {
    -not (git -C $repo status --porcelain)
}

function Stop-Matching([scriptblock]$filter) {
    Get-CimInstance Win32_Process | Where-Object -FilterScript $filter | ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        Wait-Process -Id $_.ProcessId -ErrorAction SilentlyContinue
    }
}

New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
# Deploys run one at a time, each for the commit checked out when it starts.
$mutex = [System.Threading.Mutex]::new($false, 'Local\007SolverDeployServer')
try { [void]$mutex.WaitOne() } catch [System.Threading.AbandonedMutexException] {}
try {
    $commit = git -C $repo rev-parse --short HEAD
    if (-not ($Force -or (Test-Clean))) {
        Write-Log "Skipped ${commit}: uncommitted changes"
        return
    }
    Write-Log "Building $commit$(if ($Force) { ' with any uncommitted changes' })"
    # A server started by hand, as from a build tree, would block the build or the port.
    Stop-Matching { $_.Name -eq '007solver.exe' -and $_.CommandLine -match '--serve' }
    # The service build needs the MSVC environment and UTF-8 output for Ninja.
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $vcvars = & $vswhere -latest -products * -find 'VC\Auxiliary\Build\vcvars64.bat'
    cmd /c "`"$vcvars`" >nul && chcp 65001 >nul && npm --prefix `"$repo\desktop`" run build -- --no-bundle > `"$buildLog`" 2>&1"
    if ($LASTEXITCODE -ne 0) {
        Write-Log "Build of $commit failed (build\deploy-build.log); the previous server keeps running"
        return
    }
    if ((git -C $repo rev-parse --short HEAD) -ne $commit -or -not ($Force -or (Test-Clean))) {
        Write-Log "Skipped ${commit}: the tree changed during the build"
        return
    }

    # Stops the server and its services, whose files the copies below replace.
    Stop-Matching { $_.ExecutablePath -like "$serverDir\*" }
    $release = Join-Path $repo 'desktop\src-tauri\target\release'
    New-Item -ItemType Directory -Force -Path $serverDir | Out-Null
    Copy-Item (Join-Path $release '007solver.exe') $server -Force
    Copy-Item (Join-Path $release 'solver-service*.exe'), (Join-Path $release '*.dll') $serverDir -Force
    # Overwriting the entry replaces the previous version's start at logon.
    Set-ItemProperty -Path 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run' -Name '007 Solver server' -Value "`"$server`" --serve"
    Start-Process -FilePath $server -ArgumentList '--serve' -WorkingDirectory $serverDir

    foreach ($attempt in 1..20) {
        Start-Sleep -Milliseconds 500
        try {
            if ((Invoke-WebRequest -Uri 'http://127.0.0.1:8007/' -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) {
                Write-Log "Deployed $commit"
                return
            }
        } catch {}
    }
    Write-Log "Deployed $commit, but the server does not answer on port 8007"
} catch {
    Write-Log "Deploy failed: $_"
    throw
} finally {
    $mutex.ReleaseMutex()
}
