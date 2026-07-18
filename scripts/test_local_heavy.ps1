param()

$ErrorActionPreference = "Stop"

function Resolve-Python {
    $venvPython = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return (Resolve-Path $venvPython).Path
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $pythonCommand) {
        return $pythonCommand.Source
    }

    throw "Python executable not found. Activate the project environment or install Python first."
}

function Assert-Command {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        throw "Required command '$Name' was not found. Install it or add it to PATH before pushing."
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Resolve-Python

if (-not $env:SUMO_HOME) {
    throw "SUMO_HOME is not set. Configure your local SUMO installation before running local heavy validation."
}

Assert-Command -Name "sumo"

$sumoBinary = Get-Command sumo -ErrorAction Stop
Write-Host "Running local heavy validation with $python"
Write-Host "Using SUMO_HOME=$($env:SUMO_HOME)"
Write-Host "Using sumo binary $($sumoBinary.Source)"

Push-Location $repoRoot
try {
    & $python -m pytest -m local_heavy tests/integration_local/test_gym_api.py tests/integration_local/test_pz_api.py
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
