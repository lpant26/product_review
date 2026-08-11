$pythonPath = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"

if (-not (Test-Path $pythonPath)) {
    Write-Host "Python was not found at: $pythonPath"
    Write-Host "Please install Python or update this script with your Python path."
    exit 1
}

& $pythonPath main.py @args
