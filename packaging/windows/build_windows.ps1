# Build Kol Studio for Windows (run in PowerShell on a Windows 10/11 x64 PC).
# Needs: uv (https://docs.astral.sh/uv/), Inno Setup 6 (https://jrsoftware.org/isinfo.php) on PATH as iscc.
#   powershell -ExecutionPolicy Bypass -File packaging\windows\build_windows.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\..")
$Version = (Select-String -Path pyproject.toml -Pattern '^version = "(.+)"').Matches[0].Groups[1].Value
$BlueRef = "db047ea"   # maxmelichov/BlueTTS commit tested with BlueV3 v3.0.1

uv venv --python 3.12 .venv
uv pip install --python .venv -e .
# BlueTTS without its dependencies: phonemizer / espeak-ng (GPL) are only used for non-Hebrew languages.
uv pip install --python .venv --no-deps "blue-onnx @ git+https://github.com/maxmelichov/BlueTTS@$BlueRef"
uv pip install --python .venv pyinstaller

.\.venv\Scripts\python scripts\make_icon.py
if (Test-Path build) { Remove-Item -Recurse -Force build }
if (Test-Path dist) { Remove-Item -Recurse -Force dist }
.\.venv\Scripts\pyinstaller --noconfirm --log-level WARN packaging\kol_studio.spec

iscc packaging\windows\installer.iss "/DAppVersion=$Version"
Get-FileHash "dist\KolStudio-$Version-windows-x64-setup.exe" -Algorithm SHA256
Write-Host "built: dist\KolStudio-$Version-windows-x64-setup.exe"
