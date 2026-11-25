param(
    [string]$OpenCv = 'C:\opencv\build',
    [string]$UrRtde = "$env:ProgramFiles\ur_rtde",
    [string]$Boost = 'C:\local\boost_1_75_0'
)

$ErrorActionPreference = 'Stop'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) { throw 'Visual Studio Installer (vswhere) is required.' }
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'Install the Visual Studio Desktop development with C++ workload.' }
$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
$OpenCv = (Resolve-Path -LiteralPath $OpenCv).Path
$UrRtde = (Resolve-Path -LiteralPath $UrRtde).Path
$Boost = (Resolve-Path -LiteralPath $Boost).Path

$build = Join-Path $PSScriptRoot 'build'
New-Item -ItemType Directory -Force -Path $build | Out-Null
$compile = @"
@echo off
call "$vcvars" || exit /b 1
cl /nologo /std:c++17 /EHsc /W4 /O2 /MD /DNDEBUG /D_WIN32_WINNT=0x0601 /DBOOST_ALL_NO_LIB /I"$OpenCv\include" /I"$UrRtde\include" /I"$Boost" "$PSScriptRoot\main.cpp" "$PSScriptRoot\vision.cpp" "$PSScriptRoot\planner.cpp" "$PSScriptRoot\robot.cpp" /Fe:domino.exe /link /LIBPATH:"$OpenCv\x64\vc15\lib" /LIBPATH:"$UrRtde\lib" opencv_world343.lib rtde.lib libboost_thread-vc142-mt-x64-1_75.lib libboost_system-vc142-mt-x64-1_75.lib libboost_chrono-vc142-mt-x64-1_75.lib ws2_32.lib mswsock.lib
"@
$batch = Join-Path $build 'compile.cmd'
Set-Content -LiteralPath $batch -Value $compile -Encoding Default
Push-Location $build
try {
    & $batch
    if ($LASTEXITCODE -ne 0) { throw "Build failed with exit code $LASTEXITCODE." }
    Copy-Item -LiteralPath "$OpenCv\x64\vc15\bin\opencv_world343.dll" -Destination $build -Force
} finally {
    Pop-Location
}
