$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
if (Test-Path $vswhere) {
    $vsPath = & $vswhere -latest -property installationPath
    if ($vsPath) {
        $vcvars = "$vsPath\VC\Auxiliary\Build\vcvarsall.bat"
        if (Test-Path $vcvars) {
            $batContent = "@echo off`r`ncall `"$vcvars`" x64`r`npy run_stress_tests.py"
            Set-Content -Path run_with_msvc.bat -Value $batContent
            cmd.exe /c run_with_msvc.bat
        } else {
            Write-Output "Error: vcvarsall.bat not found at $vcvars"
        }
    } else {
        Write-Output "Error: VS installation path not found by vswhere"
    }
} else {
    Write-Output "Error: vswhere.exe not found. Is Visual Studio installed?"
}
