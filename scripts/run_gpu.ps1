# Executa argumentos Python no ambiente do projeto que tenha CUDA funcional.
# Prioriza .venv após ela apontar para o ambiente CUDA; LITOBENCH_GPU_PYTHON
# pode indicar outro Python CUDA como fallback.
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$externalPython = if ($env:LITOBENCH_GPU_PYTHON) {
    $env:LITOBENCH_GPU_PYTHON
} else {
    'D:\Codex\PythonEnvs\lithography-cu128\Scripts\python.exe'
}

# Cache, temporários, modelos e runtime ficam em D: só durante este processo.
$processEnvironment = [ordered]@{
    TEMP = 'D:\Codex\Temp\lithography'
    TMP = 'D:\Codex\Temp\lithography'
    PIP_CACHE_DIR = 'D:\Codex\Cache\pip'
    TORCH_HOME = 'D:\Codex\Cache\torch'
    HF_HOME = 'D:\Codex\Cache\huggingface'
    MPLCONFIGDIR = 'D:\Codex\Cache\matplotlib'
    JUPYTER_RUNTIME_DIR = 'D:\Codex\Temp\jupyter'
}
$oldEnvironment = @{}
foreach ($name in $processEnvironment.Keys) {
    $oldEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
$scriptExitCode = 1

try {
    foreach ($name in $processEnvironment.Keys) {
        $directory = $processEnvironment[$name]
        if (-not (Test-Path -LiteralPath $directory -PathType Container)) {
            throw "Diretório configurado ausente: $directory"
        }
        [Environment]::SetEnvironmentVariable($name, $directory, 'Process')
    }

    $pythonCandidates = @($projectPython, $externalPython) | Select-Object -Unique
    $selectedPython = $null
    $cudaProbe = 'import sys, torch; print("PyTorch: " + torch.__version__); print("CUDA build: " + str(torch.version.cuda)); ok = torch.cuda.is_available(); print("CUDA disponível: " + str(ok)); sys.exit(0 if ok else 2)'

    foreach ($candidate in $pythonCandidates) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            continue
        }

        Write-Host "Verificando CUDA em: $candidate"
        & $candidate -c $cudaProbe
        $probeExitCode = $LASTEXITCODE
        if ($probeExitCode -eq 0) {
            $selectedPython = $candidate
            break
        }
        Write-Warning "Este Python não confirmou CUDA disponível (código $probeExitCode)."
    }

    if (-not $selectedPython) {
        throw 'Nenhum Python CUDA funcional foi encontrado. Configure .venv\Scripts\python.exe ou LITOBENCH_GPU_PYTHON.'
    }

    if ($args.Count -eq 0) {
        throw 'Informe um módulo ou script Python, por exemplo: .\scripts\run_gpu.ps1 -m jupyter lab .\teste.ipynb'
    }

    & $selectedPython @args
    $scriptExitCode = $LASTEXITCODE
}
catch {
    [Console]::Error.WriteLine("Falha ao preparar execução GPU: {0}", $_.Exception.Message)
    $scriptExitCode = 1
}
finally {
    foreach ($name in $processEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $oldEnvironment[$name], 'Process')
    }
}

exit $scriptExitCode
