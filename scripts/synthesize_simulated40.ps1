param([string]$ManifestFile = '')
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$taskRoot = Split-Path -Parent $PSScriptRoot
$manifestPath = Join-Path $taskRoot 'data/simulated/pilot40_20261026/private/manifest.json'
if ($ManifestFile) { $manifestPath = $ManifestFile }
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $synth.SelectVoice('Microsoft Zira Desktop')
    foreach ($sample in $manifest.samples) {
        if (Test-Path -LiteralPath $sample.raw_audio) { continue }
        $synth.SetOutputToWaveFile($sample.raw_audio)
        $synth.SpeakSsml($sample.ssml)
        $synth.SetOutputToNull()
        Write-Output $sample.id
    }
} finally {
    $synth.Dispose()
}
