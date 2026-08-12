param(
    [Parameter(Mandatory = $true)][string]$Directory,
    [Parameter(Mandatory = $true)][string]$Version
)

$ErrorActionPreference = 'Stop'
$package = Join-Path $Directory "OnnxModelRuntime.NET.$Version.nupkg"
$symbols = Join-Path $Directory "OnnxModelRuntime.NET.$Version.snupkg"
if (-not (Test-Path $package)) { throw "Expected package was not produced: $package" }
if (-not (Test-Path $symbols)) { throw "Expected symbol package was not produced: $symbols" }

$temp = Join-Path ([System.IO.Path]::GetTempPath()) ("onnx-model-runtime-nupkg-" + [Guid]::NewGuid())
New-Item -ItemType Directory -Path $temp | Out-Null
try {
    [System.IO.Compression.ZipFile]::ExtractToDirectory($package, $temp)

    foreach ($required in @(
        'README.md',
        '128x128_compressed.png',
        'LICENSE',
        'lib/net10.0/OnnxModelRuntime.dll'
    )) {
        if (-not (Test-Path (Join-Path $temp $required))) { throw "Package is missing '$required'." }
    }

    $nuspec = Get-ChildItem $temp -Filter '*.nuspec' | Select-Object -First 1
    if (-not $nuspec) { throw 'Package nuspec is missing.' }
    [xml]$xml = Get-Content $nuspec.FullName -Raw
    $metadata = $xml.package.metadata
    if ([string]$metadata.id -ne 'OnnxModelRuntime.NET') { throw "Unexpected package id '$($metadata.id)'." }
    if ([string]$metadata.version -ne $Version) { throw "Unexpected package version '$($metadata.version)'." }
    if (-not $metadata.license) { throw 'Package is missing license metadata.' }
    if (-not $metadata.repository) { throw 'Package is missing repository metadata.' }
    if ([string]$metadata.readme -ne 'README.md') { throw 'NuGet README must be the root README.md.' }
    if ([string]$metadata.icon -ne '128x128_compressed.png') { throw 'NuGet icon must be the root 128x128_compressed.png.' }

    $modelWeights = Get-ChildItem $temp -Recurse -File | Where-Object {
        $_.Extension -in '.onnx', '.safetensors', '.gguf', '.bin' -and $_.FullName -notmatch '[\\/]lib[\\/]'
    }
    if ($modelWeights) { throw 'The managed NuGet unexpectedly contains model weights/test models.' }
} finally {
    Remove-Item $temp -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "Validated OnnxModelRuntime.NET $Version package contents."
