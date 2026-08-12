$ErrorActionPreference = 'Stop'

$tagPrefix = 'v'
$configPath = Join-Path $PSScriptRoot 'release.json'
if (Test-Path $configPath) {
    $config = Get-Content $configPath -Raw | ConvertFrom-Json
    if ($config.tagPrefix) { $tagPrefix = [string]$config.tagPrefix }
}

$tags = @(git tag --list "$tagPrefix*" --sort=-v:refname)
if ($LASTEXITCODE -ne 0) { throw 'Unable to list Git tags.' }
$lastTag = if ($tags.Count -gt 0) { $tags[0].Trim() } else { $null }

if ($lastTag) {
    $changedFiles = @(git diff --name-only "$lastTag..HEAD")
} else {
    $changedFiles = @(git ls-files)
}
if ($LASTEXITCODE -ne 0) { throw 'Unable to determine changed files.' }

$packagePatterns = @(
    '^src/OnnxModelRuntime/',
    '^Directory\.Build\.props$',
    '^Directory\.Packages\.props$',
    '^global\.json$',
    '^README\.md$',
    '^128x128_compressed\.png$',
    '^LICENSE$'
)

$impacting = @($changedFiles | Where-Object {
    $path = $_
    $packagePatterns | Where-Object { $path -match $_ } | Select-Object -First 1
})

$shouldRelease = $impacting.Count -gt 0
Write-Host "Previous release tag: $($lastTag ?? '<none>')"
Write-Host "Package-affecting files: $($impacting.Count)"
$impacting | ForEach-Object { Write-Host "  $_" }

if ($env:GITHUB_OUTPUT) {
    "should_release=$($shouldRelease.ToString().ToLowerInvariant())" | Out-File -FilePath $env:GITHUB_OUTPUT -Encoding utf8 -Append
    "last_tag=$lastTag" | Out-File -FilePath $env:GITHUB_OUTPUT -Encoding utf8 -Append
}
