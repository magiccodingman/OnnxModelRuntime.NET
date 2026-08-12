$ErrorActionPreference = 'Stop'

$config = Get-Content (Join-Path $PSScriptRoot 'release.json') -Raw | ConvertFrom-Json
$initialVersion = [System.Version]::Parse([string]$config.initialVersion)
$prefix = [string]$config.tagPrefix
$increment = [string]$config.defaultIncrement
if ($increment -ne 'patch') { throw "Unsupported defaultIncrement '$increment'. Only patch is currently supported." }

function Get-MaxVersion([System.Version[]] $versions) {
    if (-not $versions -or $versions.Count -eq 0) { return $null }
    return ($versions | Sort-Object -Descending | Select-Object -First 1)
}

$gitVersions = @()
foreach ($tag in @(git tag --list "$prefix*")) {
    $raw = $tag.Trim()
    if ($raw.StartsWith($prefix)) { $raw = $raw.Substring($prefix.Length) }
    $parsed = $null
    if ([System.Version]::TryParse($raw, [ref]$parsed)) { $gitVersions += $parsed }
}

$nugetVersions = @()
foreach ($packageId in $config.packageIds) {
    $id = ([string]$packageId).ToLowerInvariant()
    $uri = "https://api.nuget.org/v3-flatcontainer/$id/index.json"
    try {
        $response = Invoke-RestMethod -Uri $uri -Method Get
        foreach ($rawVersion in $response.versions) {
            if ([string]$rawVersion -match '^[0-9]+\.[0-9]+\.[0-9]+$') {
                $parsed = $null
                if ([System.Version]::TryParse([string]$rawVersion, [ref]$parsed)) { $nugetVersions += $parsed }
            }
        }
    }
    catch {
        if ($_.Exception.Response -and [int]$_.Exception.Response.StatusCode -eq 404) {
            Write-Host "NuGet package '$packageId' is not published yet."
        } else {
            Write-Warning "Could not query NuGet version history for '$packageId': $($_.Exception.Message)"
        }
    }
}

$gitMax = Get-MaxVersion $gitVersions
$nugetMax = Get-MaxVersion $nugetVersions
$floorCandidates = @($gitMax, $nugetMax) | Where-Object { $_ -ne $null }

if ($floorCandidates.Count -eq 0) {
    $next = $initialVersion
} else {
    $floor = Get-MaxVersion $floorCandidates
    $next = [System.Version]::new($floor.Major, $floor.Minor, $floor.Build + 1)
}

$version = "$($next.Major).$($next.Minor).$($next.Build)"
Write-Host "Resolved release version: $version"

foreach ($packageId in $config.packageIds) {
    $id = ([string]$packageId).ToLowerInvariant()
    $uri = "https://api.nuget.org/v3-flatcontainer/$id/$version/$id.$version.nupkg"
    try {
        $response = Invoke-WebRequest -Uri $uri -Method Head -SkipHttpErrorCheck
        if ($response.StatusCode -eq 200) {
            throw "Package $packageId $version already exists on NuGet.org."
        }
    }
    catch {
        if ($_.Exception.Message -like 'Package * already exists*') { throw }
    }
}

if ($env:GITHUB_OUTPUT) {
    "version=$version" | Out-File -FilePath $env:GITHUB_OUTPUT -Encoding utf8 -Append
}
