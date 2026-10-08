# Build TECHNICAL_REPORT.md into a styled standalone HTML file.
# No LaTeX required. Works with pandoc alone.

$src = "docs\TECHNICAL_REPORT.md"
$out = "docs\TECHNICAL_REPORT.html"

if (-not (Get-Command pandoc -ErrorAction SilentlyContinue)) {
    Write-Host "pandoc not found."
    exit 1
}

Write-Host "Building $out ..."
pandoc $src `
    -o $out `
    --standalone `
    --toc `
    --toc-depth=3 `
    --number-sections `
    --highlight-style=tango `
    --metadata title="π-Causal Sandbox: Technical Report" `
    --css=https://cdn.jsdelivr.net/npm/water.css@2/out/water.css

Write-Host "Done: $out"
Write-Host "Open in browser: file:///$((Resolve-Path $out).Path -replace '\\','/')"