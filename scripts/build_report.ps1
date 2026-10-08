# Build TECHNICAL_REPORT.md into PDF via pandoc on Windows.
# Requires: pandoc, MiKTeX (for xelatex)

$src = "docs\TECHNICAL_REPORT.md"
$out = "docs\TECHNICAL_REPORT.pdf"

if (-not (Get-Command pandoc -ErrorAction SilentlyContinue)) {
    Write-Host "pandoc not found. Install from https://pandoc.org/installing.html"
    exit 1
}

Write-Host "Building $out ..."
pandoc $src `
    -o $out `
    --pdf-engine=xelatex `
    --variable mainfont="DejaVu Sans" `
    --variable geometry:margin=1in `
    --toc `
    --number-sections

Write-Host "Done: $out"