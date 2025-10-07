# run_grid_light.ps1 - small debug grid (run from project root with venv activated)
# Activate venv before running: `. .\.venv\Scripts\Activate.ps1`

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

$python = "python"   # uses venv python when venv is activated
$optimizers = @('sgd','adam')
$noises = @(0.0, 0.1, 0.3)
$precisions = @('float32','float64')
$seeds = @(0,1)
$epochs = 3
$outdir = "results\light"

# Ensure outdir exists
if (-not (Test-Path $outdir)) {
    New-Item -ItemType Directory -Path $outdir -Force | Out-Null
}

foreach ($opt in $optimizers) {
    foreach ($noise in $noises) {
        foreach ($prec in $precisions) {
            foreach ($seed in $seeds) {
                $lr = if ($opt -eq 'adam') { 0.001 } else { 0.01 }
                Write-Output "Run: $opt noise=$noise prec=$prec seed=$seed lr=$lr"
                & $python -m src.train --optimizer $opt --lr $lr --noise $noise --precision $prec --seed $seed --epochs $epochs --outdir $outdir --dataset mnist
            }
        }
    }
}

Write-Output "Running analysis..."
& $python -m src.analyze --indir $outdir
Write-Output "Script finished."
