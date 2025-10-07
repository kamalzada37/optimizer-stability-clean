# optimizer-stability
Code for "Numerical Optimization in Machine Learning: Stability and Robustness of Optimizers Under Noise and Precision Constraints"

## Quick start
1. Create a virtual environment:
python -m venv .venv
..venv\Scripts\Activate.ps1
pip install -r requirements.txt
2. Run a single experiment:


python -m src.train --optimizer adam --lr 0.001 --noise 0.0 --precision float32 --seed 0 --epochs 3 --outdir results/light --dataset mnist

3. Analyze results:


python -m src.analyze --indir results/light


Do NOT commit `.venv`, `results/`, or `data/`.
