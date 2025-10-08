# Optimizer Stability Research

This repository contains the code and experiments for our research on **Numerical Optimization Stability and Robustness in Deep Learning**.  
The study investigates how different optimizers — such as **SGD**, **Adam**, and **Sharpness-Aware Minimization (SAM)** — perform under varying levels of label noise and numerical precision (`float32` vs `float64`).


## Research Overview

Training deep neural networks involves complex optimization landscapes that can lead to unstable convergence, especially under noisy or low-precision conditions.  
This project systematically evaluates how optimizers behave when subjected to:

- **Label Noise:** Random mislabeling of training data (0%, 10%, 30%)  
- **Precision Constraints:** Comparing single-precision (`float32`) vs double-precision (`float64`) arithmetic  
- **Different Optimizers:** 
  - **SGD (Stochastic Gradient Descent)** — baseline method  
  - **Adam** — adaptive learning rate optimization  
  - **SAM (Sharpness-Aware Minimization)** — modern method improving flatness and generalization

Our goal is to analyze **accuracy stability**, **training behavior**, and **resilience to noise and numeric errors** across optimizers.


##  Methods and Formulae

The training objective is the standard supervised classification loss with optional noise:

\[
\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} y_i \log(\hat{y}_i)
\]

where \( \hat{y}_i = f_\theta(x_i) \) are the model predictions.

For **SAM (Sharpness-Aware Minimization)**, the optimizer perturbs the weights in the direction of the gradient before updating:

\[
\epsilon = \rho \frac{\nabla_\theta \mathcal{L}(\theta)}{ \| \nabla_\theta \mathcal{L}(\theta) \|_2 }
\]

\[
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta + \epsilon)
\]

where:
- \( \rho \) is the neighborhood radius (e.g., 0.05)  
- \( \eta \) is the learning rate  

This method seeks **flat minima**, improving robustness to noise and generalization stability.


##  Dataset

We used the **MNIST dataset** of handwritten digits (60,000 training, 10,000 test).  
It is loaded automatically via `torchvision.datasets.MNIST`, and artificial label noise is applied by randomly flipping a percentage of labels according to the specified noise level.

| Dataset | Samples | Classes | Purpose |

|----------|----------|----------|----------|

| MNIST | 60,000 train / 10,000 test | 10 | Stability & noise evaluation |


##  Experimental Setup

Experiments were run using PowerShell automation scripts to explore multiple configurations:

| Parameter | Values |

|------------|---------|

| Optimizers | `sgd`, `adam`, `sam_sgd`, `sam_adam` |

| Label noise | `0.0`, `0.1`, `0.3` |

| Precision | `float32`, `float64` |

| Seeds | `0, 1, 2, 3, 4` |

| Epochs | 3–20 |

| Learning rate | 0.01 (SGD), 0.001 (Adam) |

Script example (`run_grid_light.ps1`):

python -m src.train --optimizer adam --lr 0.001 --noise 0.0 --precision float32 --seed 0 --epochs 3 --outdir results/light --dataset mnist
Results and Plots
All generated outputs (CSV summaries, accuracy plots, and training curves) are saved under:


results/light/
File	Description
summary.csv	Aggregated results (mean/std accuracy per configuration)
precision_gap.csv	Accuracy gap between float32 and float64
plot_noise0.png	Final test accuracy at 0% noise
learning_curve_noise0.png	Training/test accuracy vs epochs at 0% noise
learning_curve_noise10.png	Same for 10% noise

 Project Structure

optimizer-stability/
│
├── src/

│   ├── train.py          # Core training loop and model definition

│   ├── analyze.py        # Aggregates results and generates charts

│   └── models/           # CNN architectures used

│
├── results/              # Generated experiment outputs

├── run_grid.ps1          # Full grid of experiments

├── run_grid_light.ps1    # Small debug grid (quick tests)

├── requirements.txt      # Python dependencies

├── README.md             # Project overview

└── .gitignore

 Quick Start
1. Clone and create environment

git clone https://github.com/kamalzada37/optimizer-stability-clean.git
cd optimizer-stability-clean
python -m venv .venv
.\.venv\Scripts\Activate.ps1
2. Install dependencies

# CPU-only installation
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
3. Run training

python -m src.train --optimizer sgd --lr 0.01 --noise 0.1 --precision float32 --seed 0 --epochs 3 --outdir results/light --dataset mnist
4. Analyze results

python -m src.analyze --indir results/light

# Citation
If you use this code or results, please cite our work:

Kamal Zada, M. & ParsaKarimi, Z. , (2025). "Numerical Optimization in Machine Learning: Stability and Robustness of Optimizers Under Noise and Precision Constraints."

🧰 License
This project is released under the MIT License — feel free to use, modify, and build upon it for academic or research purposes.

## 📚 Citation (BibTeX)
If you use this repository or build upon this research, please cite it as:

@misc{Parsa & Kamal Zada2025optimizerstability,
  title  = {Numerical Optimization in Machine Learning: Stability and Robustness of Optimizers Under Noise and Precision Constraints},
  authors = {Mustafa Kamal Zada & Ziaulhaq ParsaKarimi},
  year   = {2025},
  howpublished = {\url{https://github.com/kamalzada37/optimizer-stability-clean}},
  note   = {Version 1.0, GitHub repository}
}

🤝 Acknowledgments
PyTorch for deep learning framework

Torchvision for datasets and transforms

MNIST Dataset by Yann LeCun et al.

Research supervision and academic support from the contributing authors

