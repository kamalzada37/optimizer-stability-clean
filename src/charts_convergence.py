# charts_convergence.py
import json, glob
import pandas as pd
import matplotlib.pyplot as plt

files = glob.glob("results/light/res_*.json")
rows = []
for f in files:
    d = json.load(open(f))
    m = d["meta"]
    conv = d["history"].get("convergence_epoch", None)
    if conv:
        rows.append({
            "optimizer": m["optimizer"],
            "noise": m["noise"],
            "precision": m["precision"],
            "conv_epoch": conv
        })

df = pd.DataFrame(rows)

if not df.empty:
    df.boxplot(by=["optimizer","noise"], column="conv_epoch", grid=False)
    plt.title("Convergence Speed")
    plt.suptitle("")
    plt.ylabel("Epochs to reach loss < 0.1")
    plt.tight_layout()
    plt.savefig("results/light/convergence_boxplot.png")
