# src/analyze.py
"""
Analyze results directory and produce:
  - results summary CSV (summary.csv)
  - precision_gap CSV (precision_gap.csv)
  - bar plots plot_noise{n}.png
  - learning curves learning_curve_noise{n}.png

Usage:
  python -m src.analyze --indir results/light
"""
import os, glob, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_results(results_dir):
    files = glob.glob(os.path.join(results_dir, "res_*.json"))
    rows = []
    for f in files:
        try:
            r = json.load(open(f))
        except Exception:
            continue
        m = r.get('meta', {})
        dataset = m.get('dataset', 'mnist')
        history = r.get('history', {})
        final_acc = None
        final_loss = None
        if dataset == 'mnist' and history.get('test_acc'):
            final_acc = history['test_acc'][-1]
        if dataset != 'mnist' and history.get('test_loss'):
            final_loss = history['test_loss'][-1]
        rows.append({
            'file': f,
            'dataset': dataset,
            'optimizer': m.get('optimizer'),
            'lr': m.get('lr'),
            'noise': m.get('noise'),
            'precision': m.get('precision'),
            'seed': m.get('seed'),
            'final_acc': final_acc,
            'final_loss': final_loss,
            'history': history,
            'diverged': r.get('diverged', False),
            'elapsed_sec': r.get('elapsed_sec', None)
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

def save_summary(df, out_csv):
    # group by optimizer, noise, precision and compute mean/std/count
    agg_col = 'final_acc' if df['dataset'].iloc[0] == 'mnist' else 'final_loss'
    g = df.dropna(subset=[agg_col]).groupby(['optimizer','noise','precision'])
    summ = g[agg_col].agg(['mean','std','count']).reset_index()
    summ = summ.rename(columns={'mean':'mean_acc' if agg_col=='final_acc' else 'mean_loss',
                                'std':'std_acc' if agg_col=='final_acc' else 'std_loss',
                                'count':'n'})
    summ.to_csv(out_csv, index=False)
    print("Saved", out_csv)
    return summ

def compute_precision_gap(df, out_csv):
    # pivot to get float32/float64 mean per (optimizer, noise)
    df_f = df.dropna(subset=['final_acc'])
    pivot = df_f.pivot_table(index=['optimizer','noise'], columns='precision', values='final_acc', aggfunc='mean')
    if pivot.empty:
        print("No final_acc data for precision gap.")
        return None
    pivot = pivot.reset_index()
    # compute gap float64 - float32 if both present
    if 'float64' in pivot.columns and 'float32' in pivot.columns:
        pivot['gap'] = pivot['float64'] - pivot['float32']
    else:
        pivot['gap'] = np.nan
    pivot.to_csv(out_csv, index=False)
    print("Saved", out_csv)
    return pivot

def plot_bars(df, out_dir):
    # for each noise, plot mean +/- std bar chart
    df_clean = df.dropna(subset=['final_acc'])
    if df_clean.empty:
        print("No final_acc to plot.")
        return
    for noise in sorted(df_clean['noise'].unique()):
        sub = df_clean[df_clean['noise'] == noise]
        g = sub.groupby(['optimizer','precision'])['final_acc'].agg(['mean','std']).unstack(fill_value=np.nan)
        # create paired bar chart where groups = optimizers, columns = precision
        fig, ax = plt.subplots(figsize=(6,4))
        # reshape for plotting
        means = []
        errs = []
        labels = []
        precisions = sorted(sub['precision'].unique())
        for opt in sorted(sub['optimizer'].unique()):
            labels.append(opt)
            row_means = []
            row_errs = []
            for prec in precisions:
                val = sub[(sub['optimizer']==opt) & (sub['precision']==prec)]['final_acc'].mean()
                err = sub[(sub['optimizer']==opt) & (sub['precision']==prec)]['final_acc'].std()
                row_means.append(val)
                row_errs.append(err if not np.isnan(err) else 0.0)
            means.append(row_means)
            errs.append(row_errs)
        means = np.array(means)
        errs = np.array(errs)
        # bar positions
        x = np.arange(len(labels))
        width = 0.35
        for i, prec in enumerate(precisions):
            pos = x + (i - (len(precisions)-1)/2) * (width)
            ax.bar(pos, means[:, i], width=width, yerr=errs[:, i], label=str(prec), capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25)
        ax.set_ylabel('Final accuracy' if df_clean['dataset'].iloc[0]=='mnist' else 'Final loss')
        ax.set_title(f'Final metric (noise={noise})')
        ax.legend(title='precision')
        plt.tight_layout()
        fname = os.path.join(out_dir, f'plot_noise{int(noise*100)}.png')
        plt.savefig(fname)
        plt.close()
        print("Saved", fname)

def plot_learning_curves(df, out_dir):
    files = df['file'].tolist()
    for noise in sorted(df['noise'].unique()):
        plt.figure(figsize=(6,4))
        plotted = False
        for _, row in df[df['noise']==noise].iterrows():
            hist = row['history']
            if not hist:
                continue
            losses = hist.get('train_loss', [])
            if not losses:
                continue
            epochs = range(1, len(losses)+1)
            label = f"{row['optimizer']}_{row['precision']}_s{row['seed']}"
            plt.plot(epochs, losses, label=label, linewidth=0.8)
            plotted = True
        if plotted:
            plt.xlabel('Epoch')
            plt.ylabel('Train loss')
            plt.title(f'Train loss (noise {noise})')
            plt.legend(fontsize='small', ncol=2)
            plt.tight_layout()
            fname = os.path.join(out_dir, f'learning_curve_noise{int(noise*100)}.png')
            plt.savefig(fname)
            plt.close()
            print("Saved", fname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', default='results/light', help='results directory with res_*.json files')
    args = parser.parse_args()
    df = load_results(args.indir)
    if df.empty:
        print("No results found in", args.indir, ". Run experiments first.")
        exit(0)
    # save detailed summary
    out_summary = os.path.join(args.indir, 'summary.csv')
    save_summary(df, out_summary)
    out_precision = os.path.join(args.indir, 'precision_gap.csv')
    compute_precision_gap(df, out_precision)
    plot_bars(df, args.indir)
    plot_learning_curves(df, args.indir)
    print("Analysis complete.")
