import os, pandas as pd

csv_path = os.path.join('results','light','summary.csv')
df = pd.read_csv(csv_path)

pivot = df.pivot_table(
    index=['optimizer','noise'],
    columns='precision',
    values='mean_acc',
    aggfunc='mean'
)

# Only compute gap if both columns exist
if 'float32' in pivot.columns and 'float64' in pivot.columns:
    pivot['gap'] = pivot['float64'] - pivot['float32']
else:
    pivot['gap'] = None
    print("⚠️ Warning: Missing float32/float64 rows in summary.csv")

out_path = os.path.join('results','light','precision_gap.csv')
pivot.to_csv(out_path)
print(f"Saved {out_path}")
print(pivot)
