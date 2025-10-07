# create_summary.py  (put in project root)
import json, glob, os, pandas as pd, argparse

def create_summary(indir='results/light'):
    files = glob.glob(os.path.join(indir, 'res_*.json'))
    rows = []
    for f in files:
        d = json.load(open(f))
        m = d['meta']
        rows.append({
            'optimizer': m['optimizer'],
            'noise': m['noise'],
            'precision': m['precision'],
            'seed': m['seed'],
            'final_acc': d['history'].get('test_acc', [None])[-1],
            'diverged': d.get('diverged', False)
        })
    df = pd.DataFrame(rows)
    os.makedirs(indir, exist_ok=True)
    df.to_csv(os.path.join(indir, 'summary.csv'), index=False)
    print("Saved", os.path.join(indir, 'summary.csv'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', default='results/light')
    args = parser.parse_args()
    create_summary(args.indir)
