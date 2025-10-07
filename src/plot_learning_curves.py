import json
import glob
import matplotlib.pyplot as plt
import argparse
import os

def plot_learning_curves(indir):
    files = glob.glob(os.path.join(indir, 'res_*.json'))
    for noise in [0.0, 0.1, 0.3]:
        plt.figure()
        has_data = False
        for f in files:
            with open(f) as file:
                data = json.load(file)
            if data['meta']['noise'] == noise:
                opt = data['meta']['optimizer']
                prec = data['meta']['precision']
                seed = data['meta']['seed']
                losses = data['history']['train_loss']
                if losses:  # Check if losses exist
                    epochs = range(1, len(losses) + 1)
                    plt.plot(epochs, losses, label=f'{opt}_{prec}_seed{seed}')
                    has_data = True
        if has_data:
            plt.xlabel('Epoch')
            plt.ylabel('Train Loss')
            plt.title(f'Train Loss vs. Epochs (Noise {noise})')
            plt.legend()
            plt.savefig(os.path.join(indir, f'learning_curve_noise{int(noise*100)}.png'))
            plt.close()
        else:
            plt.close()
            print(f'No data for noise {noise} in {indir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', default='results/light')
    args = parser.parse_args()
    plot_learning_curves(args.indir)
