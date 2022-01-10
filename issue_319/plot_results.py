import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def bool2color(hits: list):
    return list(map(lambda x: 'green' if x else 'red', hits))


def main(args):
    trues = []
    preds = []
    true_confs = []
    pred_confs = []
    for csv_path in args.csv_paths:
        df = pd.read_csv(csv_path)
        trues.append(df.true.values)
        preds.append(df.pred.values)
        true_confs.append(df.true_confidence.values)
        pred_confs.append(df.pred_confidence.values)

    n_samples = len(trues[0])
    if args.max_samples:
        n_samples = args.max_samples
        trues = list(map(lambda list: list[:n_samples], trues))
        preds = list(map(lambda list: list[:n_samples], preds))
        true_confs = list(map(lambda list: list[:n_samples], true_confs))
        pred_confs = list(map(lambda list: list[:n_samples], pred_confs))

    # Find the predictions that are correct
    hits_0 = preds[0] == trues[0]
    hits_1 = preds[1] == trues[0]
    hits_total = hits_0 & hits_1

    # Convert the mask of hits to colors names
    hits_0 = bool2color(hits_0)
    hits_1 = bool2color(hits_1)
    hits_total = bool2color(hits_total)

    # Prepare the plots folder
    exp_name = args.experiment_name
    if args.max_samples:
        exp_name += f'_N-{args.max_samples}'
    exp_plots_path = os.path.join(args.plots_path, exp_name)
    if not os.path.exists(exp_plots_path):
        print(f"Creating the '{exp_plots_path}' directory to store the plots")
        os.makedirs(exp_plots_path)

    plt.scatter(pred_confs[0], pred_confs[1], c=hits_total, alpha=0.5)
    plt.grid(linestyle='--', linewidth=0.5)
    plt.xlabel(args.labels[0])
    plt.ylabel(args.labels[1])
    plt.title("Pred. confidence")
    plot_name = f"{exp_name}_plot0.png"
    plt.savefig(os.path.join(exp_plots_path, plot_name),
                bbox_inches='tight', dpi=1200)
    plt.clf()

    range_ = np.arange(n_samples)
    plt.scatter(range_, pred_confs[0], c=hits_0, marker='o', alpha=0.5)
    plt.scatter(range_, pred_confs[1], c=hits_1, marker='^', alpha=0.5)
    plt.xticks(np.arange(n_samples, step=0.1 * n_samples))
    plt.legend(args.labels)
    plt.grid(linestyle='--', linewidth=0.5)
    plt.xlabel('Sample')
    plt.ylabel('Pred. confidence')
    plt.title("Pred. confidence")
    plot_name = f"{exp_name}_plot1.png"
    plt.savefig(os.path.join(exp_plots_path, plot_name),
                bbox_inches='tight', dpi=1200)
    plt.clf()

    plt.scatter(true_confs[0], true_confs[1], c=hits_total, alpha=0.5)
    plt.grid(linestyle='--', linewidth=0.5)
    plt.xlabel(args.labels[0])
    plt.ylabel(args.labels[1])
    plt.title("True confidence")
    plot_name = f"{exp_name}_plot2.png"
    plt.savefig(os.path.join(exp_plots_path, plot_name),
                bbox_inches='tight', dpi=1200)
    plt.clf()

    range_ = np.arange(n_samples)
    plt.scatter(range_, true_confs[0], c=hits_0, marker='o', alpha=0.5)
    plt.scatter(range_, true_confs[1], c=hits_1, marker='^', alpha=0.5)
    plt.xticks(np.arange(n_samples, step=0.1 * n_samples))
    plt.legend(args.labels)
    plt.grid(linestyle='--', linewidth=0.5)
    plt.xlabel('Sample')
    plt.ylabel('Pred. confidence')
    plt.title("True confidence")
    plot_name = f"{exp_name}_plot3.png"
    plt.savefig(os.path.join(exp_plots_path, plot_name),
                bbox_inches='tight', dpi=1200)
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Auxiliar script to create plots from CSV data')
    parser.add_argument('--experiment-name', '-exp-name',
                        help='Name of the experiment to show',
                        required=True,
                        type=str,
                        metavar='exp_name')
    parser.add_argument('--labels',
                        help='Names to show in the plot legend',
                        required=True,
                        type=str,
                        nargs=2,
                        metavar='name')
    parser.add_argument('--csv-paths',
                        help='Paths to the CSVs to extract data from',
                        required=True,
                        type=str,
                        nargs=2,
                        metavar='Path')
    parser.add_argument('--max-samples',
                        help='Maximum number of samples to take for the plots',
                        required=False,
                        type=int,
                        metavar='N')
    parser.add_argument('--plots-path',
                        help='Path to the folder to store the plots',
                        required=False,
                        type=str,
                        default='plots',
                        metavar='plots_dir_path')

    main(parser.parse_args())
