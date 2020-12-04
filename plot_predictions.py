from datasets import *
from models import *
from narx_double_descent import train, evaluate, mse
from util import parse_funct_arguments
import pandas as pd
import pickle


def plot_predictions(u, y, y_pred, ax):
    ax.step(k, u, color='red')
    ax.set_xlabel('k')
    ax.set_ylabel('u')

    axt = ax.twinx()
    axt.plot(k, y, color='green', label='true')
    axt.plot(k[-len(y_pred):], y_pred, color='blue', label='pred.')
    axt.set_xlabel('k')
    axt.set_ylabel('y', rotation=-90)
    return ax


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Plot dataset input / output together '
                                                 'with model predictions')
    parser.add_argument('-d', '--dset', type=str, default='ChenDSet',
                        help='number of repetitions')
    parser.add_argument('-m', '--nonlinear_model', default='RBFSampler',
                        help='number of repetitions')
    parser.add_argument('--file', default='',
                        help='input csv with execution detail.')
    parser.add_argument('--i', default=7, type=int,
                        help='csv entry to use.')
    parser.add_argument('--mdls_folder', default='',
                        help='path to folder with pickled models.')
    parser.add_argument('--split', default='test', choices=['test', 'train'],
                        help='split to plot.')
    parser.add_argument('--tp', default='sim', choices=['sim', 'pred'],
                        help='select between free-run-simulation and one-step-ahead predition.')
    parser.add_argument('--save', default='',
                        help='save plot.')
    parser.add_argument('--plot_style', nargs='*', default=[],
                        help='plot styles to be used.')
    parser.add_argument('--sequence', default=0, type=int,
                        help='sequence selected for plotting (when more than one sequence is present).')
    parser.add_argument('--nth_input', default=0, type=int,
                        help='input selected for plotting (when more than one input is present).')
    parser.add_argument('--nth_output', default=0, type=int,
                        help='output selected for plotting (when more than one output is present).')
    args, unk = parser.parse_known_args()

    if args.plot_style:
        plt.style.use(args.plot_style)

    if args.file:  # If file is available read model and dataset from file
        print('Loading model and dataset specification from file...')
        df = pd.read_csv(args.file)
        entry = df.loc[args.i]
        mdl = eval(entry['mdl'])
        dset = eval(entry['dset'])
    else:   # otherwise read from command line
        print('Loading model and dataset specification from command line...')
        # Get model (from command line)
        ModelTmp = eval(args.nonlinear_model)
        Model, _, unk = parse_funct_arguments(ModelTmp, unk)
        mdl = Model()
        # Get dataset (from the command line)
        DatasetTmp = eval(args.dset)
        Dataset, _, unk = parse_funct_arguments(DatasetTmp, unk)
        dset = Dataset()
    print(mdl)
    print(dset)

    # Get predictions:
    sname = 'z_{}_{}'.format(args.tp, args.split)
    if args.mdls_folder:
        print('Loading model from file...')
        fname = os.path.join(args.mdls_folder, repr(mdl)+'.pkl')
        with open(fname, 'rb') as f:
            mdl = pickle.load(f)
    else:
        print('Training model from scratch...')
        mdl = train(mdl, dset)
    performance, pred_train, pred_test = evaluate(mdl, dset)
    y_pred = pred_train[sname] if args.split == 'train' else pred_test[sname]

    if args.split == 'train':
        u, y = dset.get_train()
    else:
        u, y = dset.get_test()
    k = np.arange(len(u))

    print('mse = {}'.format(mse(y[dset.sys.order:, ...], y_pred)))

    fig, ax = plt.subplots()

    plot_predictions(u[:, args.sequence, args.nth_input],
                     y[:, args.sequence, args.nth_output],
                     y_pred[:, args.sequence, args.nth_output], ax)
    plt.legend()
    if args.save:
        plt.savefig(args.save)
    else:
        plt.show()