import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Plot performance as a function of the proportion '
                                                 'n features / n samples rate.')
    parser.add_argument('file', default='./performance.csv',
                        help='input csv.')
    parser.add_argument('--tp', default='pred', choices=['pred', 'sim', 'norm'],
                        help='input csv.')
    parser.add_argument('--save', default='',
                        help='input csv.')
    parser.add_argument('--xticks', default=[], type=float, nargs='+',
                        help='input csv.')
    parser.add_argument('--ymax', default=0, type=float,
                        help='max y-lim in the plot.')
    parser.add_argument('--xmax', default=-1, type=float,
                        help='max x-lim in the plot')
    parser.add_argument('--xmin', default=-1, type=float,
                        help='min x-lim in the plot')
    parser.add_argument('--plot_style', nargs='*', default=[],
                        help='plot styles to be used')
    parser.add_argument('--xlinear_scale', action='store_true',
                        help='use linear scale in x-axis. By default use log scale in x.')
    parser.add_argument('--ylog_scale', action='store_true',
                        help='use log scale in x-axis. By default use linear scale in x.')
    parser.add_argument('--dont_plot_baseline', action='store_true',
                        help='dont plot baseline')
    parser.add_argument('--omit_legend', action='store_true',
                        help='dont show legend')
    args, unk = parser.parse_known_args()

    if args.plot_style:
        plt.style.use(args.plot_style)

    df = pd.read_csv(args.file)
    del df['mdl']
    del df['dset']
    df_baseline = df[df['proportion'] == 0]
    df_mdl = df[df['proportion'] > 0]

    q1 = df_mdl.groupby(axis=0, by='proportion').quantile(0.25)
    median = df_mdl.groupby(axis=0, by='proportion').quantile(0.5)
    q3 = df_mdl.groupby(axis=0, by='proportion').quantile(0.75)
    proportions = median.index
    fig, ax = plt.subplots()
    if args.tp in ['pred', 'sim']:
        risk_baseline = np.mean(df_baseline['mse_{}_test'.format(args.tp)])
        s = 'mse_{}_test'.format(args.tp)
        plt.plot(proportions, median[s], label='test')
        plt.fill_between(proportions, q1[s], q3[s], alpha=0.2)
        if not args.ylog_scale:
            plt.ylim(bottom=-0.05)
        s = 'mse_{}_train'.format(args.tp)
        plt.plot(proportions, median[s], label='train')
        plt.fill_between(proportions, q1[s], q3[s], alpha=0.2)
        if not args.dont_plot_baseline:
            ax.axhline(risk_baseline, ls='--', label='baseline')
        ax.axvline(1, ls='--', alpha=0.5)
        plt.ylabel('MSE')
    elif args.tp == 'norm':
        s = 'param_norm'
        plt.plot(proportions, median[s])
        plt.fill_between(proportions, q1[s], q3[s], alpha=0.2)
        plt.ylabel(r'$\|\theta\|$')
        ax.axvline(1, ls='--', alpha=0.5)
        plt.yscale('log')
    plt.xlabel('num features / num samples')
    if not args.xlinear_scale:
        plt.xscale('log')
    if args.ylog_scale:
        plt.yscale('log')
    if args.xticks:
        plt.xticks(args.xticks)
    if args.ymax > 0:
        plt.ylim(top=args.ymax)
    if args.xmax > 0:
        plt.xlim(right=args.xmax)
    if args.xmin > 0:
        plt.xlim(left=args.xmin)
    if not args.omit_legend:
        plt.legend()
    if args.save:
        plt.savefig(args.save)
    else:
        plt.show()