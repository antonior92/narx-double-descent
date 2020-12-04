import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Plot performance as a function of the proportion '
                                                 'n features / n samples rate.')
    parser.add_argument('files', nargs='+',
                        help='input csv files.')
    parser.add_argument('--labels', nargs='*', default=[],
                        help='names to include in the legend.')
    parser.add_argument('--tp', default='pred', choices=['pred', 'sim', 'norm'],
                        help='input csv.')
    parser.add_argument('--split', default='test', choices=['train', 'test'],
                        help='split to plot.')
    parser.add_argument('--save', default='',
                        help='input csv.')
    parser.add_argument('--ymax', default=10, type=float,
                        help='max y-lim in the plot.')
    parser.add_argument('--xmax', default=-1, type=float,
                        help='max x-lim in the plot')
    parser.add_argument('--xmin', default=-1, type=float,
                        help='min x-lim in the plot')
    parser.add_argument('--plot_style', nargs='*', default=[],
                        help='plot styles to be used')
    parser.add_argument('--linear_scale', action='store_true',
                        help='use linear scale')
    args, unk = parser.parse_known_args()

    if args.plot_style:
        plt.style.use(args.plot_style)

    fig, ax = plt.subplots()
    for i, file in enumerate(args.files):
        try:
            label = args.labels[i]
        except:
            label = ''
        df = pd.read_csv(file)
        del df['mdl']
        del df['dset']
        df_baseline = df[df['proportion'] == 0]
        df_mdl = df[df['proportion'] > 0]
        q1 = df_mdl.groupby(axis=0, by='proportion').quantile(0.25)
        median = df_mdl.groupby(axis=0, by='proportion').quantile(0.5)
        q3 = df_mdl.groupby(axis=0, by='proportion').quantile(0.75)
        proportions = median.index
        s = 'mse_{}_{}'.format(args.tp, args.split)
        ax.plot(proportions, median[s], label=label)
        ax.fill_between(proportions, q1[s], q3[s], alpha=0.2)

    ax.set_ylim((-0.05, args.ymax))
    ax.axvline(1, ls='--', alpha=0.5)
    ax.set_ylabel('MSE')
    ax.set_xlabel('num features / num samples')
    if not args.linear_scale:
        ax.set_xscale('log')
    if args.xmax > 0:
        ax.set_xlim(right=args.xmax)
    if args.xmin > 0:
        ax.set_xlim(left=args.xmin)
    plt.legend()
    if args.save:
        plt.savefig(args.save)
    else:
        plt.show()