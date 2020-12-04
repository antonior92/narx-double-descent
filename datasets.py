import numpy as np
import scipy.signal as sgn
import itertools
from collections.abc import Iterable
import pandas as pd
import os


def expand_list_of_lists(lofl):
    return list(itertools.chain(*lofl))


def back_to_original_shape(y, n_seq, n_out):
    return y.reshape(-1, n_seq, n_out)


def prepare_ns(ns):
    ns = list(ns)
    if isinstance(ns[0], Iterable):
        return expand_list_of_lists([[(inp, i) for i in ns_] for inp, ns_ in enumerate(ns)])
    else:
        return [(0, i) for i in ns]


def get_order(nus, nys):
    return max([nys for _inp, nys in nys] + [nus for _inp, nus in nus])


def construct_linear_system(u, y, nus, nys):
    seq_len, n_seq, out_dim = y.shape
    nys = prepare_ns(nys)
    nus = prepare_ns(nus)
    order = get_order(nus, nys)
    regressors = []
    for inp, ny in nys:
        regressors.append(y[order - ny:seq_len - ny, :, inp].flatten())
    for inp, nu in nus:
        regressors.append(u[order - nu:seq_len - nu, :, inp].flatten())
    X = np.stack(regressors, axis=1)
    y = y[order:, ...].reshape(-1, out_dim)
    return X, y.squeeze()


# --- Generic Dynamic System ---
class DynamicalSystem(object):
    def __init__(self, nys, nus, fn, sd_v=0.1, sd_w=0.0):
        self.nys = prepare_ns(nys)
        self.nus = prepare_ns(nus)
        self.fn = fn
        self.sd_v = sd_v
        self.sd_w = sd_w

    @property
    def order(self):
        return get_order(self.nys, self.nus)

    @property
    def input_dim(self):
        return max([inp+1 for inp, _nys in self.nys])

    @property
    def out_dim(self):
        return max([inp+1 for inp, _nus in self.nus])

    def prepare_z(self, u, y, k):
        ys = [y[k - i, :, inp] for inp, i in self.nys]
        us = [u[k - i, :, inp] for inp, i in self.nus]
        return np.stack(ys + us, axis=-1)

    def __call__(self, u, y0=None, seed=0):
        rng = np.random.RandomState(seed)
        seq_len, n_seq, inp_dim = u.shape
        y = np.zeros((seq_len, n_seq, self.out_dim))
        w = self.sd_w * rng.randn(*y.shape)
        v = self.sd_v * rng.randn(*y.shape)
        if y0 is not None:
            y[:self.order,...] = y0[:self.order, ...]
        for k in range(self.order+1, seq_len):
            z = self.prepare_z(u, y, k)
            y[k, ...] = np.apply_along_axis(self.fn, 1, z).reshape(n_seq, self.out_dim) + v[k, ...]
        return y + w


# --- Input generator ---
class RandomInput(object):
    def __init__(self, sd=1.0, hold=5, cutoff_freq=1.0, input_dim=1):
        self.sd = sd
        self.hold = hold
        self.input_dim = input_dim
        self.cutoff_freq = cutoff_freq
        if cutoff_freq < 1.0:
            self.sos = sgn.ellip(8, 0.1, 60, cutoff_freq, output='sos')
        else:
            self.sos = None

    def __call__(self, n, n_sequences=1, seed=0):
        rng = np.random.RandomState(seed)
        n_individual = int(np.ceil(n / self.hold))
        u = np.repeat(rng.normal(0, self.sd, (n_individual, n_sequences, self.input_dim)), self.hold, axis=0)
        u = u[:n, ...]
        if self.sos is not None:
            u = sgn.sosfiltfilt(self.sos, u, axis=0)
        return u


# ---- Datasets ----
class SimulatedDSet(object):
    def __init__(self, num_train_samples: int = 100, num_test_samples: int = 100, sd_v: float = 0.1,
                 sd_w: float = 0,  sd_u: float = 1.0, hold: int = 1, cutoff_freq: float = 1.0, seed: int = 1):
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples

        self.inp = RandomInput(sd_u, hold, cutoff_freq)
        self.sys = DynamicalSystem(nys=[1, 2], nus=[1, 2], fn=self.fn, sd_v=sd_v, sd_w=sd_w)

        self.seed = seed
        self.train_seed_noise = 4*seed
        self.train_seed_input = 4*seed+1
        self.test_seed_noise = 4*seed+2
        self.test_seed_input = 4*seed+3

    def __repr__(self):
        return '{}({},{},{},{},{},{},{},{})'.format(
            type(self).__name__, self.num_train_samples,  self.num_test_samples,
            self.sys.sd_v, self.sys.sd_w, self.inp.sd, self.inp.hold, self.inp.cutoff_freq,
            self.seed)

    def get_train(self):
        u = self.inp(self.num_train_samples, seed=self.train_seed_input)
        y = self.sys(u, seed=self.train_seed_noise)
        return u, y

    def get_test(self):
        u = self.inp(self.num_test_samples, seed=self.test_seed_noise)
        y = self.sys(u, seed=self.test_seed_input)
        return u, y

    @property
    def nys(self):
        return NotImplementedError()

    @property
    def nus(self):
        return NotImplementedError()

    def fn(self, x):
        return NotImplementedError()

    @property
    def effective_num_train_samples(self):
        nu_max = max(self.nus)
        ny_max = max(self.nus)
        return self.num_train_samples - max(nu_max, ny_max)


class ChenDSet(SimulatedDSet):
    @property
    def nys(self):
        return [1, 2]

    @property
    def nus(self):
        return [1, 2]

    def fn(self, x):
        y1, y2, u1, u2 = x
        return (0.8 - 0.5 * np.exp(-y1 ** 2)) * y1 - (0.3 + 0.9 * np.exp(-y1 ** 2)) * y2 + u1 + 0.2 * u2 + 0.1 * u1 * u2


class Order2LinearDSet(SimulatedDSet):
    @property
    def nys(self):
        return [1, 2]

    @property
    def nus(self):
        return [1, 2]

    def fn(self, x):
        y1, y2, u1, u2 = x
        return 1.5*y1 - 0.7*y2 + u1 + 0.5*u2


class CoupledElectricalDrives(object):

    def __init__(self, dset_choice: str = 'all', dset_path: str = '../data/coupled_electric_drives',
                 valid_split: int = 0.4):
        self.dset_choice = dset_choice
        self.dset_path = dset_path
        self.valid_split = valid_split
        # Define what to use according to dset_choice
        dset_choice = [dset_choice.lower()] if dset_choice.lower() in ['prbs', 'unif'] else ['prbs', 'unif']
        u_names, y_names = [], []
        self.n_sequences = 0
        if 'prbs' in dset_choice:
            u_names += ['u1', 'u2', 'u3']
            y_names += ['z1', 'z2', 'z3']
            self.n_sequences += 3
        if 'unif' in dset_choice:
            u_names += ['u11', 'u12']
            y_names += ['z11', 'z12']
            self.n_sequences += 2
        # Get dataframe
        paths = [os.path.join(dset_path, 'DATA{}.csv'.format(d.upper())) for d in dset_choice]
        df = pd.concat([pd.read_csv(p) for p in paths], axis=1)
        # Get data tensors
        self.u = np.array(df[u_names].values)[:, :, None]
        self.y = np.array(df[y_names].values)[:, :, None]
        # Get number of training samples
        self.n_train = int(np.ceil(self.u.shape[0] * (1 - valid_split)))

    def get_train(self):
        return self.u[:self.n_train, :, :], self.y[:self.n_train, :, :]

    def get_test(self):
        return self.u[self.n_train:, :, :], self.y[self.n_train:, :, :]

    @property
    def nys(self):
        return [1, 2]

    @property
    def nus(self):
        return [1, 2]

    def __repr__(self):
        return '{}({},{})'.format(
            type(self).__name__, self.dset_choice, self.dset_path, self.valid_split)

    @property
    def effective_num_train_samples(self):
        nu_max = max(self.nus)
        ny_max = max(self.nus)
        return (self.n_train - max(nu_max, ny_max)) * self.n_sequences


# ---- Plot dataset ----
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import argparse
    from util import parse_funct_arguments

    parser = argparse.ArgumentParser(description='Plot dataset input / output')
    parser.add_argument('--dset', default='ChenDSet', type=str,
                        help='dataset to plot.')
    parser.add_argument('--split', default='train', choices=['test', 'train'],
                        help='split to plot.')
    parser.add_argument('--save', default='',
                        help='save plot.')
    parser.add_argument('--sequence', default=0, type=int,
                        help='sequence selected for plotting (when more than one sequence is present).')
    parser.add_argument('--nth_input', default=0, type=int,
                        help='input selected for plotting (when more than one input is present).')
    parser.add_argument('--nth_output', default=0, type=int,
                        help='output selected for plotting (when more than one output is present).')
    parser.add_argument('--plot_style', nargs='*', default=[],
                        help='plot styles to be used.')
    args, unk = parser.parse_known_args()

    if args.plot_style:
        plt.style.use(args.plot_style)

    # Get dataset (from the command line)
    DatasetTmp = eval(args.dset)
    Dataset, _, unk = parse_funct_arguments(DatasetTmp, unk)
    dset = Dataset()

    if args.split == 'train':
        u, y = dset.get_train()
    else:
        u, y = dset.get_test()
    k = np.arange(len(u))

    fig, ax = plt.subplots()
    ax.step(k, u[:, args.sequence, args.nth_input], color='blue')
    ax.set_xlabel('k')
    ax.set_ylabel('u')

    axt = ax.twinx()
    axt.plot(k, y[:, args.sequence, args.nth_output], color='red')
    axt.set_xlabel('k')
    axt.set_ylabel('y', rotation=-90)
    if not args.save:
        plt.show()
    else:
        plt.savefig(args.save)




