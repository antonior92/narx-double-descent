import numpy as np
from models import *
from datasets import *
from util import parse_funct_arguments
import pickle
import itertools


def mse(y_true, y_mdl):
    return np.mean((y_true - y_mdl)**2)


def train(mdl, dset):
    # Get train
    u_train, y_train = dset.get_train()
    # Fit
    X_train, z_train = construct_linear_system(u_train, y_train, dset.nus, dset.nys)
    mdl = mdl.fit(X_train, z_train)
    return mdl


def evaluate(mdl, dset):
    # Get test
    u_train, y_train = dset.get_train()
    u_test, y_test = dset.get_test()
    X_train, z_train = construct_linear_system(u_train, y_train, dset.nus, dset.nys)
    X_test, z_test = construct_linear_system(u_test, y_test, dset.nus, dset.nys)
    # One-step-ahead prediction
    y_pred_train = back_to_original_shape(mdl.predict(X_train), n_seq=y_train.shape[1], n_out=y_train.shape[2])
    y_pred_test = back_to_original_shape(mdl.predict(X_test), n_seq=y_test.shape[1], n_out=y_test.shape[2])
    # Free run simulation
    simulate = DynamicalSystem(dset.nys, dset.nus, mdl.predict, sd_v=0, sd_w=0)
    y_sim_train = simulate(u_train)[simulate.order:, ...]
    y_sim_test = simulate(u_test)[simulate.order:, ...]
    d = {'mdl': repr(mdl), 'dset': repr(dset),
         'mse_pred_train': mse(y_train[simulate.order:, ...], y_pred_train),
         'mse_pred_test': mse(y_test[simulate.order:, ...], y_pred_test),
         'mse_sim_train': mse(y_train[simulate.order:, ...], y_sim_train),
         'mse_sim_test': mse(y_test[simulate.order:, ...], y_sim_test)
          }
    if hasattr(mdl, 'param_norm'):
        d['param_norm'] = mdl.param_norm
    pred_train = {'z_pred_train': y_pred_train, 'z_sim_train': y_sim_train}
    pred_test = {'z_pred_test': y_pred_test, 'z_sim_test': y_sim_test}
    return d, pred_train, pred_test


# ---- Main script ----
if __name__ == "__main__":
    from tqdm import tqdm
    import pandas as pd
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Estimate NARX model for different n features / n samples rate.')
    parser.add_argument('-r', '--repetitions', default=1, type=int,
                        help='number of repetitions')
    parser.add_argument('-o', '--output', default='./performance.csv',
                        help='output csv file.')
    parser.add_argument('-d', '--dset', type=str, default='ChenDSet',
                        help='number of repetitions')
    parser.add_argument('-m', '--nonlinear_model', default='RBFSampler',
                        help='number of repetitions')
    parser.add_argument('-n', '--num_points', default=60, type=int,
                        help='number of points')
    parser.add_argument('-l', '--lower_proportion', default=-1, type=float,
                        help='the lowest value for the proportion (n features / n samples) is 10^l.')
    parser.add_argument('-u', '--upper_proportion', default=2, type=float,
                        help='the upper value for the proportion (n features / n samples) is 10^u.')
    parser.add_argument('-s', '--save_models', nargs='?', default='', const='./models',
                        help='save intermediary models.')
    parser.add_argument('-w', '--reuse_weights', action='store_true',
                        help='use weights from previous model (with less features) when estimate the next one.')
    args, unk = parser.parse_known_args()

    # Saving models (when needed)
    if args.save_models:
        if not os.path.isdir(args.save_models):
            os.mkdir(args.save_models)

        def save_mdl(mdl):
            fname = os.path.join(args.save_models, repr(mdl)+'.pkl')
            with open(fname, 'wb') as f:
                pickle.dump(mdl, f)
    else:
        def save_mdl(_mdl):
            pass

    # Get model (from command line)
    ModelTmp = eval(args.nonlinear_model)
    Model, _, unk = parse_funct_arguments(ModelTmp, unk, free_arguments=['n_features', 'random_state'])

    # Get dataset (from the command line)
    DatasetTmp = eval(args.dset)
    Dataset, _, unk = parse_funct_arguments(DatasetTmp, unk)
    dset = Dataset()

    tqdm.write("Estimating baseline performance...")
    baseline_mdl = Linear()
    baseline_list = []
    for seed in tqdm(range(args.repetitions)):
        np.random.seed(seed)
        d, pred_train, pred_test = evaluate(train(baseline_mdl, dset), dset)
        d['seed'] = seed
        d['proportion'] = 0  # To signal it is the baseline (n features being a constant)
        baseline_list.append(d)
        # Save model
        save_mdl(baseline_mdl)
    df = pd.DataFrame(baseline_list)
    df.to_csv(args.output, index=False)
    tqdm.write("Done")

    tqdm.write("Estimating performance as a function of proportion...")
    list_dict = []
    underp = np.logspace(args.lower_proportion, 0, args.num_points // 2)
    overp = np.logspace(0.00001, args.upper_proportion, args.num_points - args.num_points // 2)
    proportions = np.concatenate((underp, overp))
    run_instances = list(itertools.product(range(args.repetitions), proportions))
    prev_mdl = None  # used only if reuse_weights is True
    num_samples = dset.effective_num_train_samples
    for seed, proportion in tqdm(run_instances):
        n_features = int(proportion * num_samples)
        mdl = Model(n_features=n_features, random_state=seed)
        if args.reuse_weights and hasattr(mdl, 'reuse_weights_from_mdl'):
            if prev_mdl is not None:
                mdl.reuse_weights_from_mdl(prev_mdl)
            prev_mdl = mdl
        d, pred_train, pred_test = evaluate(train(mdl, dset), dset)
        d['proportion'] = proportion
        d['seed'] = seed
        df = df.append(d, ignore_index=True)
        df.to_csv(args.output, index=False)
        # Save model
        save_mdl(mdl)
    tqdm.write("Done")

