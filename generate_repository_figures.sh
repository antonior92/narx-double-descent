#!/bin/bash

# Dataset ChenDSet
python datasets.py --dset ChenDSet --save img/chen/dset.png

# Dataset CoupledEletricalDrive
python datasets.py --dset CoupledElectricalDrives --sequence 0 --save img/ce8/dset_prbs.png
python datasets.py --dset CoupledElectricalDrives --sequence 3 --save img/ce8/dset_unif.png

# experiment #1
python plot_double_descent.py results/chen/rbfsampler.csv --tp pred --ymax 1.5 --plot_style ggplot --save img/chen/rbfs_pred.png
python plot_double_descent.py results/chen/rbfsampler.csv --tp sim --ymax 4.0 --plot_style ggplot --save img/chen/rbfs_sim.png
python plot_double_descent.py results/chen/rbfsampler.csv --tp norm --plot_style ggplot --save img/chen/rbfs_norm.png
DSET="-d ChenDSet --cutoff_freq 0.7 --hold 1 --num_train_samples 400"
MODEL="-m RBFSampler  --gamma 0.6"
python plot_predictions.py $DSET $MODEL --n_features 149 --random_state 7 --save img/chen/rbfs_before.png
python plot_predictions.py $DSET $MODEL --n_features 40000 --random_state 7 --save img/chen/rbfs_interp.png

# experiment #2
python  plot_multiple_dd.py results/chen/rbfsampler.csv  results/chen/rbfsampler_r{0.01,0.001,0.0001,0.00001,0.000001}.csv  \
  --labels  "min-norm"  "\$\lambda=10^{-2}\$" "\$\lambda=10^{-3}\$" "\$\lambda=10^{-4}\$" "\$\lambda=10^{-5}\$" "\$\lambda=10^{-6}\$" \
  --ymax 1.5 --plot_style ggplot --save img/chen/rbfs_ridge.png

# experiment #3
python plot_double_descent.py results/chen/rbfsample_ensemble.csv --tp pred --ymax 1.5 --plot_style ggplot --save img/chen/rbfs_ensemble_pred.png
python plot_double_descent.py results/chen/rbfsample_ensemble.csv --tp sim --ymax 4.0 --plot_style ggplot --save img/chen/rbfs_ensemble_sim.png
python plot_double_descent.py results/chen/rbfsample_ensemble.csv --tp norm --plot_style ggplot --save img/chen/rbfs_ensemble_norm.png

# experiment #4
python plot_double_descent.py results/chen/rbfnet.csv --tp pred --ymax 1.5 --plot_style ggplot --save img/chen/rbfnet_pred.png

# experiment #5
python plot_double_descent.py results/ce8/rbfsampler.csv --tp pred --ymax 0.2 --plot_style ggplot --save img/ce8/rbfs_pred.png
python plot_double_descent.py results/ce8/rbfsampler.csv --tp sim --ymax 8.0 --plot_style ggplot --save img/ce8/rbfs_sim.png
python plot_double_descent.py results/ce8/rbfsampler.csv --tp norm --plot_style ggplot --save img/ce8/rbfs_norm.png

# experiment #6
python plot_double_descent.py results/chen/randomforest.csv --tp pred --ymax 0.8 --plot_style ggplot --save img/chen/rf_pred.png
python plot_double_descent.py results/chen/randomforest.csv --tp sim --ymax 2.0 --plot_style ggplot --save img/chen/rf_sim.png
DSET="-d ChenDSet --cutoff_freq 0.7 --hold 1 --num_train_samples 3000"
MODEL="-m RandomForest"
python plot_predictions.py $DSET $MODEL --n_features 600 --random_state 5 --save img/chen/rf_before.png
python plot_predictions.py $DSET $MODEL --n_features 200000 --random_state 9 --save img/chen/rf_interp.png