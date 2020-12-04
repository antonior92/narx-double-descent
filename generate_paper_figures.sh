#!/bin/bash

OUT='figures_for_paper'
STY="--plot_style ggplot mystyle.mplsty"

# Generate temporary style files
mkdir $OUT
echo "font.family: serif
font.serif: Times
text.usetex: True
figure.autolayout: True" > mystyle.mplsty
echo "figure.figsize:  7.7, 5.1
font.size: 18" > half_colwidth.mplsty
echo "figure.figsize:  6.1, 4.0
font.size: 18" > onethird_colwidth.mplsty
echo "figure.figsize:  5.2, 3.6
font.size: 18" > onequarter_colwidth.mplsty

# Fig 1 (a)
python plot_double_descent.py results/ce8/rbfsampler.csv --tp pred --ymax 0.2 --xmax 1 --dont_plot_baseline\
 --plot_style ggplot mystyle.mplsty onequarter_colwidth.mplsty --save $OUT/ce8_ushape.png
# Fig 1 (b)
python plot_double_descent.py results/ce8/rbfsampler.csv --tp pred --ymax 0.2 --dont_plot_baseline --xticks 0.01 0.1 1.0 10.0 100.0 \
 --omit_legend --plot_style ggplot mystyle.mplsty onequarter_colwidth.mplsty  --save $OUT/ce8_doubledescent.png
# Fig 2 (a)
python plot_double_descent.py results/chen/rbfsampler.csv --tp pred --ymax 1.5 \
 --plot_style ggplot mystyle.mplsty onethird_colwidth.mplsty --save $OUT/chen_rff_onestepahead.png
# Fig 2 (b)
python plot_double_descent.py results/chen/rbfsampler.csv --tp sim --ymax 4.0 \
 --omit_legend --plot_style ggplot mystyle.mplsty onethird_colwidth.mplsty --save $OUT/chen_rff_freerun.png
# Fig 2 (c)
python plot_double_descent.py results/chen/rbfsampler.csv --tp norm --ymax 100000 \
 --omit_legend --plot_style ggplot mystyle.mplsty onethird_colwidth.mplsty --save $OUT/chen_rff_norm.png
# Fig 3
python plot_multiple_dd.py results/chen/rbfsampler.csv  results/chen/rbfsampler_r{0.01,0.001,0.0001,0.00001,0.000001}.csv  \
  --labels  "min-norm"  "\$\lambda=10^{-2}\$" "\$\lambda=10^{-3}\$" "\$\lambda=10^{-4}\$" "\$\lambda=10^{-5}\$" "\$\lambda=10^{-6}\$" \
  --ymax 1.5 --plot_style ggplot mystyle.mplsty half_colwidth.mplsty --save $OUT/chen_rff_vanishing_ridge.png
# Fig 4 (a)
python plot_double_descent.py results/chen/rbfsample_ensemble.csv --tp pred --ymax 1.5 \
    --plot_style ggplot mystyle.mplsty onequarter_colwidth.mplsty --save $OUT/chen_rff_ensembles_onestepahead.png
# Fig 4 (b)
python plot_double_descent.py results/chen/rbfsample_ensemble.csv --tp sim --ymax 4.0 \
    --omit_legend --plot_style ggplot mystyle.mplsty onequarter_colwidth.mplsty --save $OUT/chen_rff_ensembles_freerun.png
# Fig 5
python plot_double_descent.py results/chen/rbfnet.csv --tp pred --ymax 1.5 \
    --plot_style ggplot mystyle.mplsty half_colwidth.mplsty --save $OUT/chen_rbfnet_onestepahead.png
# Fig 6 (a)
python plot_double_descent.py results/chen/randomforest.csv --tp pred --ymax 0.8001 \
    --plot_style ggplot mystyle.mplsty onequarter_colwidth.mplsty --save $OUT/chen_randomforest_onestepahead.png
# Fig 6 (b)
python plot_double_descent.py results/chen/randomforest.csv --tp sim --ymax 2.0 \
    --omit_legend --plot_style ggplot mystyle.mplsty onequarter_colwidth.mplsty --save $OUT/chen_randomforest_freerun.png

# remove artifacts
rm mystyle.mplsty
rm half_colwidth.mplsty
rm onethird_colwidth.mplsty
rm onequarter_colwidth.mplsty