# SmartPredictThenOptimize
Code for the paper ["Smart 'Predict, then Optimize'"](https://arxiv.org/abs/1710.08005) by Adam Elmachtoub and Paul Grigas.

The code will run on Julia v1.5.0 and requires several packages as listed in the headers of various files.

This code is helpful for replicating the experiments in the above paper. For training your own models on your own data using SPO+, it is recommended to use the excellent [PyEPO package](https://github.com/khalil-research/PyEPO) by Bo Tang and Elias B. Khalil.

## Overview

The code is divided into several folders:
- `solver` contains all of the files needed to run the SPO+ (SGD and reformulation approaches), random forests, least squares, and least absolute loss methods.
- `oracles` contains examples of optimization oracles such as shortest path and portfolio optimization.
- `tests` contains some basic sanity check tests.
- `experiments` contains files concerning the experiments in the paper.
- `plots` contains R code for constructing the plots in the paper.

The `solver` folder contains the files `reformulation.jl` and `sgd.jl` for training a model with the SPO+ loss function using a fixed value of the regularization parameter. To actually a train a model in practice, it is recommended to use the file `validation_set.jl` and the corresponding function is `validation_set_alg`, which additionally performs cross-validation using the validation set approach. The documentation in the file explains the syntax for calling the `validation_set_alg` function.

The files in the folder `experiments` can simply be run to replicate the experiments included in the paper.

## Energy portfolio toy model

The repository also contains a SPO-compatible energy portfolio oracle in
`oracles/energy_portfolio_oracle.jl`. It models monthly bilateral contracts,
deterministic spot purchase/sale, and a frozen PLD scenario tree used by the
CVaR term. The predicted PLD vector enters only the deterministic spot
objective, so the feasible set remains fixed across the oracle calls used by
SPO+.

The energy-specific training modules are:

- `solver/energy_scenarios.jl` for injected scenario trees and validation;
- `solver/energy_predictors.jl` for multi-output ridge/MSE prediction;
- `solver/energy_spo.jl` for SPO+ training with gradient
	`2 * (1 - beta) * (a(c) - a(2 * c_hat - c))`, where
	`a = e_plus - e_minus`.

The real data pipeline is available through `load_energy_dataset` in
`solver/energy_scenarios.jl`. It joins
`data/training_dataset/pld_historico_SE.csv`,
`data/training_dataset/cenarios_pld_cmo_SE.csv`, and
`data/training_dataset/ena_semanal_historico_SE.csv` by `TI`. Each year and
week-of-month combination is one sample with twelve monthly values; the four
scenario columns form a `4 x 12` frozen scenario tree. The experiment splits
2006/2019 for testing and 2007/2018 for validation, and normalizes using only
the training split.

Run `julia experiments/energy_portfolio_run.jl` to train for 20 epochs with
batch size 100 and write the detailed results to
`energy_portfolio_training.log`. The toy uses a deterministic decision
evaluated with CVaR over fixed scenarios; it is not the original Python model
with scenario-indexed recourse variables.
