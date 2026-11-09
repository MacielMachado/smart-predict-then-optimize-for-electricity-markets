include("../oracles/energy_portfolio_oracle.jl")
include("../solver/energy_scenarios.jl")
include("../solver/energy_scenario_generator.jl")
include("../solver/energy_predictors.jl")
include("../solver/energy_spo.jl")
include("../solver/energy_cross_validation.jl")

using Cairo, DataFrames, Fontconfig, Gadfly, LinearAlgebra, Printf, Statistics

const ENERGY_DEMAND = Float64[1050, 980, 1020, 1000, 970, 920,
    900, 890, 930, 980, 1010, 1050]

"""
Default experiment settings. Every entry can be overridden by a keyword of the same name in
`run_energy_portfolio`, `run_energy_portfolio_cv` and `run_energy_portfolio_synthetic`.

Decision problem: `gamma` is the minimum contracted share of the demand; the contract price is
`contract_markup` × the mean real PLD of the 12 months before the decision (`:trailing_mean`) or a
fixed nominal price (`:fixed`, `fixed_contract_price`, the original setting with gamma = 0.99).
Predictors: with `anchor = :climatology` (default) both learn the deviation of the PLD from the
seasonal historical mean of the training targets, so strong regularization shrinks them toward that
mean instead of toward a non-seasonal constant (`anchor = :none` predicts the PLD directly). The ridge
lambda is chosen in `lambda_grid` by the validation objective; SPO+ uses that lambda, starts from the
ridge solution (`spo_init = :ridge`) or from zero (`:zero`), and returns the epoch with the best
validation objective (evaluated every `eval_every` epochs).

SPO+ path (one epoch = one subgradient step on a minibatch of `batchsize` decisions): `num_epochs`
steps of length `step_size`/√t; `spo_averaging = true` uses the running mean of the iterates;
`spo_select_epoch = true` returns the epoch with the best validation objective, `false` always
returns the model after `num_epochs` (fixed number of epochs; the validation still chooses lambda);
`spo_epoch0_eligible = true` lets the validation keep the starting point (the ridge) unchanged.

Evaluation: `clip_predictions = true` bounds every prediction by the real PLD floor and ceiling in
force at the decision date before it enters the oracle (validation and test, all methods; not inside
the SPO+ gradient). `refit = true` refits the three methods, after lambda and the epoch have been
chosen, on every decision known before the first test decision (fold.refit_indices): the seasonal
mean and the anchor are recomputed and the ridge is refitted with the chosen lambda. The SPO+ model
of the test depends on `spo_refit`:
- `:retrain`: SPO+ runs again from the refitted ridge for the chosen number of epochs (scenarios of
  the refit decisions come from the test generator);
- `:transfer`: the correction learned on the training decisions is added to the refitted ridge,
  prediction = refitted ridge prediction + s_r ⊙ (B_spo - B_ridge) x̃, where x̃ and s_r are the
  feature standardization and residual scale of the training fit (no new SPO+ steps).
`track_test = true` also records the test regret of the SPO+ path at every evaluation (diagnostic
curve; it does not select anything and uses the models trained before the refit).
"""
const ENERGY_DEFAULTS = (beta = 0.4, alpha = 0.95, gamma = 0.8, delta_inf = 0.9, delta_sup = 1.1,
    contract_price_rule = :trailing_mean, contract_markup = 1.0, fixed_contract_price = 280.0,
    anchor = :climatology, lambda_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0],
    num_epochs = 1000, batchsize = 64, step_size = 0.1, step_rule = :normalized_decaying,
    spo_init = :ridge, spo_averaging = false, spo_select_epoch = true, spo_epoch0_eligible = false,
    eval_every = 10, clip_predictions = true, refit = true, spo_refit = :retrain, track_train = true, track_test = false, log_portfolios = false,
    generator_variant = :G2, psi = 10.0, num_scenarios = 200, num_scenarios_test = 1000, seed = 1)

function _energy_options(kwargs)
    given = kwargs isa NamedTuple ? kwargs : values(kwargs)   # aceita NamedTuple ou palavras-chave
    unknown = setdiff(keys(given), keys(ENERGY_DEFAULTS))
    isempty(unknown) || error("unknown option(s): $(collect(unknown))")
    return merge(ENERGY_DEFAULTS, given)
end

# One Gurobi environment shared by all oracle models (a new environment per model would check out
# the license hundreds of times).
const _ENERGY_GUROBI_ENV = Ref{Any}(nothing)
function _energy_gurobi_env()
    _ENERGY_GUROBI_ENV[] === nothing && (_ENERGY_GUROBI_ENV[] = Gurobi.Env())
    return _ENERGY_GUROBI_ENV[]
end

"""Oracle of one sample together with the contract prices of its decision (real R\$/MWh)."""
struct EnergySampleOracle
    oracle::EnergyPortfolioOracle
    contract_prices::Vector{Float64}
end
(sample_oracle::EnergySampleOracle)(c::AbstractVector{<:Real}) = sample_oracle.oracle(c)

# Calendar-month profile reordered so that position h is the h-th month after the decision.
_rotate_months(values, start_month) = [values[mod1(start_month + h - 1, length(values))] for h in 1:length(values)]

function energy_contract_prices(dataset::EnergyDataset, i::Integer; contract_price_rule = :trailing_mean,
    contract_markup = 1.0, fixed_contract_price = 280.0)
    price = if contract_price_rule == :trailing_mean
        Float64(contract_markup) * dataset.pld_trailing_real[i]
    elseif contract_price_rule == :fixed
        Float64(fixed_contract_price) / dataset.price_level[i]      # nominal at the decision date
    else
        error("contract_price_rule must be :trailing_mean or :fixed")
    end
    return fill(price, size(dataset.c, 1))
end

function build_energy_oracles(dataset::EnergyDataset, indices, scenario_trees;
    beta = 0.4, alpha = 0.95, gamma = 0.8, delta_inf = 0.9, delta_sup = 1.1,
    contract_price_rule = :trailing_mean, contract_markup = 1.0, fixed_contract_price = 280.0)
    length(scenario_trees) == length(indices) || error("one scenario tree is required per sample")
    oracles = EnergySampleOracle[]
    for (k, i) in enumerate(indices)
        tree = scenario_trees[k]
        num_scenarios = size(tree, 1)
        demand = _rotate_months(ENERGY_DEMAND, dataset.start_months[i])
        contract_prices = energy_contract_prices(dataset, i; contract_price_rule, contract_markup,
            fixed_contract_price)
        oracle = build_energy_oracle(tree, fill(1.0 / num_scenarios, num_scenarios),
            demand, contract_prices;
            alpha = alpha, beta = beta, gamma = gamma,
            delta_inf = delta_inf, delta_sup = delta_sup,
            spot_upper_bound = demand, gurobiEnv = _energy_gurobi_env())
        push!(oracles, EnergySampleOracle(oracle, contract_prices))
    end
    return oracles
end

"""Optimal objective of each sample's problem when its realized PLD is known (hindsight)."""
energy_hindsight_objectives(oracles, c_raw::AbstractMatrix{<:Real}) =
    [oracles[k](c_raw[:, k]).objective_value for k in 1:length(oracles)]

"""
Fit the scenario generators of a fold, generate one scenario set per sample and build the oracles.
Training and validation scenarios come from the generator fitted up to `fold.generator_cutoff`;
test scenarios from the one fitted up to `fold.test_generator_cutoff` (the month before the first
test decision).
"""
function prepare_energy_fold(dataset::EnergyDataset, fold::EnergyYearFold; options...)
    o = _energy_options(options)
    fit(cutoff) = fit_energy_scenario_generator(dataset.monthly, cutoff;
        variant = o.generator_variant, psi = o.psi, ipca_ref = dataset.ipca_ref)
    generator = fit(fold.generator_cutoff)
    test_generator = fold.test_generator_cutoff == fold.generator_cutoff ? generator :
        fit(fold.test_generator_cutoff)
    trees(gen, indices, S) = build_energy_scenario_trees(dataset, gen, indices; num_scenarios = S, seed = o.seed)
    train_trees = trees(generator, fold.train_indices, o.num_scenarios)
    validation_trees = trees(generator, fold.validation_indices, o.num_scenarios)
    test_trees = trees(test_generator, fold.test_indices, o.num_scenarios_test)
    oracle_options = (beta = o.beta, alpha = o.alpha, gamma = o.gamma, delta_inf = o.delta_inf,
        delta_sup = o.delta_sup, contract_price_rule = o.contract_price_rule,
        contract_markup = o.contract_markup, fixed_contract_price = o.fixed_contract_price)
    train_oracles = build_energy_oracles(dataset, fold.train_indices, train_trees; oracle_options...)
    validation_oracles = build_energy_oracles(dataset, fold.validation_indices, validation_trees; oracle_options...)
    test_oracles = build_energy_oracles(dataset, fold.test_indices, test_trees; oracle_options...)
    # Refit decisions: everything known before the first test decision, scenarios from the test generator
    refit_oracles = if o.refit
        refit_trees = trees(test_generator, fold.refit_indices, o.num_scenarios)
        build_energy_oracles(dataset, fold.refit_indices, refit_trees; oracle_options...)
    else
        EnergySampleOracle[]
    end
    return (generator = generator, test_generator = test_generator,
        train_trees = train_trees, validation_trees = validation_trees, test_trees = test_trees,
        train_oracles = train_oracles, validation_oracles = validation_oracles, test_oracles = test_oracles,
        refit_oracles = refit_oracles, refit = o.refit,
        validation_hindsight = energy_hindsight_objectives(validation_oracles, dataset.c[:, fold.validation_indices]),
        test_hindsight = energy_hindsight_objectives(test_oracles, dataset.c[:, fold.test_indices]))
end

function write_generator_details(io, generator; label = "generator")
    println(io, label, ": ", describe_energy_scenario_generator(generator))
    for warning in check_energy_scenario_generator(generator)
        println(io, label, "_warning=", warning)
    end
end

_normalized_mse(predictions_raw, c_raw, normalization) =
    0.5 * sum(abs2, (predictions_raw .- c_raw) ./ normalization.cost_scale) / size(c_raw, 2)

"""
Evaluate decisions made with `predictions_raw` (real R\$/MWh, one column per sample), optionally
clipped to `bounds = (lower, upper)` (one value per sample) before entering the oracle. Per sample:
realized cost (contracts + realized PLD on the spot position), in-model CVaR of the decision under
its scenario set, objective (1-β)·cost + β·CVaR and, when `hindsight` is given, the regret
objective - hindsight optimum (the SPO loss of the decision).
"""
function energy_metrics_from_predictions(predictions_raw::AbstractMatrix{<:Real}, c_raw, oracle_builders;
    beta = 0.4, alpha = 0.95, hindsight = nothing, detailed = false, bounds = nothing)
    if bounds !== nothing                   # (lower, upper): real PLD limits of each decision
        lower, upper = bounds
        predictions_raw = clamp.(predictions_raw, reshape(lower, 1, :), reshape(upper, 1, :))
    end
    costs = Float64[]
    cvars = Float64[]
    objectives = Float64[]
    results = EnergyPortfolioOracleResult[]
    for i in 1:size(predictions_raw, 2)
        result = oracle_builders[i](predictions_raw[:, i])
        realized_cost = dot(oracle_builders[i].contract_prices, result.w) +
            dot(c_raw[:, i], result.action)
        push!(costs, realized_cost)
        num_scenarios = length(result.scenario_costs)
        push!(cvars, empirical_cvar(result.scenario_costs, fill(1.0 / num_scenarios, num_scenarios), alpha))
        push!(objectives, (1.0 - beta) * realized_cost + beta * cvars[end])
        detailed && push!(results, result)
    end
    regrets = hindsight === nothing ? fill(NaN, length(objectives)) : objectives .- hindsight
    normalized_regret = hindsight === nothing ? NaN : sum(regrets) / sum(abs, hindsight)
    return (mean_cost = mean(costs), mean_cvar = mean(cvars),
        mean_objective = mean(objectives), mean_regret = mean(regrets), normalized_regret = normalized_regret,
        costs = costs, cvars = cvars, objectives = objectives, regrets = regrets,
        results = results, predictions_raw = Matrix{Float64}(predictions_raw))
end

"""Metrics of a linear model; `offset` (matrix, one column per sample) is the anchor added to its predictions."""
energy_metrics(model, X_normalized, c_raw, oracle_builders, normalization; offset = 0.0, kwargs...) =
    energy_metrics_from_predictions(denormalize_energy_cost(predict_energy(model, X_normalized), normalization) .+ offset,
        c_raw, oracle_builders; kwargs...)

"""Seasonal historical mean: mean real PLD of each calendar month over the training targets."""
function energy_climatology_predictions(dataset::EnergyDataset, train_indices, indices)
    sums = zeros(12)
    counts = zeros(Int, 12)
    for j in train_indices, h in 1:size(dataset.c, 1)
        month_value = month(dataset.target_months[j][h])
        sums[month_value] += dataset.c[h, j]
        counts[month_value] += 1
    end
    means = sums ./ max.(counts, 1)
    return [means[month(dataset.target_months[i][h])] for h in 1:size(dataset.c, 1), i in indices]
end

function energy_spo_objective(B, X_normalized, c_normalized, oracle_builders,
    normalization, beta, alpha, lambda; offset = zeros(size(c_normalized)))
    values = Float64[]
    c_mean = normalization.cost_mean
    c_scale = normalization.cost_scale
    for i in 1:size(X_normalized, 2)
        c_raw = c_normalized[:, i] .* c_scale .+ c_mean .+ offset[:, i]
        chat_raw = (B * X_normalized[:, i]) .* c_scale .+ c_mean .+ offset[:, i]
        true_result = oracle_builders[i](c_raw)
        auxiliary_result = oracle_builders[i](2.0 .* chat_raw .- c_raw)
        probabilities = fill(1.0 / length(true_result.xi), length(true_result.xi))
        theta_chat_dot_true = (1.0 - beta) * dot(oracle_builders[i].contract_prices, true_result.w) +
            beta * true_result.eta +
            beta / (1.0 - alpha) * dot(probabilities, true_result.xi) +
            (1.0 - beta) * dot(chat_raw, true_result.action)
        push!(values, -auxiliary_result.objective_value +
            2.0 * theta_chat_dot_true - true_result.objective_value)
    end
    return mean(values) + 0.5 * lambda * sum(abs2, B)
end

function write_portfolio_details(io, epoch, label, indices, dataset, metrics)
    println(io, "PORTFOLIOS epoch=$epoch split=$label")
    for (local_index, dataset_index) in enumerate(indices)
        result = metrics.results[local_index]
        println(io, "sample=$dataset_index decision_date=$(dataset.decision_dates[dataset_index])")
        for month in 1:ENERGY_HORIZON
            println(io, "month=$month date=$(dataset.target_months[dataset_index][month]) " *
                "w=$(result.w[month]) e_plus=$(result.e_plus[month]) " *
                "e_minus=$(result.e_minus[month])")
        end
    end
end

"""
Train and evaluate the three methods on one fold: seasonal historical mean, ridge (MSE) and SPO+.
Returns the fold evaluation plus the objects needed for logs and plots. `log_io` (optional) receives
the fold log. `prepared` (optional) reuses the generators, scenarios and oracles of
`prepare_energy_fold` (e.g. to compare several SPO+ settings on the same fold); it must have been
built with the same scenario, oracle and refit options.
"""
function train_and_evaluate_energy_fold(dataset::EnergyDataset, fold::EnergyYearFold;
    log_io = nothing, prepared = nothing, options...)
    o = _energy_options(options)
    write_log(args...) = log_io === nothing || println(log_io, args...)
    train, validation, test = fold.train_indices, fold.validation_indices, fold.test_indices
    prepared = prepared === nothing ? prepare_energy_fold(dataset, fold; options...) : prepared
    o.refit && !prepared.refit && error("the prepared fold has no refit oracles (built with refit = false)")
    o.spo_refit in (:retrain, :transfer) || error("spo_refit must be :retrain or :transfer")
    metric_options = (beta = o.beta, alpha = o.alpha)
    bounds(indices) = o.clip_predictions ? (dataset.pld_min_real[indices], dataset.pld_max_real[indices]) : nothing

    # Anchor (seasonal historical mean of the fitting targets) and normalization of c - anchor
    function anchored_data(fit_indices)
        anchor = o.anchor == :climatology ? energy_climatology_predictions(dataset, fit_indices, 1:size(dataset.c, 2)) :
            o.anchor == :none ? zeros(size(dataset.c)) : error("anchor must be :climatology or :none")
        residual = dataset.c .- anchor
        normalization = fit_energy_normalization(dataset.X[:, fit_indices], residual[:, fit_indices])
        X_normalized, c_normalized = normalize_energy_data(dataset.X, residual, normalization)
        return (anchor = anchor, normalization = normalization, X = X_normalized, c = c_normalized)
    end
    selection = anchored_data(train)
    normalization = selection.normalization
    X_train, c_train = selection.X[:, train], selection.c[:, train]
    X_validation, X_test = selection.X[:, validation], selection.X[:, test]
    anchor_train, anchor_validation, anchor_test =
        selection.anchor[:, train], selection.anchor[:, validation], selection.anchor[:, test]
    c_train_raw, c_validation_raw, c_test_raw = dataset.c[:, train], dataset.c[:, validation], dataset.c[:, test]

    write_log("test_year=$(fold.test_year) validation_year=$(fold.validation_year) train_years=$(fold.train_years)")
    write_log("train=$(length(train)) validation=$(length(validation)) test=$(length(test)) ",
        "refit=$(length(fold.refit_indices)) train_cutoff=$(fold.generator_cutoff) ",
        "test_generator_cutoff=$(fold.test_generator_cutoff)")
    write_log("options=$(o)")
    write_log("features=$(ENERGY_FEATURE_NAMES)")
    log_io === nothing || write_generator_details(log_io, prepared.generator; label = "generator")
    log_io === nothing || write_generator_details(log_io, prepared.test_generator; label = "test_generator")

    # Ridge (MSE): lambda chosen by the validation objective
    lambda_scores = Float64[]
    for lambda in o.lambda_grid
        model = fit_energy_ridge(X_train, c_train, lambda)
        score = energy_metrics(model, X_validation, c_validation_raw, prepared.validation_oracles,
            normalization; offset = anchor_validation, bounds = bounds(validation), metric_options...).mean_objective
        push!(lambda_scores, score)
        write_log("ridge lambda=$lambda validation_objective=$score")
    end
    ridge_lambda = o.lambda_grid[argmin(lambda_scores)]
    mse_model = fit_energy_ridge(X_train, c_train, ridge_lambda)
    mse_train = energy_metrics(mse_model, X_train, c_train_raw, prepared.train_oracles, normalization;
        offset = anchor_train, bounds = bounds(train), metric_options...)
    mse_validation = energy_metrics(mse_model, X_validation, c_validation_raw, prepared.validation_oracles,
        normalization; offset = anchor_validation, bounds = bounds(validation), metric_options...,
        hindsight = prepared.validation_hindsight)
    mse_baseline = (train_mse = energy_mse_loss(mse_model, X_train, c_train),
        validation_mse = _normalized_mse(mse_validation.predictions_raw, c_validation_raw, normalization),
        train_cost = mse_train.mean_cost, validation_cost = mse_validation.mean_cost,
        train_cvar = mse_train.mean_cvar, validation_cvar = mse_validation.mean_cvar,
        validation_objective = mse_validation.mean_objective)
    write_log("ridge_lambda=$ridge_lambda mse_validation_objective=$(mse_validation.mean_objective) " *
        "mse_validation_regret=$(mse_validation.mean_regret)")
    # Diagnostic: test regret of the models fitted on the training decisions only (no selection)
    test_metric_options = (metric_options..., hindsight = prepared.test_hindsight, bounds = bounds(test))
    mse_test_regret_before_refit = o.track_test ? energy_metrics(mse_model, X_test, c_test_raw,
        prepared.test_oracles, normalization; offset = anchor_test, test_metric_options...).mean_regret : NaN

    # SPO+: same lambda, warm start, best epoch by the validation objective
    history_series = (epochs = Int[], spo_train_mse = Float64[], spo_validation_mse = Float64[],
        spo_train_cost = Float64[], spo_validation_cost = Float64[],
        spo_train_cvar = Float64[], spo_validation_cvar = Float64[],
        spo_validation_objective = Float64[], spo_test_regret = Float64[], objective_history = Float64[],
        gradient_norm_history = Float64[])
    B_init = o.spo_init == :ridge ? copy(mse_model.B) :
        o.spo_init == :zero ? zeros(size(c_train, 1), size(X_train, 1)) :
        error("spo_init must be :ridge or :zero")
    objective_function = B -> energy_spo_objective(B, X_train, c_train, prepared.train_oracles,
        normalization, o.beta, o.alpha, ridge_lambda; offset = anchor_train)
    epoch_callback = function(epoch, B, objective, gradient_norm)
        model = EnergyRidgePredictor(B)
        validation_metrics = energy_metrics(model, X_validation, c_validation_raw,
            prepared.validation_oracles, normalization; offset = anchor_validation,
            bounds = bounds(validation), metric_options..., detailed = o.log_portfolios)
        train_metrics = o.track_train ? energy_metrics(model, X_train, c_train_raw,
            prepared.train_oracles, normalization; offset = anchor_train, bounds = bounds(train),
            metric_options...) : (mean_cost = NaN, mean_cvar = NaN)
        test_regret = o.track_test ? energy_metrics(model, X_test, c_test_raw, prepared.test_oracles,
            normalization; offset = anchor_test, test_metric_options...).mean_regret : NaN
        push!(history_series.epochs, epoch)
        push!(history_series.spo_train_mse, energy_mse_loss(model, X_train, c_train))
        push!(history_series.spo_validation_mse,
            _normalized_mse(validation_metrics.predictions_raw, c_validation_raw, normalization))
        push!(history_series.spo_train_cost, train_metrics.mean_cost)
        push!(history_series.spo_validation_cost, validation_metrics.mean_cost)
        push!(history_series.spo_train_cvar, train_metrics.mean_cvar)
        push!(history_series.spo_validation_cvar, validation_metrics.mean_cvar)
        push!(history_series.spo_validation_objective, validation_metrics.mean_objective)
        push!(history_series.spo_test_regret, test_regret)
        push!(history_series.objective_history, objective)
        push!(history_series.gradient_norm_history, gradient_norm)
        write_log(@sprintf("epoch=%04d spo_objective=%.10e gradient_norm=%.6e norm_B=%.6e train_mse=%.6e validation_mse=%.6e train_cost=%.10e validation_cost=%.10e validation_cvar=%.10e validation_objective=%.10e test_regret_diagnostic=%.10e",
            epoch, objective, gradient_norm, norm(B), history_series.spo_train_mse[end],
            history_series.spo_validation_mse[end], train_metrics.mean_cost,
            validation_metrics.mean_cost, validation_metrics.mean_cvar, validation_metrics.mean_objective,
            test_regret))
        o.log_portfolios && log_io !== nothing &&
            write_portfolio_details(log_io, epoch, "validation", validation, dataset, validation_metrics)
        log_io === nothing || flush(log_io)
        return o.spo_select_epoch ? validation_metrics.mean_objective : nothing   # nothing = no selection
    end
    spo_options = (batchsize = o.batchsize, step_size = o.step_size, step_rule = o.step_rule,
        averaging = o.spo_averaging)
    spo_model, history = train_energy_spo(X_train, c_train, prepared.train_oracles, o.beta;
        lambda = ridge_lambda, numiter = o.num_epochs, spo_options..., B_init = B_init,
        cost_mean = normalization.cost_mean, cost_scale = normalization.cost_scale,
        cost_offset = anchor_train, objective_function = objective_function,
        epoch_callback = epoch_callback, evaluate_initial = o.spo_epoch0_eligible,
        eval_every = o.eval_every, rng = MersenneTwister(o.seed))
    write_log("spo_best_epoch=$(history.best_epoch) spo_best_validation_objective=$(history.best_score) " *
        "mse_validation_objective=$(mse_validation.mean_objective) norm_B_ridge=$(norm(mse_model.B)) " *
        "norm_B_spo=$(norm(spo_model.B))")

    # Refit on every decision known before the test, with the chosen lambda and number of epochs
    raw_predictions(model, X, normalization_used, offset) =
        denormalize_energy_cost(predict_energy(model, X), normalization_used) .+ offset
    final = if o.refit
        refit = fold.refit_indices
        fitted = anchored_data(refit)
        X_refit, c_refit = fitted.X[:, refit], fitted.c[:, refit]
        mse_final = fit_energy_ridge(X_refit, c_refit, ridge_lambda)
        mse_test_predictions = raw_predictions(mse_final, fitted.X[:, test], fitted.normalization, fitted.anchor[:, test])
        refit_init = o.spo_init == :ridge ? copy(mse_final.B) : zeros(size(mse_final.B))
        spo_final, spo_test_predictions = if o.spo_refit == :transfer
            # correction learned on the training decisions, in R$/MWh, added to the refitted ridge
            correction = denormalize_energy_cost((spo_model.B .- mse_model.B) * X_test, normalization) .-
                normalization.cost_mean
            (nothing, mse_test_predictions .+ correction)
        elseif history.best_epoch == 0                         # validation kept the starting point
            model = EnergyRidgePredictor(refit_init)
            (model, raw_predictions(model, fitted.X[:, test], fitted.normalization, fitted.anchor[:, test]))
        else
            model = first(train_energy_spo(X_refit, c_refit, prepared.refit_oracles, o.beta;
                lambda = ridge_lambda, numiter = history.best_epoch, spo_options..., B_init = refit_init,
                cost_mean = fitted.normalization.cost_mean, cost_scale = fitted.normalization.cost_scale,
                cost_offset = fitted.anchor[:, refit], eval_every = max(history.best_epoch, 1),
                rng = MersenneTwister(o.seed + 1)))
            (model, raw_predictions(model, fitted.X[:, test], fitted.normalization, fitted.anchor[:, test]))
        end
        write_log("refit decisions=$(length(refit)) (decision dates $(minimum(dataset.decision_dates[refit])) " *
            "to $(maximum(dataset.decision_dates[refit]))) spo_refit=$(o.spo_refit) spo_epochs=$(history.best_epoch) " *
            "norm_B_ridge_refit=$(norm(mse_final.B))" *
            (spo_final === nothing ? "" : " norm_B_spo_refit=$(norm(spo_final.B))"))
        (mean_fit = refit, mse = mse_final, spo = spo_final, normalization = fitted.normalization,
            mse_test_predictions = mse_test_predictions, spo_test_predictions = spo_test_predictions)
    else
        (mean_fit = train, mse = mse_model, spo = spo_model, normalization = normalization,
            mse_test_predictions = raw_predictions(mse_model, X_test, normalization, anchor_test),
            spo_test_predictions = raw_predictions(spo_model, X_test, normalization, anchor_test))
    end

    # Test: seasonal mean, MSE, SPO+ (predictions clipped inside energy_metrics_from_predictions)
    test_options = (test_metric_options..., detailed = true)
    mean_predictions = energy_climatology_predictions(dataset, final.mean_fit, test)
    test_metrics = [
        energy_metrics_from_predictions(mean_predictions, c_test_raw, prepared.test_oracles; test_options...),
        energy_metrics_from_predictions(final.mse_test_predictions, c_test_raw, prepared.test_oracles; test_options...),
        energy_metrics_from_predictions(final.spo_test_predictions, c_test_raw, prepared.test_oracles; test_options...)]
    test_mses = [_normalized_mse(metrics.predictions_raw, c_test_raw, final.normalization) for metrics in test_metrics]
    objectives = [m.mean_objective for m in test_metrics]
    evaluation = EnergyYearEvaluation(fold.test_year, fold.validation_year, ridge_lambda, history.best_epoch,
        [m.mean_cost for m in test_metrics], [m.mean_cvar for m in test_metrics], objectives,
        [m.mean_regret for m in test_metrics], [m.normalized_regret for m in test_metrics], test_mses,
        _energy_informative(objectives))
    coverage = scenario_interval_coverage(prepared.test_trees, c_test_raw; level = 0.9)
    energy_score = mean(scenario_energy_score(prepared.test_trees[k], c_test_raw[:, k]) for k in 1:length(test))
    for (k, name) in enumerate(ENERGY_METHODS)
        write_log("FINAL $name test_cost=$(evaluation.cost[k]) test_cvar=$(evaluation.cvar[k]) " *
            "test_objective=$(evaluation.objective[k]) test_regret=$(evaluation.regret[k]) " *
            "test_normalized_regret=$(evaluation.normalized_regret[k]) " *
            "test_mse=$(evaluation.mse[k])")
    end
    write_log("informative_year=$(evaluation.informative)")
    write_log("generator_test_coverage_90=$coverage generator_test_energy_score=$energy_score")
    if log_io !== nothing
        for (k, name) in enumerate(ENERGY_METHODS)
            write_portfolio_details(log_io, history.best_epoch, "test_$(name)", test, dataset, test_metrics[k])
        end
        flush(log_io)
    end
    return (evaluation = evaluation, test_metrics = test_metrics, coverage = coverage,
        energy_score = energy_score, history = history, history_series = history_series,
        mse_test_regret_before_refit = mse_test_regret_before_refit,
        mse_baseline = mse_baseline, mse_model = final.mse,
        spo_model = final.spo === nothing ? spo_model : final.spo,
        normalization = final.normalization, prepared = prepared)
end

function save_training_plots(history, mse_baseline, output_dir)
    mkpath(output_dir)
    epochs = history.epochs
    function save_plot(path, title, ylabel, series)
        rows = vcat([DataFrame(epoch = epochs, value = values,
            series = fill(label, length(epochs))) for (values, _, label) in series]...)
        colors = [color for (_, color, _) in series]
        labels = [label for (_, _, label) in series]
        plot_object = plot(rows, x = :epoch, y = :value, color = :series,
            Geom.line, Geom.point, Guide.title(title), Guide.xlabel("Epoch"),
            Guide.ylabel(ylabel), Guide.manual_color_key("Series", labels),
            Scale.color_discrete_manual(colors...),
            Theme(background_color = "white", panel_fill = "white", key_position = :top))
        draw(PNG(path, 8inch, 5inch), plot_object)
    end
    constant(value) = fill(value, length(epochs))

    save_plot(joinpath(output_dir, "mse_loss_comparison.png"),
        "MSE loss: SPO+ versus MSE baseline", "Normalized MSE",
        [(history.spo_train_mse, "#d95f02", "spo_train_mse"),
         (history.spo_validation_mse, "#7570b3", "spo_validation_mse"),
         (constant(mse_baseline.train_mse), "#1b9e77", "mse_train_mse"),
         (constant(mse_baseline.validation_mse), "#66a61e", "mse_validation_mse")])
    save_plot(joinpath(output_dir, "cost_comparison.png"),
        "Operational cost: SPO+ versus MSE", "Cost (real R\$)",
        [(history.spo_train_cost, "#d95f02", "spo_train_cost"),
         (history.spo_validation_cost, "#7570b3", "spo_validation_cost"),
         (constant(mse_baseline.train_cost), "#1b9e77", "mse_train_cost"),
         (constant(mse_baseline.validation_cost), "#66a61e", "mse_validation_cost")])
    save_plot(joinpath(output_dir, "cvar_comparison.png"),
        "CVaR: SPO+ versus MSE", "CVaR (real R\$)",
        [(history.spo_train_cvar, "#d95f02", "spo_train_cvar"),
         (history.spo_validation_cvar, "#7570b3", "spo_validation_cvar"),
         (constant(mse_baseline.train_cvar), "#1b9e77", "mse_train_cvar"),
         (constant(mse_baseline.validation_cvar), "#66a61e", "mse_validation_cvar")])
    save_plot(joinpath(output_dir, "validation_objective.png"),
        "Validation objective: SPO+ versus MSE", "(1-β) cost + β CVaR",
        [(history.spo_validation_objective, "#d95f02", "spo_validation_objective"),
         (constant(mse_baseline.validation_objective), "#1b9e77", "mse_validation_objective")])
    save_plot(joinpath(output_dir, "objective.png"),
        "SPO+ objective", "Objective", [(history.objective_history, "#1f78b4", "spo_objective")])
    save_plot(joinpath(output_dir, "gradient_norm.png"),
        "SPO+ gradient norm", "Gradient norm", [(history.gradient_norm_history, "#e31a1c", "spo_gradient_norm")])
end

"""
Single rolling-origin fold (development, debugging, hyperparameter inspection). Any entry of
ENERGY_DEFAULTS can be passed as a keyword.
"""
function run_energy_portfolio(; data_dir = "data/training_dataset",
    log_path = "energy_portfolio_training.log", plot_dir = "energy_portfolio_plots",
    test_year = 2019, min_train_years = 5, validation_span = 1, options...)
    dataset = load_energy_dataset(data_dir)
    fold = only(build_energy_year_folds(dataset; first_test_year = test_year,
        last_test_year = test_year, min_train_years, validation_span))
    log_file = open(log_path, "w")
    println(log_file, "ENERGY PORTFOLIO TRAINING")
    result = try
        train_and_evaluate_energy_fold(dataset, fold; log_io = log_file, options...)
    finally
        close(log_file)
    end
    save_training_plots(result.history_series, result.mse_baseline, plot_dir)
    return merge(result, (dataset = dataset, fold = fold, log_path = log_path))
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_energy_portfolio()
    println("Energy portfolio training completed. Log: ", result.log_path)
end
