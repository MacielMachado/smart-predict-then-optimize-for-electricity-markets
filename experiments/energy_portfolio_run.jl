include("../oracles/energy_portfolio_oracle.jl")
include("../solver/energy_scenarios.jl")
include("../solver/energy_predictors.jl")
include("../solver/energy_spo.jl")

using Cairo, DataFrames, Fontconfig, Gadfly, LinearAlgebra, Printf, Statistics

const ENERGY_DEMAND = Float64[1050, 980, 1020, 1000, 970, 920,
    900, 890, 930, 980, 1010, 1050]
const ENERGY_CONTRACT_PRICES = fill(280.0, 12)

function build_energy_oracles(dataset::EnergyDataset, indices;
    beta = 0.4, alpha = 0.95, gamma = 0.99,
    delta_inf = 0.9, delta_sup = 1.1)
    probabilities = fill(0.25, 4)
    return [build_energy_oracle(dataset.scenario_trees[i], probabilities,
        ENERGY_DEMAND, ENERGY_CONTRACT_PRICES;
        alpha = alpha, beta = beta, gamma = gamma,
        delta_inf = delta_inf, delta_sup = delta_sup,
        spot_upper_bound = ENERGY_DEMAND) for i in indices]
end

function energy_metrics(model, X_normalized, c_raw, oracle_builders,
    normalization; beta = 0.4, alpha = 0.95, detailed = false)
    predictions_raw = denormalize_energy_cost(predict_energy(model, X_normalized), normalization)
    probabilities = fill(0.25, 4)
    costs = Float64[]
    cvars = Float64[]
    objectives = Float64[]
    results = EnergyPortfolioOracleResult[]
    for i in 1:size(X_normalized, 2)
        result = oracle_builders[i](predictions_raw[:, i])
        realized_cost = dot(ENERGY_CONTRACT_PRICES, result.w) +
            dot(c_raw[:, i], result.action)
        push!(costs, realized_cost)
        push!(cvars, empirical_cvar(result.scenario_costs, probabilities, alpha))
        push!(objectives, (1.0 - beta) * realized_cost + beta * cvars[end])
        detailed && push!(results, result)
    end
    return (mean_cost = mean(costs), mean_cvar = mean(cvars),
        mean_objective = mean(objectives), costs = costs, cvars = cvars,
        objectives = objectives, results = results,
        predictions_raw = predictions_raw)
end

function energy_spo_objective(B, X_normalized, c_normalized, oracle_builders,
    normalization, beta, alpha, lambda)
    probabilities = fill(0.25, 4)
    values = Float64[]
    c_mean = normalization.cost_mean
    c_scale = normalization.cost_scale
    for i in 1:size(X_normalized, 2)
        c_raw = c_normalized[:, i] .* c_scale .+ c_mean
        chat_raw = (B * X_normalized[:, i]) .* c_scale .+ c_mean
        true_result = oracle_builders[i](c_raw)
        auxiliary_result = oracle_builders[i](2.0 .* chat_raw .- c_raw)
        theta_chat_dot_true = (1.0 - beta) * dot(ENERGY_CONTRACT_PRICES, true_result.w) +
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
        println(io, "sample=$dataset_index year=$(dataset.years[dataset_index]) week=$(dataset.weeks[dataset_index])")
        for month in 1:12
            println(io, "month=$month date=$(dataset.dates[dataset_index][month]) " *
                "w=$(result.w[month]) e_plus=$(result.e_plus[month]) " *
                "e_minus=$(result.e_minus[month])")
        end
    end
end

function save_training_plots(history, mse_baseline, output_dir)
    mkpath(output_dir)
    epochs = collect(1:length(history.objective_history))
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

    save_plot(joinpath(output_dir, "mse_loss_comparison.png"),
        "MSE loss: SPO+ versus MSE baseline", "Normalized MSE",
        [(history.spo_train_mse, "#d95f02", "spo_train_mse"),
         (history.spo_validation_mse, "#7570b3", "spo_validation_mse"),
         (fill(mse_baseline.train_mse, length(epochs)), "#1b9e77", "mse_train_mse"),
         (fill(mse_baseline.validation_mse, length(epochs)), "#66a61e", "mse_validation_mse")])
    save_plot(joinpath(output_dir, "cost_comparison.png"),
        "Operational cost: SPO+ versus MSE", "Cost",
        [(history.spo_train_cost, "#d95f02", "spo_train_cost"),
         (history.spo_validation_cost, "#7570b3", "spo_validation_cost"),
         (fill(mse_baseline.train_cost, length(epochs)), "#1b9e77", "mse_train_cost"),
         (fill(mse_baseline.validation_cost, length(epochs)), "#66a61e", "mse_validation_cost")])
    save_plot(joinpath(output_dir, "cvar_comparison.png"),
        "CVaR: SPO+ versus MSE", "CVaR",
        [(history.spo_train_cvar, "#d95f02", "spo_train_cvar"),
         (history.spo_validation_cvar, "#7570b3", "spo_validation_cvar"),
         (fill(mse_baseline.train_cvar, length(epochs)), "#1b9e77", "mse_train_cvar"),
         (fill(mse_baseline.validation_cvar, length(epochs)), "#66a61e", "mse_validation_cvar")])
    save_plot(joinpath(output_dir, "objective.png"),
        "SPO+ objective", "Objective", [(history.objective_history, "#1f78b4", "spo_objective")])
    save_plot(joinpath(output_dir, "gradient_norm.png"),
        "SPO+ gradient norm", "Gradient norm", [(history.gradient_norm_history, "#e31a1c", "spo_gradient_norm")])
end

function run_energy_portfolio(; data_dir = "data/training_dataset",
    log_path = "energy_portfolio_training.log", beta = 0.4,
    alpha = 0.95, gamma = 0.99, delta_inf = 0.9, delta_sup = 1.1,
    lambda = 1e-4, num_epochs = 2000, batchsize = 100,
    step_size = 1e-8, plot_dir = "energy_portfolio_plots")
    dataset = load_energy_dataset(data_dir)
    splits = energy_split_indices(dataset)
    normalization = fit_energy_normalization(dataset.X[:, splits.train], dataset.c[:, splits.train])
    X_normalized, c_normalized = normalize_energy_data(dataset.X, dataset.c, normalization)

    train_indices = splits.train
    validation_indices = splits.validation
    test_indices = splits.test
    train_oracles = build_energy_oracles(dataset, train_indices; beta, alpha, gamma, delta_inf, delta_sup)
    validation_oracles = build_energy_oracles(dataset, validation_indices; beta, alpha, gamma, delta_inf, delta_sup)
    test_oracles = build_energy_oracles(dataset, test_indices; beta, alpha, gamma, delta_inf, delta_sup)

    X_train = X_normalized[:, train_indices]
    c_train_normalized = c_normalized[:, train_indices]
    X_validation = X_normalized[:, validation_indices]
    c_validation_normalized = c_normalized[:, validation_indices]
    X_test = X_normalized[:, test_indices]
    c_test = dataset.c[:, test_indices]

    mse_model = fit_energy_ridge(X_train, c_train_normalized, lambda)
    log_file = open(log_path, "w")
    println(log_file, "ENERGY PORTFOLIO TRAINING")
    println(log_file, "beta=$beta alpha=$alpha gamma=$gamma delta_inf=$delta_inf delta_sup=$delta_sup")
    println(log_file, "epochs=$num_epochs batchsize=$batchsize step_size=$step_size lambda=$lambda")
    println(log_file, "train=$(length(train_indices)) validation=$(length(validation_indices)) test=$(length(test_indices))")
    println(log_file, "normalization feature_mean=$(normalization.feature_mean)")
    println(log_file, "normalization feature_scale=$(normalization.feature_scale)")
    println(log_file, "normalization cost_mean=$(normalization.cost_mean)")
    println(log_file, "normalization cost_scale=$(normalization.cost_scale)")

    mse_train = energy_metrics(mse_model, X_train, dataset.c[:, train_indices], train_oracles,
        normalization; beta, alpha, detailed = true)
    mse_validation = energy_metrics(mse_model, X_validation, dataset.c[:, validation_indices], validation_oracles,
        normalization; beta, alpha, detailed = true)
    mse_baseline = (train_mse = energy_mse_loss(mse_model, X_train, c_train_normalized),
        validation_mse = energy_mse_loss(mse_model, X_validation, c_validation_normalized),
        train_cost = mse_train.mean_cost, validation_cost = mse_validation.mean_cost,
        train_cvar = mse_train.mean_cvar, validation_cvar = mse_validation.mean_cvar)
    println(log_file, "mse_train_mse=$(mse_baseline.train_mse) mse_validation_mse=$(mse_baseline.validation_mse) " *
        "mse_train_cost=$(mse_baseline.train_cost) mse_validation_cost=$(mse_baseline.validation_cost) " *
        "mse_train_cvar=$(mse_baseline.train_cvar) mse_validation_cvar=$(mse_baseline.validation_cvar)")

    history_series = (spo_train_mse = Float64[], spo_validation_mse = Float64[],
        spo_train_cost = Float64[], spo_validation_cost = Float64[],
        spo_train_cvar = Float64[], spo_validation_cvar = Float64[],
        objective_history = Float64[], gradient_norm_history = Float64[])

    objective_function = B -> energy_spo_objective(B, X_train, c_train_normalized,
        train_oracles, normalization, beta, alpha, lambda)
    epoch_callback = function(epoch, B, objective, gradient_norm)
        model = EnergyRidgePredictor(B)
        train_metrics = energy_metrics(model, X_train, dataset.c[:, train_indices], train_oracles,
            normalization; beta, alpha, detailed = true)
        validation_metrics = energy_metrics(model, X_validation, dataset.c[:, validation_indices], validation_oracles,
            normalization; beta, alpha, detailed = true)
        spo_train_mse = energy_mse_loss(model, X_train, c_train_normalized)
        spo_validation_mse = energy_mse_loss(model, X_validation, c_validation_normalized)
        push!(history_series.spo_train_mse, spo_train_mse)
        push!(history_series.spo_validation_mse, spo_validation_mse)
        push!(history_series.spo_train_cost, train_metrics.mean_cost)
        push!(history_series.spo_validation_cost, validation_metrics.mean_cost)
        push!(history_series.spo_train_cvar, train_metrics.mean_cvar)
        push!(history_series.spo_validation_cvar, validation_metrics.mean_cvar)
        push!(history_series.objective_history, objective)
        push!(history_series.gradient_norm_history, gradient_norm)
        mse_line = @sprintf("epoch=%03d mse_train_mse=%20.12e mse_validation_mse=%20.12e mse_train_cost=%20.12e mse_validation_cost=%20.12e mse_train_cvar=%20.12e mse_validation_cvar=%20.12e",
            epoch, mse_baseline.train_mse, mse_baseline.validation_mse,
            mse_baseline.train_cost, mse_baseline.validation_cost,
            mse_baseline.train_cvar, mse_baseline.validation_cvar)
        spo_line = @sprintf("epoch=%03d spo_train_mse=%20.12e spo_validation_mse=%20.12e spo_train_cost=%20.12e spo_validation_cost=%20.12e spo_train_cvar=%20.12e spo_validation_cvar=%20.12e spo_objective=%20.12e spo_gradient_norm=%20.12e",
            epoch, spo_train_mse, spo_validation_mse, train_metrics.mean_cost,
            validation_metrics.mean_cost, train_metrics.mean_cvar,
            validation_metrics.mean_cvar, objective, gradient_norm)
        println(mse_line)
        println(spo_line)
        println(log_file, mse_line)
        println(log_file, spo_line)
        write_portfolio_details(log_file, epoch, "validation", validation_indices, dataset, validation_metrics)
        flush(log_file)
    end

    spo_model, history = train_energy_spo(X_train, c_train_normalized, train_oracles, beta;
        lambda = lambda, numiter = num_epochs, batchsize = batchsize,
        step_size = step_size, cost_mean = normalization.cost_mean,
        cost_scale = normalization.cost_scale,
        objective_function = objective_function, epoch_callback = epoch_callback)

    test_metrics = energy_metrics(spo_model, X_test, c_test, test_oracles,
        normalization; beta, alpha, detailed = true)
    mse_test_metrics = energy_metrics(mse_model, X_test, c_test, test_oracles,
        normalization; beta, alpha, detailed = true)
    println(log_file, "FINAL MSE test_cost=$(mse_test_metrics.mean_cost) test_cvar=$(mse_test_metrics.mean_cvar)")
    println(log_file, "FINAL SPO+ test_cost=$(test_metrics.mean_cost) test_cvar=$(test_metrics.mean_cvar)")
    write_portfolio_details(log_file, num_epochs, "test_mse", test_indices, dataset, mse_test_metrics)
    write_portfolio_details(log_file, num_epochs, "test_spo", test_indices, dataset, test_metrics)
    save_training_plots(history_series, mse_baseline, plot_dir)
    close(log_file)

    return (dataset = dataset, splits = splits, normalization = normalization,
        mse_model = mse_model, spo_model = spo_model, history = history,
        history_series = history_series, mse_baseline = mse_baseline,
        mse_test = mse_test_metrics, spo_test = test_metrics, log_path = log_path)
end


if abspath(PROGRAM_FILE) == @__FILE__
    result = run_energy_portfolio()
    println("Energy portfolio training completed. Log: ", result.log_path)
end
