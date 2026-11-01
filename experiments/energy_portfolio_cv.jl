include("energy_portfolio_run.jl")

function run_energy_portfolio_cv(; data_dir = "data/training_dataset",
    output_dir = "energy_cv_results", evaluation_mode::Symbol = :rolling_origin,
    beta = 0.4, alpha = 0.95, gamma = 0.99, delta_inf = 0.9, delta_sup = 1.1,
    lambda = 1e-4, num_epochs = 2000, batchsize = 100, step_size = 1e-8,
    first_test_year = nothing, last_test_year = nothing, min_train_years = 5,
    generator_variant = :G2, psi = 10.0, num_scenarios = 200, num_scenarios_test = 1000, seed = 1)
    dataset = load_energy_dataset(data_dir)
    folds = build_energy_year_folds(dataset; evaluation_mode, first_test_year, last_test_year, min_train_years)
    mkpath(output_dir)
    results = EnergyYearEvaluation[]
    assessment = DataFrame(test_year = Int[], coverage_90 = Float64[], energy_score = Float64[])
    realized_test_costs = (mse = Float64[], spo = Float64[])

    for fold in folds
        normalization = fit_energy_normalization(dataset.X[:, fold.train_indices],
            dataset.c[:, fold.train_indices])
        X_normalized, c_normalized = normalize_energy_data(dataset.X, dataset.c, normalization)
        X_train = X_normalized[:, fold.train_indices]
        c_train = c_normalized[:, fold.train_indices]
        X_validation = X_normalized[:, fold.validation_indices]
        c_validation = c_normalized[:, fold.validation_indices]
        X_test = X_normalized[:, fold.test_indices]
        c_test = dataset.c[:, fold.test_indices]

        prepared = prepare_energy_fold(dataset, fold; generator_variant, psi, num_scenarios,
            num_scenarios_test, seed, beta, alpha, gamma, delta_inf, delta_sup)
        train_oracles = prepared.train_oracles
        validation_oracles = prepared.validation_oracles
        test_oracles = prepared.test_oracles
        coverage = scenario_interval_coverage(prepared.test_trees, c_test; level = 0.9)
        energy_score = mean(scenario_energy_score(prepared.test_trees[k], c_test[:, k])
            for k in 1:length(fold.test_indices))
        push!(assessment, (fold.test_year, coverage, energy_score))

        fold_log = joinpath(output_dir, "fold_$(fold.test_year).log")
        log_file = open(fold_log, "w")
        println(log_file, "ENERGY PORTFOLIO FOLD TRAINING")
        println(log_file, "evaluation_mode=$(evaluation_mode)")
        println(log_file, "test_year=$(fold.test_year) validation_year=$(fold.validation_year)")
        println(log_file, "train_years=$(fold.train_years)")
        println(log_file, "beta=$beta alpha=$alpha gamma=$gamma delta_inf=$delta_inf delta_sup=$delta_sup")
        println(log_file, "lambda=$(lambda) num_epochs=$(num_epochs) batchsize=$(batchsize) step_size=$(step_size)")
        println(log_file, "train=$(length(fold.train_indices)) validation=$(length(fold.validation_indices)) test=$(length(fold.test_indices))")
        println(log_file, "num_scenarios=$num_scenarios num_scenarios_test=$num_scenarios_test seed=$seed")
        write_generator_details(log_file, prepared)
        println(log_file, "generator_test_coverage_90=$coverage generator_test_energy_score=$energy_score")
        println(log_file, "normalization feature_mean=$(normalization.feature_mean)")
        println(log_file, "normalization feature_scale=$(normalization.feature_scale)")
        println(log_file, "normalization cost_mean=$(normalization.cost_mean)")
        println(log_file, "normalization cost_scale=$(normalization.cost_scale)")

        mse_model = fit_energy_ridge(X_train, c_train, lambda)
        mse_train_metrics = energy_metrics(mse_model, X_train,
            dataset.c[:, fold.train_indices], train_oracles,
            normalization; beta, alpha, detailed = true)
        mse_validation_metrics = energy_metrics(mse_model, X_validation,
            dataset.c[:, fold.validation_indices], validation_oracles,
            normalization; beta, alpha, detailed = true)
        mse_baseline = (train_mse = energy_mse_loss(mse_model, X_train, c_train),
            validation_mse = energy_mse_loss(mse_model, X_validation, c_validation),
            train_cost = mse_train_metrics.mean_cost,
            validation_cost = mse_validation_metrics.mean_cost,
            train_cvar = mse_train_metrics.mean_cvar,
            validation_cvar = mse_validation_metrics.mean_cvar)
        println(log_file, "mse_train_mse=$(mse_baseline.train_mse) mse_validation_mse=$(mse_baseline.validation_mse) " *
            "mse_train_cost=$(mse_baseline.train_cost) mse_validation_cost=$(mse_baseline.validation_cost) " *
            "mse_train_cvar=$(mse_baseline.train_cvar) mse_validation_cvar=$(mse_baseline.validation_cvar)")

        objective_function = B -> energy_spo_objective(B, X_train, c_train,
            train_oracles, normalization, beta, alpha, lambda)
        epoch_callback = function(epoch, B, objective, gradient_norm)
            model = EnergyRidgePredictor(B)
            spo_train_metrics = energy_metrics(model, X_train,
                dataset.c[:, fold.train_indices], train_oracles,
                normalization; beta, alpha, detailed = true)
            spo_validation_metrics = energy_metrics(model, X_validation,
                dataset.c[:, fold.validation_indices], validation_oracles,
                normalization; beta, alpha, detailed = true)
            spo_train_mse = energy_mse_loss(model, X_train, c_train)
            spo_validation_mse = energy_mse_loss(model, X_validation, c_validation)
            mse_line = @sprintf("epoch=%03d mse_train_mse=%20.12e mse_validation_mse=%20.12e mse_train_cost=%20.12e mse_validation_cost=%20.12e mse_train_cvar=%20.12e mse_validation_cvar=%20.12e",
                epoch, mse_baseline.train_mse, mse_baseline.validation_mse,
                mse_baseline.train_cost, mse_baseline.validation_cost,
                mse_baseline.train_cvar, mse_baseline.validation_cvar)
            spo_line = @sprintf("epoch=%03d spo_train_mse=%20.12e spo_validation_mse=%20.12e spo_train_cost=%20.12e spo_validation_cost=%20.12e spo_train_cvar=%20.12e spo_validation_cvar=%20.12e spo_objective=%20.12e spo_gradient_norm=%20.12e",
                epoch, spo_train_mse, spo_validation_mse,
                spo_train_metrics.mean_cost, spo_validation_metrics.mean_cost,
                spo_train_metrics.mean_cvar, spo_validation_metrics.mean_cvar,
                objective, gradient_norm)
            println(log_file, mse_line)
            println(log_file, spo_line)
            write_portfolio_details(log_file, epoch, "validation",
                fold.validation_indices, dataset, spo_validation_metrics)
            flush(log_file)
        end

        spo_model, history = train_energy_spo(X_train, c_train, train_oracles, beta;
            lambda, numiter = num_epochs, batchsize, step_size,
            cost_mean = normalization.cost_mean, cost_scale = normalization.cost_scale,
            objective_function = objective_function, epoch_callback = epoch_callback)

        mse_metrics = energy_metrics(mse_model, X_test, c_test, test_oracles, normalization; beta, alpha)
        spo_metrics = energy_metrics(spo_model, X_test, c_test, test_oracles, normalization; beta, alpha)
        _, c_test_normalized = normalize_energy_data(dataset.X[:, fold.test_indices],
            c_test, normalization)
        mse_objective = (1.0 - beta) * mse_metrics.mean_cost + beta * mse_metrics.mean_cvar
        spo_objective = (1.0 - beta) * spo_metrics.mean_cost + beta * spo_metrics.mean_cvar
        result = EnergyYearEvaluation(fold.test_year, fold.validation_year,
            mse_metrics.mean_cost, mse_metrics.mean_cvar, energy_mse_loss(mse_model, X_test, c_test_normalized),
            spo_metrics.mean_cost, spo_metrics.mean_cvar, energy_mse_loss(spo_model, X_test, c_test_normalized),
            mse_objective, spo_objective)
        push!(results, result)
        # realized costs in real terms (price level ipca_ref) so that different years are comparable
        deflator = dataset.ipca_ref ./ dataset.ipca_t[fold.test_indices]
        append!(realized_test_costs.mse, mse_metrics.costs .* deflator)
        append!(realized_test_costs.spo, spo_metrics.costs .* deflator)

        println(log_file, "FINAL MSE test_cost=$(result.mse_cost) test_cvar=$(result.mse_cvar) test_mse=$(result.mse_mse)")
        println(log_file, "FINAL SPO+ test_cost=$(result.spo_cost) test_cvar=$(result.spo_cvar) test_mse=$(result.spo_mse)")
        println(log_file, "mse_test_objective=$(result.mse_objective) spo_test_objective=$(result.spo_objective)")
        println(log_file, "spo_objective_history=$(history.objective_history)")
        println(log_file, "spo_gradient_norm_history=$(history.gradient_norm_history)")
        close(log_file)
        println("test_year=$(fold.test_year) mse_cost=$(result.mse_cost) spo_cost=$(result.spo_cost) mse_cvar=$(result.mse_cvar) spo_cvar=$(result.spo_cvar)")
    end

    summary = _summary_statistics(results)
    _write_summary(joinpath(output_dir, "summary"), results, summary)
    _save_cv_plots(results, output_dir)
    CSV.write(joinpath(output_dir, "generator_assessment.csv"), assessment)
    # Out-of-sample risk: CVaR of the real realized costs of all test decisions (pooled over folds).
    realized_risk = DataFrame(method = ["MSE", "SPO+"],
        realized_real_cost_mean = [mean(realized_test_costs.mse), mean(realized_test_costs.spo)],
        realized_real_cost_cvar = [empirical_cvar(costs, fill(1.0 / length(costs), length(costs)), alpha)
            for costs in (realized_test_costs.mse, realized_test_costs.spo)])
    CSV.write(joinpath(output_dir, "realized_test_risk.csv"), realized_risk)
    return (folds = folds, results = results, summary = summary, assessment = assessment,
        realized_risk = realized_risk, output_dir = output_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    for variant in (:G1, :G2)
        result = run_energy_portfolio_cv(generator_variant = variant,
            output_dir = "energy_cv_results_$(variant)")
        println(variant)
        println(result.summary)
        println(result.assessment)
        println(result.realized_risk)
    end
end
