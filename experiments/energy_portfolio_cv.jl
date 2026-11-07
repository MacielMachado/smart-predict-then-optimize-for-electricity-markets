include("energy_portfolio_run.jl")

"""
Rolling-origin evaluation over all test years with enough training data. Any entry of
ENERGY_DEFAULTS can be passed as a keyword (e.g. `generator_variant = :G1`, `gamma = 0.99`).
"""
function run_energy_portfolio_cv(; data_dir = "data/training_dataset",
    output_dir = "energy_cv_results", evaluation_mode::Symbol = :rolling_origin,
    first_test_year = nothing, last_test_year = nothing, min_train_years = 5, validation_span = 1,
    options...)
    dataset = load_energy_dataset(data_dir)
    folds = build_energy_year_folds(dataset; evaluation_mode, first_test_year, last_test_year,
        min_train_years, validation_span)
    mkpath(output_dir)
    results = EnergyYearEvaluation[]
    assessment = DataFrame(test_year = Int[], coverage_90 = Float64[], energy_score = Float64[])
    realized_test_costs = [Float64[] for _ in ENERGY_METHODS]
    test_regrets = [Float64[] for _ in ENERGY_METHODS]

    for fold in folds
        log_file = open(joinpath(output_dir, "fold_$(fold.test_year).log"), "w")
        println(log_file, "ENERGY PORTFOLIO FOLD TRAINING")
        println(log_file, "evaluation_mode=$(evaluation_mode)")
        fold_result = try
            train_and_evaluate_energy_fold(dataset, fold; log_io = log_file, options...)
        finally
            close(log_file)
        end
        push!(results, fold_result.evaluation)
        push!(assessment, (fold.test_year, fold_result.coverage, fold_result.energy_score))
        for k in eachindex(ENERGY_METHODS)
            append!(realized_test_costs[k], fold_result.test_metrics[k].costs)   # already real R$
            append!(test_regrets[k], fold_result.test_metrics[k].regrets)
        end
        evaluation = fold_result.evaluation
        println("test_year=$(fold.test_year) informative=$(evaluation.informative) regret(Média, MSE, SPO+)=$(evaluation.regret) " *
            "cost=$(evaluation.cost) ridge_lambda=$(evaluation.ridge_lambda) spo_best_epoch=$(evaluation.spo_best_epoch)")
    end

    summary = _summary_statistics(results)
    _write_summary(joinpath(output_dir, "summary"), results, summary)
    _save_cv_plots(results, output_dir)
    CSV.write(joinpath(output_dir, "generator_assessment.csv"), assessment)
    # Out-of-sample risk over all test decisions, pooled over folds. The tail of the realized cost
    # reflects the market regime (all methods decide alike in crisis years), so the risk of each
    # method is measured on its regret: mean and CVaR of the per-decision regret (real R$).
    alpha = _energy_options(options).alpha
    uniform(values) = fill(1.0 / length(values), length(values))
    realized_risk = DataFrame(method = ENERGY_METHODS,
        realized_real_cost_mean = [mean(costs) for costs in realized_test_costs],
        regret_mean = [mean(regrets) for regrets in test_regrets],
        regret_cvar = [empirical_cvar(regrets, uniform(regrets), alpha) for regrets in test_regrets])
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
        println(_paired_comparisons(result.results))
        println(result.assessment)
        println(result.realized_risk)
    end
end
