using Cairo, CSV, DataFrames, Dates, Fontconfig, Gadfly, Statistics

struct EnergyYearFold
    test_year::Int
    validation_year::Int
    train_years::Vector{Int}
    train_indices::Vector{Int}
    validation_indices::Vector{Int}
    test_indices::Vector{Int}
    generator_cutoff::Date      # last month used to fit the scenario generator of the fold
end

struct EnergyYearEvaluation
    test_year::Int
    validation_year::Int
    mse_cost::Float64
    mse_cvar::Float64
    mse_mse::Float64
    spo_cost::Float64
    spo_cvar::Float64
    spo_mse::Float64
    mse_objective::Float64
    spo_objective::Float64
end

"""
Fold with test decisions in `test_year` and validation decisions in `test_year - 1`. Training
samples are the decisions whose twelve target months all end by December of `test_year - 2`, so
no realized price of the validation or test period enters training, normalization or the scenario
generator (fitted with months up to the same cutoff).
"""
function _rolling_origin_fold(dataset::EnergyDataset, test_year::Integer)
    validation_year = test_year - 1
    cutoff = Date(test_year - 2, 12, 1)
    train_indices = findall(months -> last(months) <= cutoff, dataset.target_months)
    validation_indices = findall(==(validation_year), dataset.years)
    test_indices = findall(==(test_year), dataset.years)
    train_years = sort(unique(dataset.years[train_indices]))
    return EnergyYearFold(test_year, validation_year, train_years, train_indices,
        validation_indices, test_indices, cutoff)
end

function build_energy_year_folds(dataset::EnergyDataset;
    evaluation_mode::Symbol = :rolling_origin,
    first_test_year = nothing, last_test_year = nothing, min_train_years::Integer = 5)
    # With twelve-month targets, leave-one-year-out would train on later years.
    evaluation_mode == :rolling_origin || error("evaluation_mode must be :rolling_origin")
    folds = EnergyYearFold[]
    for test_year in sort(unique(dataset.years))
        !isnothing(first_test_year) && test_year < first_test_year && continue
        !isnothing(last_test_year) && test_year > last_test_year && continue
        fold = _rolling_origin_fold(dataset, test_year)
        (isempty(fold.validation_indices) || isempty(fold.test_indices)) && continue
        if length(fold.train_years) < min_train_years
            isnothing(first_test_year) ||
                @warn "test year $test_year skipped: fewer than $min_train_years training years"
            continue
        end
        push!(folds, fold)
    end
    isempty(folds) && error("no fold satisfies the requested test years and min_train_years")
    return folds
end

function _year_mean(metrics, field)
    return mean(getfield(metrics, field))
end

function _summary_statistics(results::Vector{EnergyYearEvaluation})
    methods = [("MSE", :mse_cost, :mse_cvar, :mse_mse),
        ("SPO+", :spo_cost, :spo_cvar, :spo_mse)]
    rows = DataFrame(method = String[], cost_mean = Float64[], cost_std = Float64[],
        cvar_mean = Float64[], cvar_std = Float64[], mse_mean = Float64[], mse_std = Float64[])
    for (name, cost_field, cvar_field, mse_field) in methods
        costs = [getfield(result, cost_field) for result in results]
        cvars = [getfield(result, cvar_field) for result in results]
        mses = [getfield(result, mse_field) for result in results]
        push!(rows, (name, mean(costs), std(costs), mean(cvars), std(cvars), mean(mses), std(mses)))
    end
    return rows
end

function _write_year_log(path, fold, result, mse_metrics, spo_metrics)
    open(path, "w") do io
        println(io, "test_year=$(fold.test_year) validation_year=$(fold.validation_year)")
        println(io, "train_years=$(fold.train_years)")
        println(io, "mse_test_cost=$(result.mse_cost) mse_test_cvar=$(result.mse_cvar) mse_test_mse=$(result.mse_mse)")
        println(io, "spo_test_cost=$(result.spo_cost) spo_test_cvar=$(result.spo_cvar) spo_test_mse=$(result.spo_mse)")
        println(io, "mse_test_objective=$(result.mse_objective) spo_test_objective=$(result.spo_objective)")
        println(io, "mse_weekly_costs=$(mse_metrics.costs)")
        println(io, "mse_weekly_cvars=$(mse_metrics.cvars)")
        println(io, "spo_weekly_costs=$(spo_metrics.costs)")
        println(io, "spo_weekly_cvars=$(spo_metrics.cvars)")
    end
end

function _write_summary(path, results, summary)
    CSV.write(path * ".csv", summary)
    open(path * ".txt", "w") do io
        println(io, "year,mse_cost,mse_cvar,mse_mse,spo_cost,spo_cvar,spo_mse")
        for result in results
            println(io, "$(result.test_year),$(result.mse_cost),$(result.mse_cvar),$(result.mse_mse),$(result.spo_cost),$(result.spo_cvar),$(result.spo_mse)")
        end
        println(io, "")
        println(io, "Método | custo médio | desvio | CVaR médio | desvio | MSE médio | desvio")
        for row in eachrow(summary)
            println(io, "$(row.method) | $(row.cost_mean) | $(row.cost_std) | $(row.cvar_mean) | $(row.cvar_std) | $(row.mse_mean) | $(row.mse_std)")
        end
    end
end

function _save_cv_plots(results, output_dir)
    mkpath(joinpath(output_dir, "plots"))
    years = [result.test_year for result in results]
    function save_metric(filename, title, ylabel, mse_values, spo_values)
        frame = DataFrame(year = vcat(years, years), value = vcat(mse_values, spo_values),
            method = vcat(fill("MSE", length(years)), fill("SPO+", length(years))))
        chart = plot(frame, x = :year, y = :value, color = :method,
            Geom.line, Geom.point, Guide.title(title), Guide.xlabel("Test year"),
            Guide.ylabel(ylabel), Scale.color_discrete_manual("#1b9e77", "#d95f02"),
            Theme(background_color = "white", panel_fill = "white"))
        draw(PNG(joinpath(output_dir, "plots", filename), 8inch, 5inch), chart)
    end
    save_metric("cost_by_year.png", "Cost by test year", "Cost",
        [result.mse_cost for result in results], [result.spo_cost for result in results])
    save_metric("cvar_by_year.png", "CVaR by test year", "CVaR",
        [result.mse_cvar for result in results], [result.spo_cvar for result in results])
    save_metric("mse_by_year.png", "MSE by test year", "MSE",
        [result.mse_mse for result in results], [result.spo_mse for result in results])
end
