using Cairo, CSV, DataFrames, Fontconfig, Gadfly, Statistics

struct EnergyYearFold
    test_year::Int
    validation_year::Int
    train_years::Vector{Int}
    train_indices::Vector{Int}
    validation_indices::Vector{Int}
    test_indices::Vector{Int}
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

function _year_indices(dataset, year_value)
    return findall(==(year_value), dataset.years)
end

function build_energy_year_folds(dataset::EnergyDataset;
    evaluation_mode::Symbol = :rolling_origin,
    first_test_year = nothing, last_test_year = nothing)
    evaluation_mode in (:leave_one_year_out, :rolling_origin) ||
        error("evaluation_mode must be :leave_one_year_out or :rolling_origin")
    years = sort(unique(dataset.years))
    default_first_year = evaluation_mode == :rolling_origin ? years[3] : years[2]
    first_year = isnothing(first_test_year) ? default_first_year : first_test_year
    last_year = isnothing(last_test_year) ? years[end] : last_test_year
    folds = EnergyYearFold[]
    for test_year in years
        test_year < first_year && continue
        test_year > last_year && continue
        validation_year = test_year - 1
        validation_year in years || continue
        train_years = if evaluation_mode == :rolling_origin
            filter(year_value -> year_value < validation_year, years)
        else
            filter(year_value -> year_value != validation_year && year_value != test_year, years)
        end
        isempty(train_years) && continue
        train_indices = sort(vcat([_year_indices(dataset, year_value) for year_value in train_years]...))
        validation_indices = _year_indices(dataset, validation_year)
        test_indices = _year_indices(dataset, test_year)
        length(validation_indices) == 4 || error("validation year must have four samples")
        length(test_indices) == 4 || error("test year must have four samples")
        push!(folds, EnergyYearFold(test_year, validation_year, train_years,
            train_indices, validation_indices, test_indices))
    end
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
