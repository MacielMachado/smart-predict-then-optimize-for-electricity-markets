#=
Data interfaces for the energy portfolio toy problem.

Each sample is one decision taken on the first day of a month m for the twelve months
m, ..., m+11. All monetary values are in REAL terms: reais at the constant price level `ipca_ref`,
deflated with the price index known at the decision date. The objective is linear in prices and the
CVaR is positively homogeneous, so scaling all prices of a sample by the same constant does not
change its optimal decision; working in real terms makes costs comparable across years.

`X` holds the features of the predictor. The generator conditions on (ena_prev, ear_prev,
cmo_real_prev, real limits, calendar month), all of which are functions of `X`, so the scenario sets
depend on the context only (Remark on context-dependent scenario data). Scenario sets are not stored
here: they depend on the fold and come from energy_scenario_generator.jl.
=#

using CSV, DataFrames, Dates, Statistics

struct EnergyScenarioBatch
    X::Matrix{Float64}
    c::Matrix{Float64}
    scenario_trees::Vector{Matrix{Float64}}
    probabilities::Vector{Float64}
end

const ENERGY_HORIZON = 12
const ENERGY_FEATURE_NAMES = ["log_ena_prev", "cmo_real_prev", "ear_prev", "cmo_exante_real",
    "pld_trailing_real", "pld_min_real", "pld_max_real", "sin_month", "cos_month"]

struct EnergyDataset
    X::Matrix{Float64}                    # predictor features known at the decision date (p x n)
    c::Matrix{Float64}                    # realized real PLD of the twelve target months (12 x n)
    decision_dates::Vector{Date}          # first day of the first target month
    target_months::Vector{Vector{Date}}
    years::Vector{Int}                    # year of the decision date
    start_months::Vector{Int}             # calendar month of the first target month
    ena_prev::Vector{Float64}             # generator conditioning state (month before decision)
    ear_prev::Vector{Float64}
    cmo_real_prev::Vector{Float64}
    pld_min_real::Vector{Float64}         # limits in force at the decision date, real terms
    pld_max_real::Vector{Float64}
    pld_trailing_real::Vector{Float64}    # mean PLD of the 12 months before the decision, real terms
    price_level::Vector{Float64}          # ipca_t / ipca_ref (nominal = real * price_level)
    monthly::DataFrame                    # monthly table used to fit the generator (nominal)
    ipca_ref::Float64                     # constant price level that defines real values
end

struct EnergyNormalization
    feature_mean::Vector{Float64}
    feature_scale::Vector{Float64}
    cost_mean::Vector{Float64}
    cost_scale::Vector{Float64}
end

function validate_energy_scenarios(c::AbstractMatrix{<:Real},
    scenario_trees::Vector{<:AbstractMatrix{<:Real}},
    probabilities::AbstractVector{<:Real})
    T, n = size(c)
    length(scenario_trees) == n || error("one scenario tree is required per sample")
    all(isfinite, c) || error("c must be finite")
    probs = Vector{Float64}(probabilities)
    all(probs .>= 0) || error("probabilities must be nonnegative")
    isapprox(sum(probs), 1.0; atol = 1e-10) || error("probabilities must sum to one")
    for tree in scenario_trees
        S, tree_T = size(tree)
        tree_T == T || error("every scenario tree must have T months")
        S == length(probs) || error("every scenario tree must have S scenarios")
        all(isfinite, tree) || error("scenario trees must be finite")
    end
    return true
end

function make_energy_scenario_batch(X::AbstractMatrix{<:Real},
    c::AbstractMatrix{<:Real}, scenario_trees::Vector{<:AbstractMatrix{<:Real}},
    probabilities::AbstractVector{<:Real})
    size(X, 2) == size(c, 2) || error("X and c must have the same number of samples")
    validate_energy_scenarios(c, scenario_trees, probabilities)
    return EnergyScenarioBatch(Matrix{Float64}(X), Matrix{Float64}(c),
        [Matrix{Float64}(tree) for tree in scenario_trees], Vector{Float64}(probabilities))
end

function leave_one_out_scenario_tree(annual_costs::AbstractMatrix{<:Real}, target_index::Integer)
    n = size(annual_costs, 2)
    1 <= target_index <= n || error("target_index is out of bounds")
    n > 1 || error("leave-one-out requires at least two annual observations")
    return Matrix{Float64}(annual_costs[:, [i for i in 1:n if i != target_index]])
end

_to_date(value) = value isa Date ? value : Date(first(string(value), 10))
_present(row, columns) = all(column -> !ismissing(row[column]) && isfinite(Float64(row[column])), columns)

function _check_monthly_table(monthly::DataFrame)
    for k in 2:nrow(monthly)
        monthly.mes[k] == monthly.mes[k - 1] + Month(1) ||
            error("mensal_SE.csv must have one row per consecutive month (problem at $(monthly.mes[k]))")
    end
    all(day.(monthly.mes) .== 1) || error("monthly dates must be the first day of the month")
    return true
end

const ENERGY_DECISION_COLUMNS = [:ena_prev, :ear_prev, :cmo_prev, :cmo_exante, :ipca_t,
    :pld_min, :pld_max, :pld_media_12m_prev]
_energy_target_columns() = [Symbol("pld_h" * lpad(h, 2, '0')) for h in 1:ENERGY_HORIZON]

"""
    load_energy_dataset(data_dir)

Load `mensal_SE.csv` and `decisoes_SE.csv` (built by the dataset notebook) and build the dataset.
"""
function load_energy_dataset(data_dir::AbstractString = "data/training_dataset")
    monthly = CSV.read(joinpath(data_dir, "mensal_SE.csv"), DataFrame)
    decisions = CSV.read(joinpath(data_dir, "decisoes_SE.csv"), DataFrame)
    return build_energy_dataset(monthly, decisions)
end

"""
    build_energy_dataset(monthly, decisions)

One sample per decision date with complete features and twelve realized target months; incomplete
dates are excluded. `monthly` has columns mes, ena, ear, cmo, pld, pld_min, pld_max, ipca (nominal);
`decisions` has data_decisao, the columns in ENERGY_DECISION_COLUMNS and pld_h01..pld_h12.
"""
function build_energy_dataset(monthly::DataFrame, decisions::DataFrame)
    monthly = copy(monthly)
    decisions = copy(decisions)
    monthly.mes = _to_date.(monthly.mes)
    decisions.data_decisao = _to_date.(decisions.data_decisao)
    sort!(monthly, :mes)
    sort!(decisions, :data_decisao)
    _check_monthly_table(monthly)
    allunique(decisions.data_decisao) || error("duplicate decision dates")
    all(day.(decisions.data_decisao) .== 1) || error("decision dates must be the first day of a month")

    # Real values are expressed at the last available price level (a constant unit, no information).
    ipca_ref = Float64(last(collect(skipmissing(monthly.ipca))))
    target_columns = _energy_target_columns()
    complete = [_present(row, vcat(ENERGY_DECISION_COLUMNS, target_columns)) for row in eachrow(decisions)]
    excluded = decisions.data_decisao[.!complete]
    isempty(excluded) || @info "decision dates excluded for incomplete data" count = length(excluded) dates = excluded
    decisions = decisions[complete, :]
    n = nrow(decisions)
    n > 0 || error("no decision date with complete data")

    pld_by_month = Dict(row.mes => Float64(row.pld) for row in eachrow(monthly) if !ismissing(row.pld))
    X = zeros(length(ENERGY_FEATURE_NAMES), n)
    c = zeros(ENERGY_HORIZON, n)
    target_months = Vector{Vector{Date}}(undef, n)
    cmo_real_prev = zeros(n)
    pld_min_real = zeros(n)
    pld_max_real = zeros(n)
    pld_trailing_real = zeros(n)
    price_level = zeros(n)
    for (i, row) in enumerate(eachrow(decisions))
        t = row.data_decisao
        months = [t + Month(h - 1) for h in 1:ENERGY_HORIZON]
        target = [Float64(row[column]) for column in target_columns]
        # the target must be the PLD of the twelve months that start at the decision date
        all(haskey(pld_by_month, m) for m in months) &&
            isapprox(target, [pld_by_month[m] for m in months]; rtol = 1e-9) ||
            error("targets of decision $t do not match the monthly PLD")
        # real terms: deflated with the price level known at t (the CMO of month m-1 one month early)
        price_level[i] = Float64(row.ipca_t) / ipca_ref
        cmo_real_prev[i] = Float64(row.cmo_prev) / price_level[i]
        pld_min_real[i] = Float64(row.pld_min) / price_level[i]
        pld_max_real[i] = Float64(row.pld_max) / price_level[i]
        pld_trailing_real[i] = Float64(row.pld_media_12m_prev) / price_level[i]
        month_value = month(t)
        X[:, i] = [log(Float64(row.ena_prev)), cmo_real_prev[i], Float64(row.ear_prev),
            Float64(row.cmo_exante) / price_level[i], pld_trailing_real[i], pld_min_real[i],
            pld_max_real[i], sin(2π * month_value / 12), cos(2π * month_value / 12)]
        c[:, i] = target ./ price_level[i]
        target_months[i] = months
    end
    return EnergyDataset(X, c, decisions.data_decisao, target_months, year.(decisions.data_decisao),
        month.(decisions.data_decisao), Float64.(decisions.ena_prev), Float64.(decisions.ear_prev),
        cmo_real_prev, pld_min_real, pld_max_real, pld_trailing_real, price_level, monthly, ipca_ref)
end

"""
    energy_decisions_from_monthly(monthly)

Decision table built from a monthly table alone (used for synthetic histories, which have no weekly
CMO): same columns as decisoes_SE.csv, with the ex-ante CMO approximated by the bounded CMO of the
previous month and the price index of month m-2 as the index known at the decision.
"""
function energy_decisions_from_monthly(monthly::DataFrame)
    table = sort(copy(monthly), :mes)
    _check_monthly_table(table)
    rows = NamedTuple[]
    for k in 13:nrow(table)
        t = table.mes[k]
        previous = k - 1
        lower = table.pld_min[k]
        upper = table.pld_max[k]
        targets = [k + h - 1 <= nrow(table) ? table.pld[k + h - 1] : missing for h in 1:ENERGY_HORIZON]
        row = (data_decisao = t, ena_prev = table.ena[previous], ear_prev = table.ear[previous],
            cmo_prev = table.cmo[previous], cmo_exante = clamp(table.cmo[previous], lower, upper),
            ipca_t = table.ipca[k - 2], pld_min = lower, pld_max = upper,
            pld_media_12m_prev = mean(table.pld[(k - 12):(k - 1)]))
        target_row = NamedTuple{Tuple(_energy_target_columns())}(Tuple(targets))
        push!(rows, merge(row, target_row))
    end
    return DataFrame(rows)
end

function fit_energy_normalization(X::AbstractMatrix{<:Real}, c::AbstractMatrix{<:Real})
    feature_mean = vec(mean(X, dims = 2))
    cost_mean = vec(mean(c, dims = 2))
    feature_scale = vec(std(X, dims = 2, corrected = false))
    cost_scale = vec(std(c, dims = 2, corrected = false))
    feature_scale[feature_scale .< eps()] .= 1.0
    cost_scale[cost_scale .< eps()] .= 1.0
    return EnergyNormalization(feature_mean, feature_scale, cost_mean, cost_scale)
end

function normalize_energy_data(X::AbstractMatrix{<:Real}, c::AbstractMatrix{<:Real}, normalization::EnergyNormalization)
    return ((Matrix{Float64}(X) .- normalization.feature_mean) ./ normalization.feature_scale,
        (Matrix{Float64}(c) .- normalization.cost_mean) ./ normalization.cost_scale)
end

denormalize_energy_cost(c::AbstractMatrix{<:Real}, normalization::EnergyNormalization) =
    Matrix{Float64}(c) .* normalization.cost_scale .+ normalization.cost_mean

denormalize_energy_cost(c::AbstractVector{<:Real}, normalization::EnergyNormalization) =
    Vector{Float64}(c) .* normalization.cost_scale .+ normalization.cost_mean
