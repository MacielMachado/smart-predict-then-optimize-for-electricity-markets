#=
Data interfaces for the energy portfolio toy problem.
=#

using CSV, DataFrames, Dates, Statistics

struct EnergyScenarioBatch
    X::Matrix{Float64}
    c::Matrix{Float64}
    scenario_trees::Vector{Matrix{Float64}}
    probabilities::Vector{Float64}
end

struct EnergyDataset
    X::Matrix{Float64}
    c::Matrix{Float64}
    scenario_trees::Vector{Matrix{Float64}}
    years::Vector{Int}
    weeks::Vector{Int}
    dates::Vector{Vector{Date}}
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

function _scenario_columns(names_vector)
    columns = filter(name -> startswith(String(name), "val_pld_cenar"), names_vector)
    length(columns) == 4 || error("expected four PLD scenario columns")
    return columns
end

"""
    load_energy_dataset(data_dir)

Load and align PLD, ENA, and scenario data by `TI`. Each sample is one
`(year, week-of-month)` pair, with twelve monthly observations. The fifth
weekly observation in months that contain five Fridays is intentionally
ignored; incomplete years are excluded explicitly.
"""
function load_energy_dataset(data_dir::AbstractString = "data/training_dataset")
    pld = CSV.read(joinpath(data_dir, "pld_historico_SE.csv"), DataFrame)
    scenarios = CSV.read(joinpath(data_dir, "cenarios_pld_cmo_SE.csv"), DataFrame)
    ena = CSV.read(joinpath(data_dir, "ena_semanal_historico_SE.csv"), DataFrame)

    for frame in (pld, scenarios, ena)
        frame.TI = Date.(frame.TI)
    end

    scenario_columns = _scenario_columns(names(scenarios))
    pld = select(pld, [:TI, :PLD_MEDIA_MES])
    scenarios = select(scenarios, :TI, scenario_columns...)
    ena = select(ena, [:TI, :ena_bruta_regiao_mwmed])
    joined = innerjoin(pld, scenarios, ena; on = :TI, makeunique = true)
    sort!(joined, :TI)

    joined.year = year.(joined.TI)
    joined.month = month.(joined.TI)
    joined.week_of_month = zeros(Int, nrow(joined))
    for group in groupby(joined, [:year, :month])
        order = sortperm(group.TI)
        group.week_of_month[order] = 1:nrow(group)
    end
    joined = filter(:week_of_month => <=(4), joined)

    counts = combine(groupby(joined, [:year, :month]), nrow => :count)
    valid_years = Int[]
    for year_value in sort(unique(joined.year))
        year_counts = filter(:year => ==(year_value), counts)
        if nrow(year_counts) == 12 && all(year_counts.count .== 4)
            push!(valid_years, year_value)
        end
    end
    isempty(valid_years) && error("no complete years with four observations per month")
    joined = filter(:year => in(valid_years), joined)

    index = Dict{Tuple{Int, Int, Int}, Int}()
    for (row_index, row) in enumerate(eachrow(joined))
        key = (row.year, row.week_of_month, row.month)
        haskey(index, key) && error("duplicate aligned key $key")
        index[key] = row_index
    end

    n_samples = 4 * length(valid_years)
    X = zeros(12, n_samples)
    c = zeros(12, n_samples)
    trees = Matrix{Float64}[]
    sample_years = Int[]
    sample_weeks = Int[]
    sample_dates = Vector{Date}[]
    for (sample_index, year_value) in enumerate(valid_years)
        for week_value in 1:4
            column = 4 * (sample_index - 1) + week_value
            tree = zeros(4, 12)
            dates = Date[]
            for month_value in 1:12
                row_index = index[(year_value, week_value, month_value)]
                row = joined[row_index, :]
                X[month_value, column] = Float64(row.ena_bruta_regiao_mwmed)
                c[month_value, column] = Float64(row.PLD_MEDIA_MES)
                for scenario_index in 1:4
                    tree[scenario_index, month_value] = Float64(row[scenario_columns[scenario_index]])
                end
                push!(dates, row.TI)
            end
            push!(trees, tree)
            push!(sample_years, year_value)
            push!(sample_weeks, week_value)
            push!(sample_dates, dates)
        end
    end
    return EnergyDataset(X, c, trees, sample_years, sample_weeks, sample_dates)
end

function energy_split_indices(dataset::EnergyDataset;
    validation_years = [2007, 2018], test_years = [2006, 2019])
    validation = findall(year_value -> year_value in validation_years, dataset.years)
    test = findall(year_value -> year_value in test_years, dataset.years)
    train = findall(year_value -> !(year_value in vcat(validation_years, test_years)), dataset.years)
    return (train = train, validation = validation, test = test)
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
