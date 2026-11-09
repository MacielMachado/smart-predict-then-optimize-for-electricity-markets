using Cairo, CSV, DataFrames, Dates, Fontconfig, Gadfly, Statistics

"""
A fold of the time-respecting evaluation. Validation is used for model selection (ridge lambda and
SPO+ epoch), so its realized prices must not reach the test period, and training targets must not
reach the validation period:

    train: target months end by `train_cutoff`
    validation: decisions in `validation_years` (targets end before the test decisions start)
    test: decisions in `test_years`

Two scenario generators: one fitted up to `train_cutoff` (training and validation scenarios) and one
fitted up to `test_generator_cutoff` (test scenarios), the last month before the test decisions.

After the ridge lambda and the SPO+ epoch are chosen, the models can be refitted on `refit_indices`:
every decision whose twelve target months end by `test_generator_cutoff` (training, validation and
the decisions in between), i.e. everything known before the first test decision.
"""
struct EnergyYearFold
    test_year::Int                      # first test year (label of the fold)
    validation_year::Int                # first validation year
    train_years::Vector{Int}
    train_indices::Vector{Int}
    validation_indices::Vector{Int}
    test_indices::Vector{Int}
    refit_indices::Vector{Int}          # decisions whose targets end by test_generator_cutoff
    generator_cutoff::Date              # train_cutoff: last month seen by training and its generator
    test_generator_cutoff::Date         # last month used by the generator of the test scenarios
end

const ENERGY_METHODS = ["Média", "MSE", "SPO+"]

struct EnergyYearEvaluation
    test_year::Int
    validation_year::Int
    ridge_lambda::Float64
    spo_best_epoch::Int
    cost::Vector{Float64}               # per method, in ENERGY_METHODS order (real R$)
    cvar::Vector{Float64}
    objective::Vector{Float64}
    regret::Vector{Float64}             # mean regret per decision, real R$ (objective - hindsight optimum)
    normalized_regret::Vector{Float64}  # sum(regret) / sum(|hindsight optimum|); unstable when the optimum changes sign
    mse::Vector{Float64}
    informative::Bool                   # false when the three methods reach (almost) the same objective
end

"""A year is informative when the objectives of the methods differ by more than `tolerance` (relative)."""
_energy_informative(objectives; tolerance = 1e-3) =
    (maximum(objectives) - minimum(objectives)) > tolerance * max(maximum(abs, objectives), 1.0)

"""
Generic fold: training targets end by `train_cutoff`, validation and test decisions in the given
years. Validation decisions must have their targets end before the first test decision.
"""
function energy_holdout_fold(dataset::EnergyDataset; train_cutoff::Date,
    validation_years::AbstractVector{<:Integer}, test_years::AbstractVector{<:Integer})
    train_indices = findall(months -> last(months) <= train_cutoff, dataset.target_months)
    validation_indices = findall(in(validation_years), dataset.years)
    test_indices = findall(in(test_years), dataset.years)
    first_test = isempty(test_indices) ? nothing : minimum(dataset.decision_dates[test_indices])
    if !isnothing(first_test) && !isempty(validation_indices)
        all(last(dataset.target_months[i]) < first_test for i in validation_indices) ||
            error("validation targets overlap the test period")
    end
    all(dataset.decision_dates[i] > train_cutoff for i in validation_indices) ||
        error("validation decisions must come after the training cutoff")
    test_cutoff = isnothing(first_test) ? train_cutoff : first_test - Month(1)
    refit_indices = findall(months -> last(months) <= test_cutoff, dataset.target_months)
    isempty(intersect(refit_indices, test_indices)) || error("refit decisions overlap the test decisions")
    train_years = sort(unique(dataset.years[train_indices]))
    return EnergyYearFold(minimum(test_years), minimum(validation_years), train_years,
        train_indices, validation_indices, test_indices, refit_indices, train_cutoff, test_cutoff)
end

"""
Rolling-origin fold for `test_year` with `validation_span` validation years: validation decisions in
`test_year - 1 - validation_span`, ..., `test_year - 2` (targets end by November of `test_year - 1`),
training targets end by December of the year before the first validation year. Decisions made in
`test_year - 1` are left out because their targets reach the test year. A longer validation span
makes the choice of lambda and of the SPO+ epoch less noisy at the cost of training years.
"""
function _rolling_origin_fold(dataset::EnergyDataset, test_year::Integer; validation_span::Integer = 1)
    validation_span >= 1 || error("validation_span must be at least 1")
    first_validation_year = test_year - 1 - validation_span
    return energy_holdout_fold(dataset; train_cutoff = Date(first_validation_year - 1, 12, 1),
        validation_years = collect(first_validation_year:(test_year - 2)), test_years = [test_year])
end

function build_energy_year_folds(dataset::EnergyDataset;
    evaluation_mode::Symbol = :rolling_origin,
    first_test_year = nothing, last_test_year = nothing, min_train_years::Integer = 5,
    validation_span::Integer = 1)
    # With twelve-month targets, leave-one-year-out would train on later years.
    evaluation_mode == :rolling_origin || error("evaluation_mode must be :rolling_origin")
    folds = EnergyYearFold[]
    for test_year in sort(unique(dataset.years))
        !isnothing(first_test_year) && test_year < first_test_year && continue
        !isnothing(last_test_year) && test_year > last_test_year && continue
        fold = _rolling_origin_fold(dataset, test_year; validation_span)
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

function _summary_statistics(results::Vector{EnergyYearEvaluation})
    rows = DataFrame(method = String[], cost_mean = Float64[], cost_std = Float64[],
        cvar_mean = Float64[], cvar_std = Float64[], regret_mean = Float64[], regret_std = Float64[],
        regret_mean_informative = Float64[], normalized_regret_mean = Float64[],
        mse_mean = Float64[], mse_std = Float64[], years_best_regret = Int[])
    informative = [result.informative for result in results]
    best = [argmin(result.regret) for result in results[informative]]
    for (k, name) in enumerate(ENERGY_METHODS)
        costs = [result.cost[k] for result in results]
        cvars = [result.cvar[k] for result in results]
        regrets = [result.regret[k] for result in results]
        normalized = [result.normalized_regret[k] for result in results]
        mses = [result.mse[k] for result in results]
        informative_regrets = regrets[informative]
        push!(rows, (name, mean(costs), std(costs), mean(cvars), std(cvars), mean(regrets),
            std(regrets), isempty(informative_regrets) ? NaN : mean(informative_regrets),
            mean(normalized), mean(mses), std(mses), count(==(k), best)))
    end
    return rows
end

"""Two-sided exact sign test p-value for `wins` versus `losses` (ties dropped)."""
function _sign_test_p(wins::Integer, losses::Integer)
    n = wins + losses
    n == 0 && return 1.0
    tail = sum(binomial(big(n), k) for k in 0:min(wins, losses)) / big(2)^n
    return Float64(min(1, 2 * tail))
end

"""
Exact Wilcoxon signed-rank test of paired differences (zeros dropped, tied absolute values get
average ranks). Under the null hypothesis every sign is equally likely, so the distribution of W+
(sum of the ranks of the positive differences) is obtained by enumerating the 2^n sign patterns with
a recursion over the ranks. `alternative = :two_sided` (default), `:less` (the differences tend to be
negative, e.g. "the first method has smaller regret") or `:greater`.
"""
function _wilcoxon_signed_rank_p(differences; alternative::Symbol = :two_sided)
    alternative in (:two_sided, :less, :greater) || error("unknown alternative $alternative")
    values = [Float64(x) for x in differences if isfinite(x) && x != 0]
    n = length(values)
    n == 0 && return 1.0
    magnitudes = abs.(values)
    order = sortperm(magnitudes)
    ranks = zeros(n)
    k = 1
    while k <= n                                       # average ranks for ties
        j = k
        while j < n && magnitudes[order[j + 1]] == magnitudes[order[k]]
            j += 1
        end
        ranks[order[k:j]] .= (k + j) / 2
        k = j + 1
    end
    doubled = round.(Int, 2 .* ranks)                  # integer half-ranks
    total = sum(doubled)
    distribution = zeros(total + 1)                    # probability of each value of 2 W+
    distribution[1] = 1.0
    for r in doubled
        shifted = zeros(total + 1)
        shifted[(r + 1):end] .= distribution[1:(end - r)]
        distribution .= 0.5 .* (distribution .+ shifted)
    end
    observed = sum(doubled[values .> 0])
    lower = sum(distribution[1:(observed + 1)])        # P(W+ <= observed)
    upper = sum(distribution[(observed + 1):end])      # P(W+ >= observed)
    alternative == :less && return min(1.0, lower)
    alternative == :greater && return min(1.0, upper)
    return min(1.0, 2 * min(lower, upper))
end

"""Relative differences (a - b) / b, NaN where b is not positive."""
_relative_differences(a, b) = [y > 0 ? (x - y) / y : NaN for (x, y) in zip(a, b)]

"""
Number of paired observations needed to detect a mean relative improvement `effect` with 80% power at
the 5% (two-sided) level, given the standard deviation of the relative differences (normal approximation).
"""
_required_pairs(relative_sd, effect) = ceil(Int, (2.8 * relative_sd / effect)^2)

"""
Paired comparison over test years of the absolute regret (real R\$ per decision), for every pair of
methods: mean difference (first minus second; negative = first better), its standard error, wins,
losses and the sign-test p-value; once with all years and once with the informative years only.
"""
function _paired_comparisons(results::Vector{EnergyYearEvaluation})
    table = DataFrame(comparison = String[], years = String[], n = Int[], mean_difference = Float64[],
        se_difference = Float64[], median_relative_difference = Float64[], wins = Int[], losses = Int[],
        sign_test_p = Float64[], wilcoxon_p = Float64[])
    pairs_to_compare = [(3, 2), (3, 1), (2, 1)]                    # SPO+ vs MSE, SPO+ vs Média, MSE vs Média
    for (a, b) in pairs_to_compare, (label, selected) in (("todos", results),
            ("informativos", filter(result -> result.informative, results)))
        isempty(selected) && continue
        first_regrets = [result.regret[a] for result in selected]
        second_regrets = [result.regret[b] for result in selected]
        differences = first_regrets .- second_regrets
        relative = filter(isfinite, _relative_differences(first_regrets, second_regrets))
        wins = count(<(0), differences)
        losses = count(>(0), differences)
        se = length(differences) > 1 ? std(differences) / sqrt(length(differences)) : NaN
        push!(table, ("$(ENERGY_METHODS[a]) - $(ENERGY_METHODS[b])", label, length(differences),
            mean(differences), se, isempty(relative) ? NaN : median(relative), wins, losses,
            _sign_test_p(wins, losses), _wilcoxon_signed_rank_p(differences)))
    end
    return table
end

function _results_table(results::Vector{EnergyYearEvaluation})
    table = DataFrame(test_year = Int[], method = String[], cost = Float64[], cvar = Float64[],
        objective = Float64[], regret = Float64[], normalized_regret = Float64[], mse = Float64[],
        ridge_lambda = Float64[], spo_best_epoch = Int[], informative = Bool[])
    for result in results, (k, name) in enumerate(ENERGY_METHODS)
        push!(table, (result.test_year, name, result.cost[k], result.cvar[k], result.objective[k],
            result.regret[k], result.normalized_regret[k], result.mse[k], result.ridge_lambda,
            result.spo_best_epoch, result.informative))
    end
    return table
end

function _write_summary(path, results, summary)
    paired = _paired_comparisons(results)
    CSV.write(path * ".csv", summary)
    CSV.write(path * "_by_year.csv", _results_table(results))
    CSV.write(path * "_paired.csv", paired)
    open(path * ".txt", "w") do io
        println(io, "Valores monetários em R\$ reais (nível de preços ipca_ref).")
        println(io, "Regret = objetivo da decisão - objetivo ótimo com o PLD realizado, médio por decisão (R\$).")
        println(io, "Ano informativo = os objetivos dos três métodos diferem em mais de 0,1%.")
        println(io, "")
        println(io, "ano | informativo | método | custo | CVaR | regret | regret normalizado | MSE")
        for result in results, (k, name) in enumerate(ENERGY_METHODS)
            println(io, "$(result.test_year) | $(result.informative) | $name | $(result.cost[k]) | " *
                "$(result.cvar[k]) | $(result.regret[k]) | $(result.normalized_regret[k]) | $(result.mse[k])")
        end
        println(io, "")
        println(io, "Método | custo médio | CVaR médio | regret médio | regret médio (informativos) | anos informativos com menor regret")
        for row in eachrow(summary)
            println(io, "$(row.method) | $(row.cost_mean) | $(row.cvar_mean) | $(row.regret_mean) | " *
                "$(row.regret_mean_informative) | $(row.years_best_regret)")
        end
        println(io, "")
        println(io, "Comparação pareada do regret (negativo = primeiro método melhor)")
        println(io, "comparação | anos | n | diferença média | erro-padrão | diferença relativa mediana | " *
            "vitórias | derrotas | p (teste do sinal) | p (Wilcoxon, bilateral)")
        for row in eachrow(paired)
            println(io, "$(row.comparison) | $(row.years) | $(row.n) | $(row.mean_difference) | " *
                "$(row.se_difference) | $(row.median_relative_difference) | $(row.wins) | $(row.losses) | " *
                "$(row.sign_test_p) | $(row.wilcoxon_p)")
        end
    end
end

function _save_cv_plots(results, output_dir)
    mkpath(joinpath(output_dir, "plots"))
    table = _results_table(results)
    function save_metric(filename, title, ylabel, column)
        chart = plot(table, x = :test_year, y = column, color = :method,
            Geom.line, Geom.point, Guide.title(title), Guide.xlabel("Test year"),
            Guide.ylabel(ylabel), Scale.color_discrete_manual("#7f7f7f", "#1b9e77", "#d95f02"),
            Theme(background_color = "white", panel_fill = "white"))
        draw(PNG(joinpath(output_dir, "plots", filename), 8inch, 5inch), chart)
    end
    save_metric("regret_by_year.png", "Regret by test year", "Regret per decision (real R\$)", :regret)
    save_metric("normalized_regret_by_year.png", "Normalized regret by test year", "Normalized regret", :normalized_regret)
    save_metric("cost_by_year.png", "Cost by test year", "Cost (real R\$)", :cost)
    save_metric("cvar_by_year.png", "CVaR by test year", "CVaR (real R\$)", :cvar)
    save_metric("mse_by_year.png", "MSE by test year", "MSE (normalized)", :mse)
end
