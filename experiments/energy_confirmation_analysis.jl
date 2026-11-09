#=
Analysis of the confirmation run (protocol "Protocolo da confirmação — SPO+ com CVaR").

Reads the results.csv of the two confirmation runs (block A and block B), keeps the SPO+ and MSE
regrets of each history and computes, per history, r = (regret SPO+ - regret MSE) / regret MSE.

- Primary endpoint (H1): exact one-sided Wilcoxon signed-rank test (r < 0) over all histories.
- Secondary endpoint (H2): the same test in the subgroup with ridge lambda >= 1.
- Descriptive: wins, sign test, median and mean of r with a 95% bootstrap interval for the median,
  per block and per configuration; MSE versus the seasonal mean; scenario coverage.

Usage (from the project root):
    julia scripts/energy_confirmation_analysis.jl energy_confirmation_A energy_confirmation_B
Writes confirmation_report.txt and confirmation_by_configuration.csv in the current folder.
=#

include("energy_portfolio_synthetic.jl")

function _confirmation_rows(directories)
    tables = DataFrame[]
    for (k, directory) in enumerate(directories)
        table = CSV.read(joinpath(directory, "results.csv"), DataFrame)
        table.block = fill(string('A' + k - 1), nrow(table))
        push!(tables, table)
    end
    rows = vcat(tables...)
    keys_history = [:block, :b_ye, :spike, :noise_scale, :replication]
    spo = rows[rows.method .== "SPO+", vcat(keys_history, [:regret, :ridge_lambda, :coverage_90])]
    mse = rows[rows.method .== "MSE", vcat(keys_history, [:regret])]
    media = rows[rows.method .== "Média", vcat(keys_history, [:regret])]
    rename!(spo, :regret => :regret_spo)
    rename!(mse, :regret => :regret_mse)
    rename!(media, :regret => :regret_media)
    histories = innerjoin(innerjoin(spo, mse, on = keys_history), media, on = keys_history)
    histories.r = _relative_differences(histories.regret_spo, histories.regret_mse)
    histories.r_mse_media = _relative_differences(histories.regret_mse, histories.regret_media)
    return histories
end

function _bootstrap_median_interval(values; draws = 10_000, rng = MersenneTwister(2026))
    isempty(values) && return (NaN, NaN)
    medians = [median(rand(rng, values, length(values))) for _ in 1:draws]
    return (quantile(medians, 0.025), quantile(medians, 0.975))
end

"""One-sided exact sign test: probability of at least `wins` successes out of wins + losses under 1/2."""
function _sign_test_p_one_sided(wins::Integer, losses::Integer)
    n = wins + losses
    n == 0 && return 1.0
    return Float64(sum(binomial(big(n), k) for k in wins:n) / big(2)^n)
end

function _confirmation_summary(group)
    defined = filter(isfinite, group.r)
    effective = filter(x -> abs(x) >= 1e-9, defined)
    wins = count(<(0), effective)
    losses = count(>(0), effective)
    low, high = _bootstrap_median_interval(defined)
    return (histories = nrow(group), undefined_r = nrow(group) - length(defined),
        ties = length(defined) - length(effective), spo_better = wins, mse_better = losses,
        median_r = isempty(defined) ? NaN : median(defined), median_r_low = low, median_r_high = high,
        mean_r = isempty(defined) ? NaN : mean(defined),
        sign_test_p_one_sided = _sign_test_p_one_sided(wins, losses),
        wilcoxon_p_one_sided = _wilcoxon_signed_rank_p(effective; alternative = :less),
        mse_better_than_media = count(<(0), filter(isfinite, group.r_mse_media)),
        coverage_90 = mean(group.coverage_90))
end

function analyze_energy_confirmation(directories; report_path = "confirmation_report.txt",
    table_path = "confirmation_by_configuration.csv")
    histories = _confirmation_rows(directories)
    overall = _confirmation_summary(histories)
    subgroup = _confirmation_summary(histories[histories.ridge_lambda .>= 1, :])
    by_block = combine(groupby(histories, :block), _confirmation_summary)
    by_configuration = combine(groupby(histories, [:block, :b_ye, :spike, :noise_scale]), _confirmation_summary)
    CSV.write(table_path, by_configuration)
    open(report_path, "w") do io
        println(io, "Confirmação SPO+ (C4 com transferência) contra MSE — r = (regret SPO+ - regret MSE) / regret MSE")
        println(io, "Pastas: $(join(directories, ", "))")
        println(io)
        println(io, "H1 (principal): Wilcoxon unilateral, todas as histórias")
        println(io, "  $(overall)")
        println(io, "  Resultado: ", overall.wilcoxon_p_one_sided < 0.05 ? "CONFIRMADA (p < 0,05)" : "NÃO CONFIRMADA (p ≥ 0,05)")
        println(io)
        println(io, "H2 (secundária): subgrupo com lambda do ridge >= 1")
        println(io, "  $(subgroup)")
        println(io, "  Resultado: ", subgroup.wilcoxon_p_one_sided < 0.05 ? "CONFIRMADA (p < 0,05)" : "NÃO CONFIRMADA (p ≥ 0,05)")
        println(io)
        println(io, "Por bloco:")
        show(io, MIME("text/plain"), by_block; allcols = true)
        println(io)
        println(io, "Por configuração: ver $(table_path)")
    end
    println(read(report_path, String))
    return (histories = histories, overall = overall, subgroup = subgroup, by_block = by_block,
        by_configuration = by_configuration)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 1 || error("usage: julia scripts/energy_confirmation_analysis.jl <dir A> <dir B>")
    analyze_energy_confirmation(ARGS)
end
