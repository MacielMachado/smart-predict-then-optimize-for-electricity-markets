#=
Controlled experiment: the G2 generator fitted to the real data is used as a known data-generating
process ("truth"). Synthetic monthly histories are simulated from it, the same pipeline as the real
case is run on each history (own generator fitted per fold, same features and anchor, ridge lambda
and SPO+ epoch chosen on validation, refit before the test), and the methods are compared on the
test decisions.

Knobs (every combination is run, `replications` histories each):
- `b_ye_values`: absolute strength of the smooth storage effect χ(E) = -log(E/100) on the log CMO.
  The value fitted on the real data is small (see truth.txt), so absolute values such as 0.25 or 0.5
  are needed to make the predictor clearly misspecified. The CMO is persistent (rho_y ≈ 0.84), so the
  long-run effect is about 1/(1 - rho_y) ≈ 6 times b_ye and the price volatility grows quickly with it;
  values above about 0.5 push most prices to the floor or the ceiling;
- `spike_values` (with `spike_threshold`): a jump of the log CMO when storage is below the threshold,
  a threshold nonlinearity;
- `noise_scales`: multiplies the CMO innovations ε^y.
For every combination the CMO intercept is shifted so that the long-run mean of the log CMO stays
the same as in the fitted truth; otherwise a stronger storage effect would also mean higher prices.

Layout of one history (years counted from `first_year`; the first year only feeds the 12-month
trailing mean): training decisions in the next `train_years` years, `validation_years` validation
years, one gap year (its targets would reach the test period), then `test_years` test years.

SPO+ candidates (`spo_candidates`): several SPO+ settings can be compared on the same histories. Each
candidate is a NamedTuple with a `name` and overrides of the SPO+ options (step_size, spo_averaging,
spo_epoch0_eligible, spo_select_epoch, num_epochs, eval_every, batchsize, step_rule, spo_refit, refit).
The generators, scenarios and oracles are shared, so the candidates differ only in the SPO+ path (and,
for `refit = false`, in the absence of the refit of all methods).

Two uses (see the end of the file):
- development: `seed = 1` (the histories already analysed), the four candidates of
  ENERGY_SPO_CANDIDATES and `track_test = true` (test-regret curve along the SPO+ path);
- confirmation: a new seed, ONE frozen candidate, the number of histories fixed in advance.

Outputs (in `output_dir`):
- results.csv: one row per history, candidate and method (Média and MSE repeat across candidates);
- paired_spo_minus_mse.csv: SPO+ minus MSE on the same history, per candidate and knob level and
  pooled over all levels ("todos"): mean and median difference (real R$ per decision), relative
  difference (SPO+ - MSE) / MSE, wins, sign test, exact Wilcoxon (two-sided and one-sided "SPO+
  better", on the relative differences) and the number of histories needed to detect a 5% (and 2%)
  mean relative improvement;
- paired_mse_minus_media.csv: the same for MSE minus the seasonal mean;
- curves.csv (track_test = true): test regret of the SPO+ path and of the MSE (both fitted on the
  training decisions only) at every evaluation epoch, and curves_summary.csv with the mean and median
  relative difference per candidate and epoch.
=#

include("energy_portfolio_run.jl")

const ENERGY_SYNTHETIC_DEFAULTS = (num_epochs = 300, num_scenarios = 100, num_scenarios_test = 300,
    eval_every = 10, track_train = false)

"""Candidate SPO+ settings for the development run (each adds one containment device)."""
const ENERGY_SPO_CANDIDATES = [
    (name = "C0", step_size = 0.1),
    (name = "C1", step_size = 0.02),
    (name = "C2", step_size = 0.02, spo_averaging = true),
    (name = "C3", step_size = 0.02, spo_averaging = true, spo_epoch0_eligible = true,
        num_epochs = 50, eval_every = 5)]

"""
C4: C2 (η₀ = 0.02, mean of the iterates) with a FIXED number of epochs (50): the validation chooses the
ridge lambda but not the SPO+ epoch. Three ways to obtain the model of the test (second development
run): retrain 50 SPO+ steps from the refitted ridge, transfer the correction learned on the training
decisions to the refitted ridge, or no refit at all (reference; then MSE and the seasonal mean are not
refitted either).
"""
const ENERGY_SPO_C4 = (name = "C4", step_size = 0.02, spo_averaging = true, spo_select_epoch = false,
    num_epochs = 50, eval_every = 5)
const ENERGY_SPO_C4_VARIANTS = [
    merge(ENERGY_SPO_C4, (name = "C4_retreino", spo_refit = :retrain)),
    merge(ENERGY_SPO_C4, (name = "C4_transferencia", spo_refit = :transfer)),
    merge(ENERGY_SPO_C4, (name = "C4_sem_retreino", refit = false))]

const _SPO_CANDIDATE_KEYS = (:step_size, :spo_averaging, :spo_epoch0_eligible, :spo_select_epoch, :num_epochs,
    :eval_every, :batchsize, :step_rule, :spo_refit, :refit)

function _candidate_options(candidate)
    haskey(candidate, :name) || error("every SPO+ candidate needs a name")
    overrides = Base.structdiff(candidate, NamedTuple{(:name,)})
    unknown = setdiff(keys(overrides), _SPO_CANDIDATE_KEYS)
    isempty(unknown) || error("candidate $(candidate.name) changes non-SPO+ options $(collect(unknown))")
    return String(candidate.name), overrides
end

function _paired_table(rows, knobs, first_method, second_method)
    keys_history = vcat(knobs, [:candidate, :replication])
    first_rows = rows[rows.method .== first_method, vcat(keys_history, [:regret])]
    second_rows = rows[rows.method .== second_method, vcat(keys_history, [:regret])]
    rename!(first_rows, :regret => :first_regret)
    rename!(second_rows, :regret => :second_regret)
    table = innerjoin(first_rows, second_rows, on = keys_history)
    table.difference = table.first_regret .- table.second_regret
    table.relative = _relative_differences(table.first_regret, table.second_regret)
    function summarize(group)
        differences = group.difference
        relative = filter(isfinite, group.relative)
        relative_sd = length(relative) > 1 ? std(relative) : NaN
        wins = count(<(0), differences)
        losses = count(>(0), differences)
        return (replications = nrow(group),
            mean_difference = mean(differences),
            se_difference = length(differences) > 1 ? std(differences) / sqrt(length(differences)) : NaN,
            median_difference = median(differences),
            mean_relative = isempty(relative) ? NaN : mean(relative),
            median_relative = isempty(relative) ? NaN : median(relative),
            sd_relative = relative_sd,
            first_better = wins, second_better = losses,
            sign_test_p = _sign_test_p(wins, losses),
            wilcoxon_p = _wilcoxon_signed_rank_p(relative),
            wilcoxon_p_first_better = _wilcoxon_signed_rank_p(relative; alternative = :less),
            histories_for_5pct = isfinite(relative_sd) ? _required_pairs(relative_sd, 0.05) : -1,
            histories_for_2pct = isfinite(relative_sd) ? _required_pairs(relative_sd, 0.02) : -1)
    end
    by_level = combine(groupby(table, vcat(knobs, [:candidate])), summarize)
    pooled = combine(groupby(table, [:candidate]), summarize)
    for knob in knobs
        by_level[!, knob] = string.(by_level[!, knob])
        pooled[!, knob] = fill("todos", nrow(pooled))
    end
    return vcat(by_level, select(pooled, names(by_level)))
end

function run_energy_portfolio_synthetic(; data_dir = "data/training_dataset",
    output_dir = "energy_synthetic_results", b_ye_values = [0.0, 0.25, 0.5], spike_values = [0.0],
    spike_threshold = 30.0, noise_scales = [1.0], replications = 20, train_years = 15,
    validation_years = 2, test_years = 10, first_year = 2001, seed = 1, spo_candidates = nothing,
    options...)
    options_used = merge(ENERGY_SYNTHETIC_DEFAULTS, values(options), (seed = seed,))
    resolved_options = _energy_options(options_used)                 # validates the option names
    candidates = spo_candidates === nothing ?
        [(name = "SPO+",)] : collect(spo_candidates)
    candidate_settings = [_candidate_options(candidate) for candidate in candidates]
    allunique(first.(candidate_settings)) || error("candidate names must be unique")

    real_dataset = load_energy_dataset(data_dir)
    last_month = maximum(real_dataset.monthly.mes[.!ismissing.(real_dataset.monthly.pld)])
    truth = fit_energy_scenario_generator(real_dataset.monthly, last_month; variant = :G2,
        psi = resolved_options.psi, ipca_ref = real_dataset.ipca_ref)
    storage = energy_storage_statistics(truth; threshold = spike_threshold)
    pld_min = real_dataset.pld_min_real[end]
    pld_max = real_dataset.pld_max_real[end]

    train_cutoff = Date(first_year + train_years, 12, 1)
    validation_year_list = collect((first_year + train_years + 1):(first_year + train_years + validation_years))
    first_test_year = last(validation_year_list) + 2
    test_year_list = collect(first_test_year:(first_test_year + test_years - 1))
    num_months = 12 * (last(test_year_list) + 2 - first_year)

    mkpath(output_dir)
    open(joinpath(output_dir, "truth.txt"), "w") do io
        println(io, "fitted on real data up to $last_month; constant real limits [$pld_min, $pld_max]")
        println(io, "long-run mean chi(E)=$(storage.mean_chi); share of months with E < $spike_threshold: $(storage.share_below)")
        println(io, "seed=$seed replications=$replications candidates=$(candidates)")
        println(io, "options=$(resolved_options)")
        write_generator_details(io, truth; label = "truth")
    end
    rows = DataFrame(b_ye = Float64[], spike = Float64[], noise_scale = Float64[], replication = Int[],
        candidate = String[], method = String[], regret = Float64[], normalized_regret = Float64[],
        cost = Float64[], cvar = Float64[], mse = Float64[], ridge_lambda = Float64[],
        spo_best_epoch = Int[], coverage_90 = Float64[])
    curves = DataFrame(b_ye = Float64[], spike = Float64[], noise_scale = Float64[], replication = Int[],
        candidate = String[], epoch = Int[], spo_test_regret = Float64[], mse_test_regret = Float64[])

    for b_ye in b_ye_values, spike in spike_values, noise_scale in noise_scales, replication in 1:replications
        # keep the long-run mean of the log CMO equal to the fitted truth
        intercept_shift = (truth.b_ye - b_ye) * storage.mean_chi - spike * storage.share_below
        rng = MersenneTwister(hash((seed, b_ye, spike, noise_scale, replication)))
        monthly = simulate_energy_history(truth, Date(first_year, 1, 1), num_months;
            b_ye, spike, spike_threshold, noise_scale, intercept_shift, pld_min, pld_max, rng)
        dataset = build_energy_dataset(monthly, energy_decisions_from_monthly(monthly))
        fold = energy_holdout_fold(dataset; train_cutoff, validation_years = validation_year_list,
            test_years = test_year_list)
        prepared = prepare_energy_fold(dataset, fold; options_used...)  # shared by the candidates
        history_label = "bye$(b_ye)_spike$(spike)_noise$(noise_scale)_rep$(replication)"
        for (candidate_name, overrides) in candidate_settings
            label = "$(history_label)_$(candidate_name)"
            run_options = merge(options_used, overrides)
            log_file = open(joinpath(output_dir, "$(label).log"), "w")
            result = try
                println(log_file, "synthetic history b_ye=$b_ye spike=$spike spike_threshold=$spike_threshold " *
                    "noise_scale=$noise_scale intercept_shift=$intercept_shift candidate=$candidate_name")
                train_and_evaluate_energy_fold(dataset, fold; log_io = log_file, prepared, run_options...)
            finally
                close(log_file)
            end
            evaluation = result.evaluation
            for (k, name) in enumerate(ENERGY_METHODS)
                push!(rows, (b_ye, spike, noise_scale, replication, candidate_name, name, evaluation.regret[k],
                    evaluation.normalized_regret[k], evaluation.cost[k], evaluation.cvar[k], evaluation.mse[k],
                    evaluation.ridge_lambda, evaluation.spo_best_epoch, result.coverage))
            end
            series = result.history_series
            for (epoch, regret) in zip(series.epochs, series.spo_test_regret)
                isfinite(regret) || continue
                push!(curves, (b_ye, spike, noise_scale, replication, candidate_name, epoch, regret,
                    result.mse_test_regret_before_refit))
            end
            println("$label regret(Média, MSE, SPO+)=$(evaluation.regret) best_epoch=$(evaluation.spo_best_epoch)")
        end
        CSV.write(joinpath(output_dir, "results.csv"), rows)          # partial results survive interruptions
        nrow(curves) > 0 && CSV.write(joinpath(output_dir, "curves.csv"), curves)
    end

    knobs = [:b_ye, :spike, :noise_scale]
    summary = combine(groupby(rows, vcat(knobs, [:candidate, :method])),
        :regret => mean => :regret_mean,
        :regret => (x -> std(x) / sqrt(length(x))) => :regret_se,
        :regret => median => :regret_median,
        :mse => mean => :mse_mean, :coverage_90 => mean => :coverage_90_mean, nrow => :replications)
    paired_spo_mse = _paired_table(rows, knobs, "SPO+", "MSE")
    mse_rows = rows[rows.candidate .== first(first.(candidate_settings)), :]   # identical across candidates
    paired_mse_mean = _paired_table(mse_rows, knobs, "MSE", "Média")
    CSV.write(joinpath(output_dir, "summary.csv"), summary)
    CSV.write(joinpath(output_dir, "paired_spo_minus_mse.csv"), paired_spo_mse)
    CSV.write(joinpath(output_dir, "paired_mse_minus_media.csv"), paired_mse_mean)

    curves_summary = DataFrame()
    if nrow(curves) > 0
        curves.relative = _relative_differences(curves.spo_test_regret, curves.mse_test_regret)
        curves_summary = combine(groupby(curves, [:candidate, :epoch]),
            :relative => (x -> mean(filter(isfinite, x))) => :mean_relative,
            :relative => (x -> median(filter(isfinite, x))) => :median_relative,
            :relative => (x -> count(<(0), x) / length(x)) => :share_spo_better,
            nrow => :histories)
        sort!(curves_summary, [:candidate, :epoch])
        CSV.write(joinpath(output_dir, "curves_summary.csv"), curves_summary)
        chart = plot(curves_summary, x = :epoch, y = :median_relative, color = :candidate,
            Geom.line, Geom.point, yintercept = [0.0], Geom.hline(color = "gray"),
            Guide.title("Test regret of the SPO+ path relative to the MSE (median over histories)"),
            Guide.xlabel("Epoch"), Guide.ylabel("(SPO+ - MSE) / MSE  (negative = SPO+ better)"),
            Theme(background_color = "white", panel_fill = "white"))
        draw(PNG(joinpath(output_dir, "curves_relative_regret.png"), 8inch, 5inch), chart)
    end

    for (candidate_name, _) in candidate_settings
        subset = summary[summary.candidate .== candidate_name, :]
        plot_data = transform(subset, [:regret_mean, :regret_se] => ByRow((m, s) -> m - 1.96s) => :low,
            [:regret_mean, :regret_se] => ByRow((m, s) -> m + 1.96s) => :high)
        knob_x, other_knob = length(b_ye_values) >= length(spike_values) ? (:b_ye, :spike) : (:spike, :b_ye)
        for noise_scale in noise_scales, other_value in unique(plot_data[!, other_knob])
            selected = plot_data[(plot_data.noise_scale .== noise_scale) .& (plot_data[!, other_knob] .== other_value), :]
            nrow(selected) == 0 && continue
            chart = plot(selected, x = knob_x, y = :regret_mean, ymin = :low, ymax = :high, color = :method,
                Geom.point, Geom.line, Geom.errorbar,
                Guide.title("$candidate_name: regret vs $(knob_x) ($(other_knob) = $(other_value), noise scale $(noise_scale))"),
                Guide.xlabel(knob_x == :b_ye ? "Storage effect b_ye on the log CMO" : "Spike below the storage threshold"),
                Guide.ylabel("Regret per decision, real R\$ (mean ± 95%)"),
                Scale.color_discrete_manual("#7f7f7f", "#1b9e77", "#d95f02"),
                Theme(background_color = "white", panel_fill = "white"))
            draw(PNG(joinpath(output_dir, "regret_vs_$(knob_x)_$(other_knob)$(other_value)_noise$(noise_scale)_$(candidate_name).png"),
                8inch, 5inch), chart)
        end
    end
    return (rows = rows, summary = summary, paired = paired_spo_mse, paired_mse_mean = paired_mse_mean,
        curves = curves, curves_summary = curves_summary, truth = truth, storage = storage,
        output_dir = output_dir)
end

"""
Development run: the histories of seed 1, the four SPO+ candidates and the test-regret curves. The
candidate to freeze is chosen on these results (smallest pooled median relative difference, then
most wins) before any confirmation run.
"""
run_energy_portfolio_synthetic_development(; output_dir = "energy_synthetic_development", options...) =
    run_energy_portfolio_synthetic(; output_dir, seed = 1, spo_candidates = ENERGY_SPO_CANDIDATES,
        track_test = true, options...)

"""
Confirmation run: new histories (`seed`, default 2), ONE frozen candidate and a number of
histories per level fixed in advance. Primary endpoint: the pooled one-sided Wilcoxon p-value
(wilcoxon_p_first_better, row b_ye = "todos") below 0.05.
"""
function run_energy_portfolio_synthetic_confirmation(candidate; replications, seed = 2,
    output_dir = "energy_synthetic_confirmation", options...)
    seed == 1 && error("the confirmation must use histories not seen in the development run (seed ≠ 1)")
    return run_energy_portfolio_synthetic(; output_dir, seed, replications, spo_candidates = [candidate],
        options...)
end

"""
Second development run: the same histories (seed 1) with the three C4 variants (refit procedure).
"""
run_energy_portfolio_synthetic_development_c4(; output_dir = "energy_synthetic_development_c4", options...) =
    run_energy_portfolio_synthetic(; output_dir, seed = 1, spo_candidates = ENERGY_SPO_C4_VARIANTS,
        track_test = false, options...)

# julia scripts/energy_portfolio_synthetic.jl            -> second development run (C4 variants)
# julia scripts/energy_portfolio_synthetic.jl dev1       -> first development run (C0 to C3)
if abspath(PROGRAM_FILE) == @__FILE__
    mode = isempty(ARGS) ? "dev_c4" : ARGS[1]
    result = mode == "dev1" ? run_energy_portfolio_synthetic_development() :
        mode == "dev_c4" ? run_energy_portfolio_synthetic_development_c4() :
        error("unknown mode $mode (use dev_c4 or dev1; the confirmation is started from the REPL)")
    println(result.paired)
    nrow(result.curves_summary) > 0 && println(result.curves_summary)
end
