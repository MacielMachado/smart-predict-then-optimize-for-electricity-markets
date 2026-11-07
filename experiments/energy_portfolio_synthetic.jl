#=
Controlled experiment: the G2 generator fitted to the real data is used as a known data-generating
process. Synthetic monthly histories are simulated from it, the same pipeline as the real case is
run on each history (generator fitted per fold, same features and anchor, ridge lambda and SPO+ epoch
chosen on validation), and the three methods are compared on many test decisions.

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

Main output: paired_spo_minus_mse.csv, the SPO+ minus MSE difference of the mean regret per decision
(real R$) on the same history, with its standard error and the sign-test p-value.
=#

include("energy_portfolio_run.jl")

const ENERGY_SYNTHETIC_DEFAULTS = (num_epochs = 300, num_scenarios = 100, num_scenarios_test = 300,
    eval_every = 10, track_train = false)

function run_energy_portfolio_synthetic(; data_dir = "data/training_dataset",
    output_dir = "energy_synthetic_results", b_ye_values = [0.0, 0.25, 0.5], spike_values = [0.0],
    spike_threshold = 30.0, noise_scales = [1.0], replications = 20, train_years = 15,
    validation_years = 2, test_years = 10, first_year = 2001, seed = 1, options...)
    options_used = merge(ENERGY_SYNTHETIC_DEFAULTS, values(options), (seed = seed,))
    resolved_options = _energy_options(options_used)                 # validates the option names
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
        write_generator_details(io, truth; label = "truth")
    end
    rows = DataFrame(b_ye = Float64[], spike = Float64[], noise_scale = Float64[], replication = Int[],
        method = String[], regret = Float64[], normalized_regret = Float64[], cost = Float64[],
        cvar = Float64[], mse = Float64[], ridge_lambda = Float64[], spo_best_epoch = Int[],
        coverage_90 = Float64[])

    for b_ye in b_ye_values, spike in spike_values, noise_scale in noise_scales, replication in 1:replications
        # keep the long-run mean of the log CMO equal to the fitted truth
        intercept_shift = (truth.b_ye - b_ye) * storage.mean_chi - spike * storage.share_below
        rng = MersenneTwister(hash((seed, b_ye, spike, noise_scale, replication)))
        monthly = simulate_energy_history(truth, Date(first_year, 1, 1), num_months;
            b_ye, spike, spike_threshold, noise_scale, intercept_shift, pld_min, pld_max, rng)
        dataset = build_energy_dataset(monthly, energy_decisions_from_monthly(monthly))
        fold = energy_holdout_fold(dataset; train_cutoff, validation_years = validation_year_list,
            test_years = test_year_list)
        label = "bye$(b_ye)_spike$(spike)_noise$(noise_scale)_rep$(replication)"
        log_file = open(joinpath(output_dir, "$(label).log"), "w")
        result = try
            println(log_file, "synthetic history b_ye=$b_ye spike=$spike spike_threshold=$spike_threshold " *
                "noise_scale=$noise_scale intercept_shift=$intercept_shift")
            train_and_evaluate_energy_fold(dataset, fold; log_io = log_file, options_used...)
        finally
            close(log_file)
        end
        evaluation = result.evaluation
        for (k, name) in enumerate(ENERGY_METHODS)
            push!(rows, (b_ye, spike, noise_scale, replication, name, evaluation.regret[k],
                evaluation.normalized_regret[k], evaluation.cost[k], evaluation.cvar[k], evaluation.mse[k],
                evaluation.ridge_lambda, evaluation.spo_best_epoch, result.coverage))
        end
        println("$label regret(Média, MSE, SPO+)=$(evaluation.regret)")
        CSV.write(joinpath(output_dir, "results.csv"), rows)          # partial results survive interruptions
    end

    knobs = [:b_ye, :spike, :noise_scale]
    summary = combine(groupby(rows, vcat(knobs, :method)),
        :regret => mean => :regret_mean,
        :regret => (x -> std(x) / sqrt(length(x))) => :regret_se,
        :normalized_regret => mean => :normalized_regret_mean,
        :mse => mean => :mse_mean, :coverage_90 => mean => :coverage_90_mean, nrow => :replications)
    # Paired comparisons on the same history (same data, same scenarios)
    function paired_differences(first_method, second_method)
        table = combine(groupby(rows, vcat(knobs, :replication)),
            [:method, :regret] => ((m, r) -> r[m .== first_method][1] - r[m .== second_method][1]) => :difference)
        return combine(groupby(table, knobs),
            :difference => mean => :mean_difference,
            :difference => (x -> std(x) / sqrt(length(x))) => :se_difference,
            :difference => (x -> count(<(0), x)) => :first_better,
            :difference => (x -> count(>(0), x)) => :second_better,
            :difference => (x -> _sign_test_p(count(<(0), x), count(>(0), x))) => :sign_test_p,
            nrow => :replications)
    end
    paired_spo_mse = paired_differences("SPO+", "MSE")
    paired_mse_mean = paired_differences("MSE", "Média")
    CSV.write(joinpath(output_dir, "summary.csv"), summary)
    CSV.write(joinpath(output_dir, "paired_spo_minus_mse.csv"), paired_spo_mse)
    CSV.write(joinpath(output_dir, "paired_mse_minus_media.csv"), paired_mse_mean)

    plot_data = transform(summary, [:regret_mean, :regret_se] => ByRow((m, s) -> m - 1.96s) => :low,
        [:regret_mean, :regret_se] => ByRow((m, s) -> m + 1.96s) => :high)
    knob_x, other_knob = length(b_ye_values) >= length(spike_values) ? (:b_ye, :spike) : (:spike, :b_ye)
    for noise_scale in noise_scales, other_value in unique(plot_data[!, other_knob])
        subset = plot_data[(plot_data.noise_scale .== noise_scale) .& (plot_data[!, other_knob] .== other_value), :]
        nrow(subset) == 0 && continue
        chart = plot(subset, x = knob_x, y = :regret_mean, ymin = :low, ymax = :high, color = :method,
            Geom.point, Geom.line, Geom.errorbar,
            Guide.title("Regret vs $(knob_x) ($(other_knob) = $(other_value), noise scale $(noise_scale))"),
            Guide.xlabel(knob_x == :b_ye ? "Storage effect b_ye on the log CMO" : "Spike below the storage threshold"),
            Guide.ylabel("Regret per decision, real R\$ (mean ± 95%)"),
            Scale.color_discrete_manual("#7f7f7f", "#1b9e77", "#d95f02"),
            Theme(background_color = "white", panel_fill = "white"))
        draw(PNG(joinpath(output_dir, "regret_vs_$(knob_x)_$(other_knob)$(other_value)_noise$(noise_scale).png"),
            8inch, 5inch), chart)
    end
    return (rows = rows, summary = summary, paired = paired_spo_mse, paired_mse_mean = paired_mse_mean,
        truth = truth, storage = storage, output_dir = output_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_energy_portfolio_synthetic()
    println(result.summary)
    println(result.paired)
    println(result.paired_mse_mean)
end
