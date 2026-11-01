#=
Conditional PLD scenario generator (G1: inflow only; G2: inflow + stored energy).

Three layers, fitted by OLS on the monthly table of one fold:
  1  inflow anomaly      a_{τ+1} = φ_ζ a_τ + ε^a
  1b stored energy (G2)  E_{τ+1} = Proj_[0,100](Λ(ν;γ^e) + ρ_e E_τ + b_ea a_{τ+1} + ε^e)
  2  log real CMO        y_{τ+1} = Λ(ν;γ^y) + ρ_y y_τ + b_ya a_{τ+1} + [b_yl a_τ (G1) | b_ye χ(E_τ) (G2)] + ε^y
  3  settlement map      PLD = Proj_[L,U](Proj_[L,U](ϖ (e^y - ψ)) + ϖ ε^b)
Innovations (ε^a, ε^e, ε^y) are resampled jointly (one whole month per draw) within the season of
the simulated month; the basis ε^b is resampled separately from the pool of the price-formation
regime (before / since the hourly PLD of 2021). The state is propagated without censoring.

Expected columns of the monthly table: mes, ena, ear, cmo, pld, pld_min, pld_max, ipca.
=#

using DataFrames, Dates, LinearAlgebra, Random, Statistics

const ENERGY_WET_MONTHS = (12, 1, 2, 3, 4)
const ENERGY_HOURLY_PLD_START = Date(2021, 1, 1)

_energy_season(month_value::Integer) = month_value in ENERGY_WET_MONTHS ? 1 : 2   # 1 = wet, 2 = dry
_energy_fourier(month_value::Integer) = [1.0, cos(2π * month_value / 12), sin(2π * month_value / 12),
    cos(4π * month_value / 12), sin(4π * month_value / 12)]
_energy_chi(storage::Real) = -log(max(Float64(storage), 1.0) / 100.0)

struct EnergyScenarioGenerator
    variant::Symbol
    cutoff::Date
    psi::Float64
    ipca_ref::Float64
    mu::Vector{Float64}             # mean of log ENA per calendar month
    sigma::Vector{Float64}          # std of log ENA per calendar month
    phi::Vector{Float64}            # inflow persistence [wet, dry]
    gamma_y::Vector{Float64}        # Fourier profile of the log real CMO
    rho_y::Float64
    b_ya::Float64
    b_yl::Float64                   # G1 only (NaN in G2)
    b_ye::Float64                   # G2 only (NaN in G1)
    gamma_e::Vector{Float64}        # G2 only (empty in G1)
    rho_e::Float64                  # G2 only
    b_ea::Float64                   # G2 only
    residuals::Matrix{Float64}      # one row per training month: [ε^a ε^e ε^y] (ε^e = 0 in G1)
    residual_months::Vector{Date}
    pools::Vector{Vector{Int}}      # residual rows per season [wet, dry]
    basis_pre::Vector{Float64}      # basis pool before the hourly PLD
    basis_post::Vector{Float64}     # basis pool since the hourly PLD
end

"""
    fit_energy_scenario_generator(monthly, cutoff; variant, psi, ipca_ref)

Fit the generator using only the months of `monthly` up to `cutoff` (inclusive). `ipca_ref` is the
price-index level that defines real values; it must be the same constant used to build the
dataset features.
"""
function fit_energy_scenario_generator(monthly::DataFrame, cutoff::Date;
    variant::Symbol = :G2, psi::Real = 10.0, ipca_ref::Real, min_pairs::Integer = 36)
    variant in (:G1, :G2) || error("variant must be :G1 or :G2")
    psi_value = Float64(psi)
    data = monthly[monthly.mes .<= cutoff, :]
    required = variant == :G2 ? [:ena, :cmo, :ipca, :ear] : [:ena, :cmo, :ipca]
    usable = [_present(row, required) for row in eachrow(data)]

    # Layer 1: standardized log-inflow anomaly with training moments per calendar month
    months_of_year = month.(data.mes)
    log_ena = [usable[k] ? log(Float64(data.ena[k])) : NaN for k in 1:nrow(data)]
    mu = zeros(12)
    sigma = zeros(12)
    for month_value in 1:12
        month_values = log_ena[usable .& (months_of_year .== month_value)]
        length(month_values) >= 2 || error("not enough training months for calendar month $month_value")
        mu[month_value] = mean(month_values)
        sigma[month_value] = std(month_values)
    end
    a = [(log_ena[k] - mu[months_of_year[k]]) / sigma[months_of_year[k]] for k in 1:nrow(data)]
    y = [usable[k] ? log(Float64(data.cmo[k]) * Float64(ipca_ref) / Float64(data.ipca[k]) + psi_value) : NaN
         for k in 1:nrow(data)]
    E = variant == :G2 ? [usable[k] ? Float64(data.ear[k]) : NaN for k in 1:nrow(data)] : fill(NaN, nrow(data))

    # Consecutive pairs (τ, τ+1), both usable; index k refers to month τ+1
    pair_rows = [k for k in 2:nrow(data) if usable[k] && usable[k - 1] &&
             data.mes[k] == data.mes[k - 1] + Month(1)]
    length(pair_rows) >= min_pairs || error("only $(length(pair_rows)) usable month pairs up to $cutoff")
    current_months = months_of_year[pair_rows]
    seasons = _energy_season.(current_months)
    a_now = a[pair_rows]
    a_prev = a[pair_rows .- 1]

    phi = zeros(2)
    for season in 1:2
        selected = seasons .== season
        phi[season] = sum(a_prev[selected] .* a_now[selected]) / sum(abs2, a_prev[selected])
    end
    residual_a = a_now .- phi[seasons] .* a_prev

    fourier = reduce(vcat, (_energy_fourier(m)' for m in current_months))

    # Layer 1b (G2): stored energy at the end of month τ+1
    gamma_e = Float64[]
    rho_e = NaN
    b_ea = NaN
    residual_e = zeros(length(pair_rows))
    if variant == :G2
        X_e = hcat(fourier, E[pair_rows .- 1], a_now)
        coefficients_e = X_e \ E[pair_rows]
        residual_e = E[pair_rows] .- X_e * coefficients_e
        gamma_e = coefficients_e[1:5]
        rho_e = coefficients_e[6]
        b_ea = coefficients_e[7]
    end

    # Layer 2: log real CMO
    last_column = variant == :G1 ? a_prev : _energy_chi.(E[pair_rows .- 1])
    X_y = hcat(fourier, y[pair_rows .- 1], a_now, last_column)
    coefficients_y = X_y \ y[pair_rows]
    residual_y = y[pair_rows] .- X_y * coefficients_y
    gamma_y = coefficients_y[1:5]
    rho_y = coefficients_y[6]
    b_ya = coefficients_y[7]
    b_yl = variant == :G1 ? coefficients_y[8] : NaN
    b_ye = variant == :G2 ? coefficients_y[8] : NaN

    residuals = hcat(residual_a, residual_e, residual_y)
    pools = [findall(==(season), seasons) for season in 1:2]
    all(!isempty, pools) || error("a season has no residuals up to $cutoff")

    # Layer 3: basis between the published PLD and the bounded monthly CMO, in real terms
    basis_pre = Float64[]
    basis_post = Float64[]
    for row in eachrow(data)
        _present(row, [:pld, :cmo, :pld_min, :pld_max, :ipca]) || continue
        bounded = clamp(Float64(row.cmo), Float64(row.pld_min), Float64(row.pld_max))
        basis = (Float64(row.pld) - bounded) * Float64(ipca_ref) / Float64(row.ipca)
        push!(row.mes >= ENERGY_HOURLY_PLD_START ? basis_post : basis_pre, basis)
    end

    return EnergyScenarioGenerator(variant, cutoff, psi_value, Float64(ipca_ref), mu, sigma, phi,
        gamma_y, rho_y, b_ya, b_yl, b_ye, gamma_e, rho_e, b_ea, residuals, data.mes[pair_rows], pools,
        basis_pre, basis_post)
end

"""
    generate_energy_scenarios(generator, decision_date, ena_prev, ear_prev, cmo_real_prev,
        ipca_t, pld_min, pld_max; horizon, num_scenarios, rng)

Simulate `num_scenarios` PLD trajectories for the `horizon` months that start at `decision_date`
(first day of a month). The conditioning state is the inflow, stored energy and real CMO of the
month before the decision. Prices are nominal at the price level `ipca_t` known at the decision
date and bounded by the limits in force at that date. Returns a `num_scenarios × horizon` matrix.
"""
function generate_energy_scenarios(generator::EnergyScenarioGenerator, decision_date::Date,
    ena_prev::Real, ear_prev::Real, cmo_real_prev::Real, ipca_t::Real, pld_min::Real, pld_max::Real;
    horizon::Integer = 12, num_scenarios::Integer = 200, rng = Random.default_rng())
    day(decision_date) == 1 || error("decision dates must be the first day of a month")
    previous_month = month(decision_date - Month(1))
    a0 = (log(Float64(ena_prev)) - generator.mu[previous_month]) / generator.sigma[previous_month]
    y0 = log(Float64(cmo_real_prev) + generator.psi)
    E0 = Float64(ear_prev)
    nominal = Float64(ipca_t) / generator.ipca_ref
    lower = Float64(pld_min)
    upper = Float64(pld_max)
    # Before any post-2021 month enters the training window, the pre-2021 pool is the only option.
    basis_pool = (decision_date >= ENERGY_HOURLY_PLD_START && !isempty(generator.basis_post)) ?
        generator.basis_post : generator.basis_pre
    start_month = month(decision_date)

    scenarios = zeros(num_scenarios, horizon)
    for s in 1:num_scenarios
        a, y, E = a0, y0, E0
        for h in 1:horizon
            month_value = mod1(start_month + h - 1, 12)
            season = _energy_season(month_value)
            j = rand(rng, generator.pools[season])                    # one whole month of innovations
            basis = isempty(basis_pool) ? 0.0 : rand(rng, basis_pool)
            fourier = _energy_fourier(month_value)
            a_next = generator.phi[season] * a + generator.residuals[j, 1]
            if generator.variant == :G1
                y_next = dot(generator.gamma_y, fourier) + generator.rho_y * y +
                    generator.b_ya * a_next + generator.b_yl * a + generator.residuals[j, 3]
                E_next = E
            else
                # the CMO of the month uses the storage at the beginning of the month (E)
                y_next = dot(generator.gamma_y, fourier) + generator.rho_y * y +
                    generator.b_ya * a_next + generator.b_ye * _energy_chi(E) + generator.residuals[j, 3]
                E_next = clamp(dot(generator.gamma_e, fourier) + generator.rho_e * E +
                    generator.b_ea * a_next + generator.residuals[j, 2], 0.0, 100.0)
            end
            cmo_nominal = nominal * (exp(y_next) - generator.psi)
            scenarios[s, h] = clamp(clamp(cmo_nominal, lower, upper) + nominal * basis, lower, upper)
            a, y, E = a_next, y_next, E_next                          # uncensored state
        end
    end
    return scenarios
end

"""Generate one scenario matrix per dataset sample in `indices` (reproducible per sample)."""
function build_energy_scenario_trees(dataset, generator::EnergyScenarioGenerator, indices;
    num_scenarios::Integer = 200, seed::Integer = 1)
    return [generate_energy_scenarios(generator, dataset.decision_dates[i], dataset.ena_prev[i],
                dataset.ear_prev[i], dataset.cmo_real_prev[i], dataset.ipca_t[i],
                dataset.pld_min[i], dataset.pld_max[i];
                horizon = size(dataset.c, 1), num_scenarios = num_scenarios,
                rng = MersenneTwister(seed + 7919 * i))
            for i in indices]
end

"""Sign and range checks of the fitted coefficients (empty when everything is as expected)."""
function check_energy_scenario_generator(generator::EnergyScenarioGenerator)
    warnings = String[]
    all(0 .< generator.phi .< 1) || push!(warnings, "phi outside (0,1): $(generator.phi)")
    0 < generator.rho_y < 1 || push!(warnings, "rho_y outside (0,1): $(generator.rho_y)")
    generator.b_ya < 0 || push!(warnings, "b_ya not negative: $(generator.b_ya)")
    if generator.variant == :G1
        generator.b_yl <= 0 || push!(warnings, "b_yl positive: $(generator.b_yl)")
    else
        0 < generator.rho_e < 1 || push!(warnings, "rho_e outside (0,1): $(generator.rho_e)")
        generator.b_ea > 0 || push!(warnings, "b_ea not positive: $(generator.b_ea)")
        generator.b_ye > 0 || push!(warnings, "b_ye not positive: $(generator.b_ye)")
    end
    return warnings
end

function describe_energy_scenario_generator(generator::EnergyScenarioGenerator)
    return "generator variant=$(generator.variant) cutoff=$(generator.cutoff) " *
        "pair_rows=$(size(generator.residuals, 1)) psi=$(generator.psi) " *
        "phi_wet=$(generator.phi[1]) phi_dry=$(generator.phi[2]) " *
        "gamma_y=$(generator.gamma_y) rho_y=$(generator.rho_y) b_ya=$(generator.b_ya) " *
        "b_yl=$(generator.b_yl) b_ye=$(generator.b_ye) gamma_e=$(generator.gamma_e) " *
        "rho_e=$(generator.rho_e) b_ea=$(generator.b_ea) " *
        "basis_pre=$(length(generator.basis_pre)) basis_post=$(length(generator.basis_post))"
end

"""Fraction of (sample, month) pair_rows whose realized PLD lies in the central `level` interval."""
function scenario_interval_coverage(scenario_trees, c::AbstractMatrix{<:Real}; level::Real = 0.9)
    lower_probability = (1 - level) / 2
    hits = 0
    total = 0
    for (k, tree) in enumerate(scenario_trees), h in 1:size(tree, 2)
        column = view(tree, :, h)
        hits += quantile(column, lower_probability) <= c[h, k] <= quantile(column, 1 - lower_probability)
        total += 1
    end
    return hits / total
end

"""Ensemble energy score of one trajectory set against the realized vector (lower is better)."""
function scenario_energy_score(tree::AbstractMatrix{<:Real}, realized::AbstractVector{<:Real})
    S = size(tree, 1)
    term_realized = mean(norm(view(tree, s, :) .- realized) for s in 1:S)
    term_spread = sum(norm(view(tree, s, :) .- view(tree, r, :)) for s in 1:S, r in 1:S) / (2 * S^2)
    return term_realized - term_spread
end
