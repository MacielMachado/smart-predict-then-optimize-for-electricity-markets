#=
SPO-compatible energy portfolio oracle with a frozen PLD scenario tree.

The oracle keeps the feasible set fixed and changes only the coefficients of
an evaluation PLD vector c in the deterministic spot objective.
=#

using JuMP, Gurobi, LinearAlgebra
const MOI = JuMP.MOI

struct EnergyPortfolioOracleResult
    objective_value::Float64
    w::Vector{Float64}
    e_plus::Vector{Float64}
    e_minus::Vector{Float64}
    eta::Float64
    xi::Vector{Float64}
    action::Vector{Float64}
    scenario_costs::Vector{Float64}
    expected_scenario_cost::Float64
    cvar::Float64
    status
end

struct EnergyPortfolioOracle
    solve
    horizon::Int
    num_scenarios::Int
    beta::Float64
end

function _validate_energy_oracle_inputs(pld_scenarios, probabilities, demand, contract_prices,
    alpha, beta, gamma, delta_inf, delta_sup, spot_upper_bound)
    S, T = size(pld_scenarios)
    length(probabilities) == S || error("probabilities must have one entry per scenario")
    length(demand) == T || error("demand must have one entry per month")
    length(contract_prices) == T || error("contract_prices must have one entry per month")
    all(isfinite, pld_scenarios) || error("pld_scenarios must be finite")
    all(isfinite, probabilities) || error("probabilities must be finite")
    all(probabilities .>= 0) || error("probabilities must be nonnegative")
    isapprox(sum(probabilities), 1.0; atol = 1e-10) || error("probabilities must sum to one")
    all(demand .>= 0) || error("demand must be nonnegative")
    all(contract_prices .>= 0) || error("contract_prices must be nonnegative")
    0 < alpha < 1 || error("alpha must be strictly between zero and one")
    0 <= beta <= 1 || error("beta must be between zero and one")
    gamma >= 0 || error("gamma must be nonnegative")
    0 <= delta_inf <= delta_sup || error("seasonality bounds are invalid")
    all(spot_upper_bound .>= 0) || error("spot_upper_bound must be nonnegative")
    return S, T
end

"""
    build_energy_oracle(pld_scenarios, probabilities, demand, contract_prices; kwargs...)

Build a reusable oracle for the SPO-compatible energy portfolio problem. The
scenario tree is fixed inside the returned closure. Each call `oracle(c)` only
changes the objective coefficients associated with deterministic spot action.
"""
function build_energy_oracle(pld_scenarios::AbstractMatrix{<:Real},
    probabilities::AbstractVector{<:Real}, demand::AbstractVector{<:Real},
    contract_prices::AbstractVector{<:Real};
    alpha::Real = 0.95,
    beta::Real = 0.0,
    gamma::Real = 1.0,
    delta_inf::Real = 0.90,
    delta_sup::Real = 1.10,
    spot_upper_bound::AbstractVector{<:Real} = collect(demand),
    gurobiEnv = nothing,
    output_flag::Bool = false)

    pld = Matrix{Float64}(pld_scenarios)
    probs = Vector{Float64}(probabilities)
    loads = Vector{Float64}(demand)
    contract_costs = Vector{Float64}(contract_prices)
    spot_bounds = Vector{Float64}(spot_upper_bound)
    alpha_value = Float64(alpha)
    beta_value = Float64(beta)
    S, T = _validate_energy_oracle_inputs(pld, probs, loads, contract_costs,
        alpha_value, beta_value, Float64(gamma), Float64(delta_inf),
        Float64(delta_sup), spot_bounds)

    optimizer_factory = if gurobiEnv === nothing
        () -> Gurobi.Optimizer()
    else
        () -> Gurobi.Optimizer(gurobiEnv)
    end
    model = Model(optimizer_factory)
    set_optimizer_attribute(model, "OutputFlag", output_flag ? 1 : 0)

    @variable(model, w[1:T] >= 0)
    @variable(model, e_plus[1:T] >= 0)
    @variable(model, e_minus[1:T] >= 0)
    @variable(model, eta)
    @variable(model, xi[1:S] >= 0)
    set_upper_bound.(e_plus, spot_bounds)
    set_upper_bound.(e_minus, spot_bounds)

    for t in 1:T
        @constraint(model, w[t] + e_plus[t] - e_minus[t] == loads[t])
        @constraint(model, w[t] >= Float64(gamma) * loads[t])
    end

    total_contracts = sum(w[t] for t in 1:T)
    for t in 1:T
        @constraint(model, T * w[t] >= Float64(delta_inf) * total_contracts)
        @constraint(model, T * w[t] <= Float64(delta_sup) * total_contracts)
    end

    scenario_cost_expr = Vector{Any}(undef, S)
    for s in 1:S
        scenario_cost_expr[s] = sum(contract_costs[t] * w[t] +
            pld[s, t] * (loads[t] - w[t]) for t in 1:T)
        @constraint(model, xi[s] >= scenario_cost_expr[s] - eta)
    end

    fixed_objective = beta_value * eta +
        beta_value / (1.0 - alpha_value) * sum(probs[s] * xi[s] for s in 1:S)

    function solve(c::AbstractVector{<:Real})
        c_value = Vector{Float64}(c)
        length(c_value) == T || error("c must have one PLD value per month")
        all(isfinite, c_value) || error("c must be finite")

        objective = fixed_objective +
            (1.0 - beta_value) * sum(contract_costs[t] * w[t] +
                c_value[t] * (e_plus[t] - e_minus[t]) for t in 1:T)
        set_objective_function(model, objective)
        set_objective_sense(model, MOI.MIN_SENSE)
        optimize!(model)

        status = termination_status(model)
        if status != MOI.OPTIMAL
            return EnergyPortfolioOracleResult(NaN, Float64[], Float64[], Float64[],
                NaN, Float64[], Float64[], Float64[], NaN, NaN, status)
        end

        w_value = value.(w)
        e_plus_value = value.(e_plus)
        e_minus_value = value.(e_minus)
        eta_value = value(eta)
        xi_value = value.(xi)
        scenario_costs = [sum(contract_costs[t] * w_value[t] +
            pld[s, t] * (loads[t] - w_value[t]) for t in 1:T) for s in 1:S]
        expected_cost = dot(probs, scenario_costs)
        cvar_value = eta_value + dot(probs, xi_value) / (1.0 - alpha_value)
        action = e_plus_value - e_minus_value

        return EnergyPortfolioOracleResult(objective_value(model), w_value,
            e_plus_value, e_minus_value, eta_value, xi_value, action,
            scenario_costs, expected_cost, cvar_value, status)
    end

    return EnergyPortfolioOracle(solve, T, S, beta_value)
end

(oracle::EnergyPortfolioOracle)(c::AbstractVector{<:Real}) = oracle.solve(c)

"""Return the cost-conjugate action `e_plus - e_minus` of an oracle result."""
energy_cost_conjugate(result::EnergyPortfolioOracleResult) = result.action

"""Return the affine objective-cost vector theta(c) for the energy oracle."""
function energy_theta(c::AbstractVector{<:Real}, contract_prices::AbstractVector{<:Real},
    probabilities::AbstractVector{<:Real}, alpha::Real, beta::Real)
    T = length(c)
    length(contract_prices) == T || error("contract_prices and c dimensions mismatch")
    S = length(probabilities)
    fixed = vcat((1.0 - beta) .* contract_prices,
        zeros(2 * T), beta, beta .* probabilities ./ (1.0 - alpha))
    variable = vcat(zeros(T), (1.0 - beta) .* c, -(1.0 - beta) .* c,
        zeros(1 + S))
    return fixed + variable
end

"""Compute the RU CVaR of a finite scenario-cost vector."""
function empirical_cvar(costs::AbstractVector{<:Real}, probabilities::AbstractVector{<:Real}, alpha::Real)
    length(costs) == length(probabilities) || error("cost/probability dimensions mismatch")
    probs = Vector{Float64}(probabilities)
    values = Vector{Float64}(costs)
    all(probs .>= 0) || error("probabilities must be nonnegative")
    isapprox(sum(probs), 1.0; atol = 1e-10) || error("probabilities must sum to one")
    0 < alpha < 1 || error("alpha must be strictly between zero and one")

    candidates = sort(unique(values))
    return minimum(eta + dot(probs, max.(values .- eta, 0.0)) / (1.0 - alpha)
        for eta in candidates)
end

"""Compute the SPO+ prediction gradient for one energy instance."""
function energy_spo_gradient(true_result::EnergyPortfolioOracleResult,
    auxiliary_result::EnergyPortfolioOracleResult, beta::Real)
    0 <= beta <= 1 || error("beta must be between zero and one")
    return 2.0 * (1.0 - Float64(beta)) .* (true_result.action - auxiliary_result.action)
end
