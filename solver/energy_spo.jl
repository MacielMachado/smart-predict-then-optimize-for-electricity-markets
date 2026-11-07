#=
SPO+ training for the energy oracle.

This module is deliberately independent from the legacy `solver/sgd.jl`
interface because the energy oracle returns an extended decision and exposes
its cost-conjugate action separately.
=#

using LinearAlgebra, Random

struct EnergySPOHistory
    objective_history::Vector{Float64}
    gradient_norm_history::Vector{Float64}
    epochs::Vector{Int}                   # epochs at which the histories were recorded
    best_epoch::Int                       # epoch selected by the validation score (0 = last epoch)
    best_score::Float64
end

function energy_spo_gradient(true_result, auxiliary_result, beta::Real)
    return 2.0 * (1.0 - Float64(beta)) .* (true_result.action - auxiliary_result.action)
end

"""
    train_energy_spo(X, c, oracle_builders, beta; ...)

Stochastic subgradient method on the SPO+ loss with ridge penalty `lambda`.

Step rules (`step_rule`):
- `:normalized_decaying` (default): η_t = step_size / (‖g_t‖ √t), so every step moves B by
  step_size/√t in Frobenius norm whatever the scale of the subgradient (the
  `long_dynamic_normalized` rule of the original SPO code);
- `:constant`: η_t = step_size.

`cost_offset` (d x n, optional) is added to every denormalized cost vector of sample i: the predictor
then learns deviations from a per-sample anchor (e.g. the seasonal historical mean), and
c_i = c_normalized * cost_scale + cost_mean + cost_offset[:, i] (same for the prediction). The
subgradient is unchanged because the offset does not depend on B.

Every `eval_every` epochs (and at the last one) `objective_function(B)` is recorded and
`epoch_callback(epoch, B, objective, gradient_norm)` is called. If the callback returns a finite
number, it is treated as a validation score (lower is better) and the B with the best score is
returned instead of the last iterate.
"""
function train_energy_spo(X::AbstractMatrix{<:Real}, c::AbstractMatrix{<:Real},
    oracle_builders::AbstractVector, beta::Real;
    lambda::Real = 0.0,
    numiter::Integer = 100,
    batchsize::Integer = size(X, 2),
    step_size::Real = 0.1,
    step_rule::Symbol = :normalized_decaying,
    B_init = zeros(size(c, 1), size(X, 1)),
    cost_mean::AbstractVector{<:Real} = zeros(size(c, 1)),
    cost_scale::AbstractVector{<:Real} = ones(size(c, 1)),
    cost_offset = nothing,
    objective_function = nothing,
    epoch_callback = nothing,
    eval_every::Integer = 1,
    rng = Random.default_rng())

    p, n = size(X)
    d, n_c = size(c)
    n == n_c || error("X and c must have the same number of samples")
    length(oracle_builders) == n || error("one oracle builder is required per sample")
    numiter > 0 || error("numiter must be positive")
    batchsize > 0 || error("batchsize must be positive")
    eval_every > 0 || error("eval_every must be positive")
    step_rule in (:normalized_decaying, :constant) || error("unknown step_rule $step_rule")
    size(B_init) == (d, p) || error("B_init has invalid dimensions")
    length(cost_mean) == d || error("cost_mean has invalid dimensions")
    length(cost_scale) == d || error("cost_scale has invalid dimensions")
    all(cost_scale .> 0) || error("cost_scale must be positive")
    0 <= beta <= 1 || error("beta must be between zero and one")
    offset = cost_offset === nothing ? zeros(d, n) : Matrix{Float64}(cost_offset)
    size(offset) == (d, n) || error("cost_offset has invalid dimensions")

    B = Matrix{Float64}(B_init)
    objective_history = Float64[]
    gradient_norm_history = Float64[]
    recorded_epochs = Int[]
    best_B = copy(B)
    best_epoch = 0
    best_score = Inf

    for epoch in 1:numiter
        indices = batchsize >= n ? collect(1:n) : rand(rng, 1:n, batchsize)
        gradient = zeros(d, p)
        for i in indices
            c_i_normalized = Vector{Float64}(c[:, i])
            c_hat_normalized = B * Vector{Float64}(X[:, i])
            c_i = c_i_normalized .* Vector{Float64}(cost_scale) .+ Vector{Float64}(cost_mean) .+ offset[:, i]
            c_hat = c_hat_normalized .* Vector{Float64}(cost_scale) .+ Vector{Float64}(cost_mean) .+ offset[:, i]
            true_result = oracle_builders[i](c_i)
            auxiliary_result = oracle_builders[i](2.0 .* c_hat .- c_i)
            gradient_c = energy_spo_gradient(true_result, auxiliary_result, beta)
            gradient_normalized = gradient_c .* Vector{Float64}(cost_scale)
            gradient += gradient_normalized * Vector{Float64}(X[:, i])'
        end
        gradient ./= length(indices)
        gradient += Float64(lambda) .* B
        gradient_norm = norm(gradient)
        step = if step_rule == :constant
            Float64(step_size)
        else
            gradient_norm > 0 ? Float64(step_size) / (gradient_norm * sqrt(epoch)) : 0.0
        end
        B -= step .* gradient

        if epoch % eval_every == 0 || epoch == numiter
            objective = objective_function === nothing ? NaN : objective_function(B)
            push!(objective_history, objective)
            push!(gradient_norm_history, gradient_norm)
            push!(recorded_epochs, epoch)
            if epoch_callback !== nothing
                score = epoch_callback(epoch, B, objective, gradient_norm)
                if score isa Real && isfinite(score) && score < best_score
                    best_score = Float64(score)
                    best_B = copy(B)
                    best_epoch = epoch
                end
            end
        end
    end

    final_B = best_epoch > 0 ? best_B : B
    return EnergyRidgePredictor(final_B), EnergySPOHistory(objective_history,
        gradient_norm_history, recorded_epochs, best_epoch, best_score)
end
