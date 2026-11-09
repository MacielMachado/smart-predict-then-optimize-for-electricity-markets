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
    best_epoch::Int                       # epoch of the returned model (0 = initial point; numiter without selection)
    best_score::Float64                   # validation score of the returned model (Inf without selection)
end

function energy_spo_gradient(true_result, auxiliary_result, beta::Real)
    return 2.0 * (1.0 - Float64(beta)) .* (true_result.action - auxiliary_result.action)
end

"""
    train_energy_spo(X, c, oracle_builders, beta; ...)

Stochastic subgradient method on the SPO+ loss with ridge penalty `lambda`. One "epoch" is one
subgradient step on a minibatch of `batchsize` samples drawn with replacement.

Step rules (`step_rule`):
- `:normalized_decaying` (default): η_t = step_size / (‖g_t‖ √t), so every step moves B by
  step_size/√t in Frobenius norm whatever the scale of the subgradient (the
  `long_dynamic_normalized` rule of the original SPO code);
- `:constant`: η_t = step_size.

`averaging = true` returns (and evaluates) the running mean of the iterates,
B̄_t = (B_1 + ... + B_t) / t, instead of the iterate B_t itself (Polyak averaging); it smooths the
zigzag of the subgradient steps and keeps the model closer to the starting point.

`cost_offset` (d x n, optional) is added to every denormalized cost vector of sample i: the predictor
then learns deviations from a per-sample anchor (e.g. the seasonal historical mean), and
c_i = c_normalized * cost_scale + cost_mean + cost_offset[:, i] (same for the prediction). The
subgradient is unchanged because the offset does not depend on B.

Every `eval_every` epochs (and at the last one) `objective_function(B)` is recorded and
`epoch_callback(epoch, B, objective, gradient_norm)` is called with the current model (the average
when `averaging = true`). If the callback returns a finite number, it is treated as a validation
score (lower is better) and the model with the best score is returned. With
`evaluate_initial = true` the starting point B_init is also scored (epoch 0), so the selection may
keep it unchanged. Without finite scores the last model is returned (best_epoch = numiter).
"""
function train_energy_spo(X::AbstractMatrix{<:Real}, c::AbstractMatrix{<:Real},
    oracle_builders::AbstractVector, beta::Real;
    lambda::Real = 0.0,
    numiter::Integer = 100,
    batchsize::Integer = size(X, 2),
    step_size::Real = 0.1,
    step_rule::Symbol = :normalized_decaying,
    averaging::Bool = false,
    B_init = zeros(size(c, 1), size(X, 1)),
    cost_mean::AbstractVector{<:Real} = zeros(size(c, 1)),
    cost_scale::AbstractVector{<:Real} = ones(size(c, 1)),
    cost_offset = nothing,
    objective_function = nothing,
    epoch_callback = nothing,
    evaluate_initial::Bool = false,
    eval_every::Integer = 1,
    rng = Random.default_rng())

    p, n = size(X)
    d, n_c = size(c)
    n == n_c || error("X and c must have the same number of samples")
    length(oracle_builders) == n || error("one oracle builder is required per sample")
    numiter >= 0 || error("numiter must be nonnegative")
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
    scale = Vector{Float64}(cost_scale)
    mean_cost = Vector{Float64}(cost_mean)

    B = Matrix{Float64}(B_init)
    B_average = copy(B)
    objective_history = Float64[]
    gradient_norm_history = Float64[]
    recorded_epochs = Int[]
    best_B = copy(B)
    best_epoch = 0
    best_score = Inf

    function record!(epoch, model, gradient_norm)
        objective = objective_function === nothing ? NaN : objective_function(model)
        push!(objective_history, objective)
        push!(gradient_norm_history, gradient_norm)
        push!(recorded_epochs, epoch)
        epoch_callback === nothing && return
        score = epoch_callback(epoch, model, objective, gradient_norm)
        if score isa Real && isfinite(score) && score < best_score
            best_score = Float64(score)
            best_B = copy(model)
            best_epoch = epoch
        end
    end

    evaluate_initial && record!(0, B, NaN)

    for epoch in 1:numiter
        indices = batchsize >= n ? collect(1:n) : rand(rng, 1:n, batchsize)
        gradient = zeros(d, p)
        for i in indices
            x_i = Vector{Float64}(X[:, i])
            c_i = Vector{Float64}(c[:, i]) .* scale .+ mean_cost .+ offset[:, i]
            c_hat = (B * x_i) .* scale .+ mean_cost .+ offset[:, i]
            true_result = oracle_builders[i](c_i)
            auxiliary_result = oracle_builders[i](2.0 .* c_hat .- c_i)
            gradient_c = energy_spo_gradient(true_result, auxiliary_result, beta)
            gradient += (gradient_c .* scale) * x_i'
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
        B_average .+= (B .- B_average) ./ epoch          # mean of B_1, ..., B_epoch

        if epoch % eval_every == 0 || epoch == numiter
            record!(epoch, averaging ? B_average : B, gradient_norm)
        end
    end

    last_model = numiter == 0 ? Matrix{Float64}(B_init) : (averaging ? B_average : B)
    if isfinite(best_score)
        return EnergyRidgePredictor(best_B), EnergySPOHistory(objective_history,
            gradient_norm_history, recorded_epochs, best_epoch, best_score)
    end
    return EnergyRidgePredictor(copy(last_model)), EnergySPOHistory(objective_history,
        gradient_norm_history, recorded_epochs, numiter, Inf)
end
