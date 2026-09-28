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
end

function energy_spo_gradient(true_result, auxiliary_result, beta::Real)
    return 2.0 * (1.0 - Float64(beta)) .* (true_result.action - auxiliary_result.action)
end

function train_energy_spo(X::AbstractMatrix{<:Real}, c::AbstractMatrix{<:Real},
    oracle_builders::AbstractVector, beta::Real;
    lambda::Real = 0.0,
    numiter::Integer = 100,
    batchsize::Integer = size(X, 2),
    step_size::Real = 0.01,
    B_init = zeros(size(c, 1), size(X, 1)),
    cost_mean::AbstractVector{<:Real} = zeros(size(c, 1)),
    cost_scale::AbstractVector{<:Real} = ones(size(c, 1)),
    objective_function = nothing,
    epoch_callback = nothing,
    rng = Random.default_rng())

    p, n = size(X)
    d, n_c = size(c)
    n == n_c || error("X and c must have the same number of samples")
    length(oracle_builders) == n || error("one oracle builder is required per sample")
    numiter > 0 || error("numiter must be positive")
    batchsize > 0 || error("batchsize must be positive")
    size(B_init) == (d, p) || error("B_init has invalid dimensions")
    length(cost_mean) == d || error("cost_mean has invalid dimensions")
    length(cost_scale) == d || error("cost_scale has invalid dimensions")
    all(cost_scale .> 0) || error("cost_scale must be positive")
    0 <= beta <= 1 || error("beta must be between zero and one")

    B = Matrix{Float64}(B_init)
    objective_history = Float64[]
    gradient_norm_history = Float64[]

    for epoch in 1:numiter
        indices = batchsize >= n ? collect(1:n) : rand(rng, 1:n, batchsize)
        gradient = zeros(d, p)
        for i in indices
            c_i_normalized = Vector{Float64}(c[:, i])
            c_hat_normalized = B * Vector{Float64}(X[:, i])
            c_i = c_i_normalized .* Vector{Float64}(cost_scale) .+ Vector{Float64}(cost_mean)
            c_hat = c_hat_normalized .* Vector{Float64}(cost_scale) .+ Vector{Float64}(cost_mean)
            true_result = oracle_builders[i](c_i)
            auxiliary_result = oracle_builders[i](2.0 .* c_hat .- c_i)
            gradient_c = energy_spo_gradient(true_result, auxiliary_result, beta)
            gradient_normalized = gradient_c .* Vector{Float64}(cost_scale)
            gradient += gradient_normalized * Vector{Float64}(X[:, i])'
        end
        gradient ./= length(indices)
        gradient += Float64(lambda) .* B
        B -= Float64(step_size) .* gradient
        objective = objective_function === nothing ? NaN : objective_function(B)
        push!(objective_history, objective)
        push!(gradient_norm_history, norm(gradient))
        if epoch_callback !== nothing
            epoch_callback(epoch, B, objective, norm(gradient))
        end
    end

    return EnergyRidgePredictor(B), EnergySPOHistory(objective_history, gradient_norm_history)
end
