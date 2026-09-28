#=
Multi-output predictors for annual PLD vectors.
=#

using LinearAlgebra

struct EnergyRidgePredictor
    B::Matrix{Float64}
end

function fit_energy_ridge(X::AbstractMatrix{<:Real}, c::AbstractMatrix{<:Real}, lambda::Real)
    p, n = size(X)
    d, n_c = size(c)
    n == n_c || error("X and c must have the same number of samples")
    lambda >= 0 || error("lambda must be nonnegative")
    X_value = Matrix{Float64}(X)
    c_value = Matrix{Float64}(c)
    gram = X_value * X_value' + n * Float64(lambda) * Matrix{Float64}(I, p, p)
    return EnergyRidgePredictor((gram \ (X_value * c_value'))')
end

predict_energy(model::EnergyRidgePredictor, X::AbstractMatrix{<:Real}) = model.B * Matrix{Float64}(X)
predict_energy(model::EnergyRidgePredictor, x::AbstractVector{<:Real}) = model.B * Vector{Float64}(x)

function energy_mse_loss(model::EnergyRidgePredictor, X::AbstractMatrix{<:Real}, c::AbstractMatrix{<:Real})
    residuals = predict_energy(model, X) - Matrix{Float64}(c)
    return 0.5 * sum(abs2, residuals) / size(X, 2)
end
