include("../solver/energy_predictors.jl")
include("../oracles/energy_portfolio_oracle.jl")
include("../solver/energy_spo.jl")

using Test

@testset "Energy predictors and SPO+" begin
    X = [1.0 1.0 1.0; 0.0 1.0 2.0]
    c = [2.0 3.0 4.0; 1.0 3.0 5.0]
    model = fit_energy_ridge(X, c, 1e-8)
    predictions = predict_energy(model, X)
    @test size(predictions) == size(c)
    @test energy_mse_loss(model, X, c) < 1e-8

    demand = [10.0, 10.0]
    contract_prices = [2.0, 2.0]
    scenarios = [1.0 2.0; 3.0 4.0]
    probabilities = [0.5, 0.5]
    builders = [build_energy_oracle(scenarios, probabilities, demand, contract_prices;
        alpha = 0.5, beta = 0.25, gamma = 0.5,
        delta_inf = 0.9, delta_sup = 1.1,
        spot_upper_bound = demand) for _ in 1:size(X, 2)]

    B, history = train_energy_spo(X, c, builders, 0.25;
        numiter = 2, batchsize = size(X, 2), step_size = 1e-3)
    @test size(B.B) == (size(c, 1), size(X, 1))
    @test length(history.gradient_norm_history) == 2

    c_hat = [2.5, 3.5]
    theta_c = energy_theta(c[:, 1], contract_prices, probabilities, 0.5, 0.25)
    theta_hat = energy_theta(c_hat, contract_prices, probabilities, 0.5, 0.25)
    theta_aux = energy_theta(2.0 .* c_hat .- c[:, 1], contract_prices, probabilities, 0.5, 0.25)
    @test 2.0 .* theta_hat .- theta_c ≈ theta_aux
end
