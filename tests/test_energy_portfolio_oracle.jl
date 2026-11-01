include("../oracles/energy_portfolio_oracle.jl")

using Test
const MOI = JuMP.MOI

@testset "Energy portfolio oracle" begin
    demand = [10.0, 12.0]
    contract_prices = [2.0, 2.0]
    scenarios = [1.0 3.0; 2.0 4.0; 5.0 1.0]
    probabilities = fill(1.0 / 3.0, 3)

    oracle = build_energy_oracle(scenarios, probabilities, demand, contract_prices;
        alpha = 0.5, beta = 0.4, gamma = 0.5,
        delta_inf = 0.9, delta_sup = 1.1,
        spot_upper_bound = demand)

    true_result = oracle([2.0, 2.5])
    auxiliary_result = oracle([3.0, 1.0])

    @test true_result.status == MOI.OPTIMAL
    @test auxiliary_result.status == MOI.OPTIMAL
    @test length(true_result.w) == 2
    @test length(true_result.action) == 2
    @test all(true_result.w .>= 0)
    @test all(true_result.e_plus .>= -1e-7)
    @test all(true_result.e_minus .>= -1e-7)
    @test all(abs.(true_result.w .+ true_result.e_plus .- true_result.e_minus .- demand) .<= 1e-6)
    @test all(true_result.w .>= 0.5 .* demand .- 1e-6)
    @test true_result.cvar ≈ empirical_cvar(true_result.scenario_costs, probabilities, 0.5) atol = 1e-6
    @test true_result.action ≈ true_result.e_plus - true_result.e_minus

    beta_one_oracle = build_energy_oracle(scenarios, probabilities, demand, contract_prices;
        alpha = 0.5, beta = 1.0, gamma = 0.5,
        delta_inf = 0.9, delta_sup = 1.1,
        spot_upper_bound = demand)
    beta_one_true = beta_one_oracle([1.0, 2.0])
    beta_one_auxiliary = beta_one_oracle([100.0, -100.0])
    @test energy_spo_gradient(beta_one_true, beta_one_auxiliary, 1.0) ≈ zeros(2) atol = 1e-8

    beta_zero_gradient = energy_spo_gradient(true_result, auxiliary_result, 0.0)
    @test beta_zero_gradient ≈ 2.0 .* (true_result.action - auxiliary_result.action)
end
