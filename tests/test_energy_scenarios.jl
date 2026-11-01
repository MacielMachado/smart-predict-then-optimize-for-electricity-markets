include("../solver/energy_scenarios.jl")

using Test

@testset "Energy scenario interfaces" begin
    X = [1.0 2.0; 3.0 4.0]
    c = [10.0 11.0; 20.0 21.0]
    trees = [c, c .+ 1.0]
    probabilities = [0.5, 0.5]

    batch = make_energy_scenario_batch(X, c, trees, probabilities)
    @test batch.X == X
    @test batch.c == c
    @test length(batch.scenario_trees) == 2
    @test validate_energy_scenarios(c, trees, probabilities)

    annual_costs = [1.0 2.0 3.0; 4.0 5.0 6.0]
    left_out = leave_one_out_scenario_tree(annual_costs, 2)
    @test left_out == [1.0 3.0; 4.0 6.0]

    dataset = load_energy_dataset()
    @test size(dataset.X, 1) == 12
    @test size(dataset.c) == size(dataset.X)
    @test length(dataset.scenario_trees) == size(dataset.X, 2)
    @test all(size(tree) == (4, 12) for tree in dataset.scenario_trees)
    @test all(dataset.years .!= 2026)
    splits = energy_split_indices(dataset)
    @test all(dataset.years[splits.test] .∈ Ref([2006, 2019]))
    @test all(dataset.years[splits.validation] .∈ Ref([2007, 2018]))

    normalization = fit_energy_normalization(dataset.X[:, splits.train], dataset.c[:, splits.train])
    X_normalized, c_normalized = normalize_energy_data(dataset.X[:, splits.train],
        dataset.c[:, splits.train], normalization)
    @test vec(mean(X_normalized, dims = 2)) ≈ zeros(12) atol = 1e-10
    @test vec(mean(c_normalized, dims = 2)) ≈ zeros(12) atol = 1e-10
end
