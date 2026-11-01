include("../experiments/energy_portfolio_run.jl")
include("../solver/energy_cross_validation.jl")

using Test

@testset "Energy temporal cross-validation folds" begin
    dataset = load_energy_dataset()
    loo = build_energy_year_folds(dataset; evaluation_mode = :leave_one_year_out,
        first_test_year = 2006, last_test_year = 2006)
    @test length(loo) == 1
    @test loo[1].test_year == 2006
    @test loo[1].validation_year == 2005
    @test all(dataset.years[loo[1].train_indices] .!= 2005)
    @test all(dataset.years[loo[1].train_indices] .!= 2006)
    @test length(loo[1].validation_indices) == 4
    @test length(loo[1].test_indices) == 4

    rolling = build_energy_year_folds(dataset; evaluation_mode = :rolling_origin,
        first_test_year = 2007, last_test_year = 2007)
    @test length(rolling) == 1
    @test rolling[1].test_year == 2007
    @test rolling[1].validation_year == 2006
    @test maximum(dataset.years[rolling[1].train_indices]) < 2006
    @test all(dataset.years[rolling[1].validation_indices] .== 2006)
    @test all(dataset.years[rolling[1].test_indices] .== 2007)
end
